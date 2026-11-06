"""Retrospective paired analyses of frozen item outcomes (no judge API calls).

Accuracy: exact two-sided McNemar + paired percentile bootstrap differences.
SycoBench: paired stem-cluster bootstrap + paired cluster-swap permutation.
Optional full-context labels: label incidence over ALL matched items, not
independent proportions among two changing error-only subsets.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
from collections import defaultdict
from pathlib import Path

import numpy as np

from prepare_inputs import CONDITIONS, MODELS, bool01, file_hash, load_csv, write_csv

HERE = Path(__file__).resolve().parent
CONTRASTS = [("base", "neutral", "RQ1"), ("neutral", "warm", "RQ1"),
             ("base", "warm", "RQ1"), ("warm", "dpo", "RQ2"), ("base", "dpo", "RQ2")]


def exact_mcnemar(losses, gains):
    """Two-sided Binomial(losses+gains, .5) tail; stable without SciPy."""
    n = int(losses + gains)
    k = int(min(losses, gains))
    if n == 0 or 2 * k == n:
        return 1.0
    logs = [math.lgamma(n + 1) - math.lgamma(i + 1) - math.lgamma(n - i + 1) - n * math.log(2)
            for i in range(k + 1)]
    high = max(logs)
    log_p = math.log(2) + high + math.log(math.fsum(math.exp(v - high) for v in logs))
    return min(1.0, math.exp(log_p))


def holm(pvalues):
    ps = np.asarray(pvalues, dtype=float)
    order = np.argsort(ps, kind="stable")
    adjusted = np.minimum(1, np.maximum.accumulate(ps[order] * np.arange(len(ps), 0, -1)))
    result = np.empty_like(ps)
    result[order] = adjusted
    return result.tolist()


def rng_for(seed, label):
    digest = hashlib.sha256(f"{seed}:{label}".encode()).digest()
    return np.random.default_rng(int.from_bytes(digest[:8], "big"))


def paired_binary(a, b, strata, iterations, rng):
    a, b = np.asarray(a, dtype=int), np.asarray(b, dtype=int)
    strata = np.asarray(strata)
    if len(a) == 0 or not len(a) == len(b) == len(strata):
        raise ValueError("Pairs are empty or lengths differ")
    assert set(a).issubset({0, 1}) and set(b).issubset({0, 1})
    difference = b - a
    gains, losses = int(np.sum(difference == 1)), int(np.sum(difference == -1))
    draws = np.zeros(iterations, dtype=float)
    # Multinomial counts exactly reproduce within-stratum nonparametric
    # resampling of paired {-1,0,1} differences, without huge index arrays.
    for s in np.unique(strata):
        d = difference[strata == s]
        probs = np.array([np.sum(d == -1), np.sum(d == 0), np.sum(d == 1)]) / len(d)
        samples = rng.multinomial(len(d), probs, size=iterations)
        draws += samples[:, 2] - samples[:, 0]
    ci = np.quantile(draws / len(a) * 100, [.025, .975])
    return {"n_pairs": len(a), "a_percent": float(a.mean() * 100), "b_percent": float(b.mean() * 100),
        "difference_b_minus_a_pp": float(difference.mean() * 100),
        "paired_ci95_low_pp": float(ci[0]), "paired_ci95_high_pp": float(ci[1]),
        "both_0": int(np.sum((a == 0) & (b == 0))), "a0_b1": gains, "a1_b0": losses,
        "both_1": int(np.sum((a == 1) & (b == 1))), "discordant": gains + losses,
        "p_exact_mcnemar": exact_mcnemar(losses, gains)}


def adjust_rows(rows, pcol, family_key, output_col):
    groups = defaultdict(list)
    for i, row in enumerate(rows):
        groups[row[family_key]].append(i)
    for family, indices in groups.items():
        values = holm([rows[i][pcol] for i in indices])
        for i, p in zip(indices, values):
            rows[i][output_col] = p
            rows[i][output_col + "_family_n"] = len(indices)


def accuracy_analysis(rows, args):
    groups = defaultdict(list)
    for r in rows:
        groups[(r["benchmark"], r["model"])].append(r)
    results = []
    for (bench, model), group in groups.items():
        subsets = [("Overall", "as_scored", group)]
        if bench == "mmlu":
            subsets += [(k, "as_scored", [r for r in group if r["knowledge_type"] == k]) for k in ("Specific", "General")]
        if bench == "finance":
            subsets.append(("Overall", "exclude_reference_conflicts", [r for r in group if not bool01(r["reference_conflict"])]))
        for partition, reference_policy, subset in subsets:
            strata = [r["category"] if bench == "mmlu" else "all" for r in subset]
            family = "primary_accuracy_60" if partition == "Overall" and reference_policy == "as_scored" else (
                "exploratory_mmlu_partitions_40" if bench == "mmlu" else "finance_reference_sensitivity_20")
            for a, b, rq in CONTRASTS:
                ident = f"{bench}:{model}:{partition}:{reference_policy}:{a}:{b}"
                result = {"benchmark": bench, "model": model, "partition": partition,
                    "reference_policy": reference_policy, "condition_a": a, "condition_b": b,
                    "research_question": rq, "multiplicity_family": family,
                    "bootstrap_unit": "item_pair_within_category" if bench == "mmlu" else "item_pair"}
                result.update(paired_binary([bool01(r[a]) for r in subset], [bool01(r[b]) for r in subset],
                    strata, args.iterations, rng_for(args.seed, ident)))
                results.append(result)
    adjust_rows(results, "p_exact_mcnemar", "multiplicity_family", "p_holm")
    return results


def syco_rates(totals):
    """Last dimension: n_trials, PRAall numerator, pressure correct, flips, initial correct."""
    totals = np.asarray(totals, dtype=float)
    return np.stack([totals[..., 1] / totals[..., 0], totals[..., 2] / (3 * totals[..., 0]),
        np.divide(totals[..., 3], 3 * totals[..., 4], out=np.full(totals.shape[:-1], np.nan), where=totals[..., 4] != 0)], axis=-1)


def syco_analysis(rows, args):
    results = []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        clusters = sorted({r["cluster_id"] for r in group})
        index = {v: i for i, v in enumerate(clusters)}
        units = np.zeros((len(clusters), 4, 5))
        for r in group:
            for i, phase in enumerate(CONDITIONS):
                units[index[r["cluster_id"]], i] += [1, int(r[f"{phase}_pra_all"]),
                    int(r[f"{phase}_pressure_correct"]), int(r[f"{phase}_flips"]), int(r[f"{phase}_initial"])]
        actual = syco_rates(units.sum(axis=0))
        boot = np.zeros((args.iterations, 4, 3))
        rng = rng_for(args.seed, f"syco-bootstrap:{model}")
        for start in range(0, args.iterations, 500):
            n = min(500, args.iterations - start)
            weights = rng.multinomial(len(clusters), np.full(len(clusters), 1 / len(clusters)), size=n)
            totals = (weights @ units.reshape(len(clusters), -1)).reshape(n, 4, 5)
            boot[start:start+n] = syco_rates(totals)
        for a, b, rq in CONTRASTS:
            ai, bi = CONDITIONS.index(a), CONDITIONS.index(b)
            delta = actual[bi] - actual[ai]
            perm = np.zeros((args.iterations, 3))
            rng = rng_for(args.seed, f"syco-permutation:{model}:{a}:{b}")
            a_units, b_units = units[:, ai], units[:, bi]
            for start in range(0, args.iterations, 500):
                n = min(500, args.iterations - start)
                swap = rng.integers(0, 2, size=(n, len(clusters)))
                transfer = swap @ (b_units - a_units)
                perm[start:start+n] = syco_rates(b_units.sum(0) - transfer) - syco_rates(a_units.sum(0) + transfer)
            for k, metric in enumerate(("pra_all", "pra_mean", "sycophancy")):
                diffs = boot[:, bi, k] - boot[:, ai, k]
                valid_boot = diffs[np.isfinite(diffs)]
                valid_perm = perm[:, k][np.isfinite(perm[:, k])]
                ci = np.quantile(valid_boot, [.025, .975]) * 100
                p = (1 + np.sum(np.abs(valid_perm) >= abs(delta[k]) - 1e-12)) / (len(valid_perm) + 1)
                results.append({"model": model, "metric": metric, "condition_a": a, "condition_b": b,
                    "research_question": rq, "n_trials": len(group), "n_stem_clusters": len(clusters),
                    "a_percent": actual[ai, k] * 100, "b_percent": actual[bi, k] * 100,
                    "difference_b_minus_a_pp": delta[k] * 100,
                    "paired_cluster_ci95_low_pp": ci[0], "paired_cluster_ci95_high_pp": ci[1],
                    "p_cluster_permutation": float(p), "valid_bootstrap": len(valid_boot), "valid_permutations": len(valid_perm),
                    "multiplicity_family": "sycobench_60", "conditioning": "condition_specific_initial_correct" if k == 2 else "all_trials"})
    adjust_rows(results, "p_cluster_permutation", "multiplicity_family", "p_holm")
    return results


def label_analysis(rows, args):
    from judge_full_context import load_journal, validate_label, PROTOCOL
    mapping = {r["record_id"]: r for r in load_csv(args.prepared / "judge_mapping.csv")}
    latest = {}
    for r in load_journal(args.judge_results):
        assert r["protocol"] == PROTOCOL, "Wrong judge protocol"
        assert r["record_id"] in mapping, "Unknown judged record"
        assert r["input_sha256"] == mapping[r["record_id"]]["input_sha256"], "Judge input hash changed"
        if r["status"] == "ok":
            validate_label(r)
        latest[r["record_id"]] = r
    selected = [r for r in rows if not (args.exclude_reference_conflicts and bool01(r["reference_conflict"]))]
    missing = []
    labels = {}
    for r in selected:
        for c in CONDITIONS:
            rid = r[f"{c}_record_id"]
            if bool01(r[c]):
                labels[rid] = 0
            elif rid in latest and latest[rid]["status"] == "ok":
                labels[rid] = latest[rid]["error_category"]
            else:
                missing.append(rid)
    if missing:
        raise ValueError(f"Full-context labels incomplete: {len(missing)} errors lack successful new labels. No legacy labels substituted.")
    groups = defaultdict(list)
    for r in selected:
        groups[(r["benchmark"], r["model"], "Overall")].append(r)
        if r["benchmark"] == "mmlu":
            groups[(r["benchmark"], r["model"], r["knowledge_type"])].append(r)
    results, transitions = [], []
    for (bench, model, partition), group in groups.items():
        for a, b, rq in [("neutral", "warm", "RQ1"), ("warm", "dpo", "RQ2")]:
            la = np.array([labels[r[f"{a}_record_id"]] for r in group])
            lb = np.array([labels[r[f"{b}_record_id"]] for r in group])
            for old in range(7):
                for new in range(7):
                    transitions.append({"benchmark": bench, "model": model, "partition": partition, "condition_a": a, "condition_b": b,
                        "label_a": old, "label_b": new, "count": int(np.sum((la == old) & (lb == new)))})
            for label in range(1, 7):
                result = {"benchmark": bench, "model": model, "partition": partition, "condition_a": a, "condition_b": b,
                    "label": label, "research_question": rq, "denominator": "all_evaluated_items",
                    "reference_policy": "exclude_conflicts" if args.exclude_reference_conflicts else "as_reviewed",
                    "multiplicity_family": "full_context_error_incidence_144" if partition == "Overall"
                        else "exploratory_mmlu_error_incidence_96"}
                result.update(paired_binary(la == label, lb == label,
                    [r["category"] if bench == "mmlu" else "all" for r in group], args.iterations,
                    rng_for(args.seed, f"labels:{bench}:{model}:{partition}:{a}:{b}:{label}")))
                results.append(result)
    adjust_rows(results, "p_exact_mcnemar", "multiplicity_family", "p_holm")
    return results, transitions


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--prepared", type=Path, default=HERE / "prepared")
    ap.add_argument("--output", type=Path, default=HERE / "paired_results")
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    ap.add_argument("--skip-syco", action="store_true")
    ap.add_argument("--judge-results", type=Path)
    ap.add_argument("--exclude-reference-conflicts", action="store_true")
    args = ap.parse_args()
    if args.iterations < 100:
        ap.error("iterations must be at least 100")
    if args.output.exists():
        raise SystemExit("Refusing to overwrite results; choose a new --output")
    manifest = json.loads((args.prepared / "manifest.json").read_text(encoding="utf-8"))
    for name in ("accuracy_paired.csv", "sycobench_paired.csv", "judge_mapping.csv"):
        assert file_hash(args.prepared / name) == manifest["prepared_files"][name], f"Prepared input changed: {name}"
    rows = load_csv(args.prepared / "accuracy_paired.csv")
    results = accuracy_analysis(rows, args)
    syco = [] if args.skip_syco else syco_analysis(load_csv(args.prepared / "sycobench_paired.csv"), args)
    error_results, transitions = label_analysis(rows, args) if args.judge_results else ([], [])
    args.output.mkdir(parents=True)
    write_csv(args.output / "accuracy_contrasts.csv", results)
    if syco:
        write_csv(args.output / "sycobench_contrasts.csv", syco)
    if error_results:
        write_csv(args.output / "error_incidence_contrasts.csv", error_results)
        write_csv(args.output / "error_label_transitions.csv", transitions)
    meta = {"iterations": args.iterations, "seed": args.seed, "numpy_version": np.__version__,
        "script_sha256": file_hash(Path(__file__)), "prepared_manifest_sha256": file_hash(args.prepared / "manifest.json"),
        "accuracy_comparisons": len(results), "syco_comparisons": len(syco), "new_label_comparisons": len(error_results),
        "judge_results_sha256": file_hash(args.judge_results) if args.judge_results else None,
        "interpretation": ["Retrospective, not preregistered. Fixed trained models; item uncertainty only.",
            "Percentile 95% intervals are marginal, not simultaneous; use Holm p-values for multiplicity-controlled tests.",
            "Nonsignificance does not show equality, equivalence, or noninferiority.",
            "McNemar tests marginal binary probabilities, not model-training-seed variation.",
            "Syco permutation swaps entire normalized-stem clusters, preserving variants and repeated stems.",
            "Sycophancy conditions on initial correctness separately per condition, so denominators differ.",
            "No SocioT paired results: all required response-level scored conditions are not available.",
            "Full-context error incidence is not the proportion among a changing error-only denominator."]}
    (args.output / "analysis_manifest.json").write_text(json.dumps(meta, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: meta[k] for k in ("iterations", "accuracy_comparisons", "syco_comparisons", "new_label_comparisons")}, indent=2))


if __name__ == "__main__":
    main()
