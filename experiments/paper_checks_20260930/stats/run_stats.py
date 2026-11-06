"""Frozen-item McNemar and paired label-bootstrap analyses, 2026-09-30.

No judge calls. Historical labels are full-context labels as confirmed by the
researcher. Source files remain unchanged. Bootstrap units are question pairs,
stratified by the 14 MMLU-Pro categories. CI percentile, p centered-null.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import math
import re
import sys
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np
from scipy.stats import binomtest

HERE = Path(__file__).resolve().parent
ROOT = HERE.parents[2]
OLD = ROOT / "experiments/judge_paired_20260928"
sys.path.insert(0, str(OLD))
from paired_tests import exact_mcnemar, holm, paired_binary, rng_for

MODELS = ("qwen", "phi4", "llama", "gemma")
BENCHMARKS = ("mmlu", "finance", "medical")
CONDITIONS = ("base", "neutral", "warm", "dpo")
CONTRASTS = (("base", "neutral"), ("base", "warm"), ("neutral", "warm"))
DPO_CONTRASTS = (("warm", "dpo"), ("base", "dpo"))
LABELS = {1: "Knowledge Deficit", 2: "Logical Derailment",
          3: "Truncation/Length Overflow", 4: "Calculation Error",
          5: "Generation or Format Failure", 6: "Other"}
PLAN = {
    "status": "retrospective; frozen before computing this run's tests",
    "accuracy_primary": "36 overall SFT comparisons: 4 models x 3 benchmarks x 3 contrasts; Holm",
    "accuracy_dpo": "24 overall DPO comparisons: 4 models x 3 benchmarks x 2 contrasts; separate Holm",
    "labels_primary": "12 Warm-minus-Neutral LD count-difference tests on fixed matched N; Holm across model-benchmark cells. Incidence tests are mathematically identical scaled statistics.",
    "labels_kd": "12 Warm-minus-Neutral KD count-difference tests on fixed matched N; separate Holm supporting family",
    "labels_exploratory": "Remaining 192 incidence tests (all 6 labels and 3 SFT contrasts minus primary LD/KD); separate Holm",
    "labels_error_share": "Same families separately for secondary shares among each condition's errors; denominators resampled with numerators",
    "bootstrap": "20,000 paired draws, seed 42; MMLU within category; domain benchmarks item pairs",
    "bootstrap_test": "Two-sided centered-null bootstrap: (1 + count(|delta* - delta_hat| >= |delta_hat|))/(B + 1). Approximate, not an exact randomization test.",
    "interval": "Percentile 95% marginal interval for each paired difference, not simultaneous",
    "accuracy_test": "Exact two-sided McNemar (conditional binomial, p=0.5) on discordant item pairs",
    "label_denominator_primary": "Absolute error-label counts on fixed matched N, with all evaluated items eligible (including correct category 0). Incidence=count/N is technical secondary presentation.",
    "operational_failures": "10 historical failures retained in Other for primary analysis; no label regeneration",
}


def sha(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows):
    with path.open("w", encoding="utf-8-sig", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def adjust(rows, pcolumn="p_bootstrap_centered"):
    families = defaultdict(list)
    for index, row in enumerate(rows):
        families[row["multiplicity_family"]].append(index)
    for indices in families.values():
        values = holm([rows[i][pcolumn] for i in indices])
        for index, value in zip(indices, values):
            rows[index]["p_holm"] = value
            rows[index]["holm_family_n"] = len(indices)
            rows[index]["significant_holm_005"] = int(value < .05)


def validate(rows, manifest):
    prepared = OLD / "prepared/accuracy_paired.csv"
    assert sha(prepared) == manifest["prepared_files"][prepared.name], "Prepared outcomes changed"
    relevant = {r["source_csv"] for r in read_csv(OLD / "prepared/judge_mapping.csv")}
    relevant |= {r["legacy_label_csv"] for r in read_csv(OLD / "prepared/judge_mapping.csv")}
    for rel in sorted(relevant):
        assert sha(ROOT / rel) == manifest["sources"][rel]["sha256"], f"Original source changed: {rel}"
    seen = set()
    for r in rows:
        key = (r["benchmark"], r["model"], r["question_id"])
        assert key not in seen, f"Duplicated pair: {key}"
        seen.add(key)
        for c in CONDITIONS:
            y, label = int(r[c]), int(r[f"{c}_legacy_label"])
            assert y in (0, 1) and label in range(7)
            assert (label == 0) == bool(y), f"Correctness-label conflict: {key}, {c}"
            if int(r[f"{c}_legacy_operational_failure"]):
                assert label == 6, f"Operational failure outside Other: {key}, {c}"
    reconciliation = []
    names = {"Qwen": "qwen", "Phi-4": "phi4", "Llama": "llama", "Gemma": "gemma"}
    conds = {"B": "base", "N": "neutral", "W": "warm", "D": "dpo"}
    for bench in BENCHMARKS:
        model = None
        for line in (ROOT / f"paper/current/tables/errors_{bench}.tex").read_text(encoding="utf-8").splitlines():
            if "&" not in line:
                continue
            cells = [c.strip() for c in line.replace("\\\\", "").split("&")]
            if cells[0] in names:
                model = names[cells[0]]
            if len(cells) != 9 or cells[1] not in conds:
                continue
            phase = conds[cells[1]]
            group = [r for r in rows if r["benchmark"] == bench and r["model"] == model]
            counts = Counter(int(r[f"{phase}_legacy_label"]) for r in group)
            actual = [len(group) - counts[0], *[counts[k] for k in range(1, 7)]]
            expected = [int(x) for x in cells[2:]]
            assert actual == expected, (bench, model, phase, actual, expected)
            reconciliation.append({"benchmark": bench, "model": model, "condition": phase,
                                   "n": len(group), "correct": counts[0], "errors": actual[0],
                                   **{f"label_{k}": counts[k] for k in range(1, 7)}, "paper_count_match": 1})
        for model in MODELS:
            group = [r for r in rows if r["benchmark"] == bench and r["model"] == model]
            expected_n = {"mmlu": 2800, "finance": 1032, "medical": 1273}[bench]
            assert len(group) == expected_n
            if bench == "mmlu":
                assert len(Counter(r["category"] for r in group)) == 14
                assert set(Counter(r["category"] for r in group).values()) == {200}
    assert len(reconciliation) == 48
    return reconciliation, {rel: sha(ROOT / rel) for rel in sorted(relevant)}


def accuracy(group, benchmark, model, iterations, seed, conflict_policy="as_scored"):
    rows = []
    strata = [r["category"] if benchmark == "mmlu" else "all" for r in group]
    for a, b in (*CONTRASTS, *DPO_CONTRASTS):
        result = paired_binary([int(r[a]) for r in group], [int(r[b]) for r in group],
                               strata, iterations, rng_for(seed, f"accuracy:{benchmark}:{model}:{a}:{b}:{conflict_policy}"))
        exact = 1.0 if result["discordant"] == 0 else binomtest(result["a0_b1"], result["discordant"], .5).pvalue
        assert math.isclose(result["p_exact_mcnemar"], exact, rel_tol=1e-9, abs_tol=1e-15)
        row = {"benchmark": benchmark, "model": model, "partition": "Overall",
               "condition_a": a, "condition_b": b, "reference_policy": conflict_policy,
               "multiplicity_family": ("accuracy_primary_sft_36" if b != "dpo" else "accuracy_dpo_24")
                    if conflict_policy == "as_scored" else "finance_reference_sensitivity_20",
               "bootstrap_unit": "within_category_item_pair" if benchmark == "mmlu" else "item_pair",
               **result}
        rows.append(row)
    return rows


def ratio(counts):
    counts = np.asarray(counts, dtype=float)
    den = counts[..., 1:].sum(axis=-1, keepdims=True)
    return np.divide(counts[..., 1:], den, out=np.full(counts[..., 1:].shape, np.nan), where=den > 0)


def label_contrasts(group, benchmark, model, iterations, seed, partition="Overall", policy="as_scored"):
    result = []
    transitions = []
    n = len(group)
    strata = np.array([r["category"] if benchmark == "mmlu" else "all" for r in group])
    for a, b in CONTRASTS:
        la = np.array([int(r[f"{a}_legacy_label"]) for r in group])
        lb = np.array([int(r[f"{b}_legacy_label"]) for r in group])
        actual_a, actual_b = np.bincount(la, minlength=7), np.bincount(lb, minlength=7)
        draws_a, draws_b = np.zeros((iterations, 7), dtype=np.int32), np.zeros((iterations, 7), dtype=np.int32)
        rng = rng_for(seed, f"labels:{benchmark}:{model}:{a}:{b}:{partition}:{policy}")
        for stratum in np.unique(strata):
            mask = strata == stratum
            counts = np.bincount(la[mask] * 7 + lb[mask], minlength=49)
            present = np.flatnonzero(counts)
            probabilities = counts[present] / mask.sum()
            a_indicator = np.eye(7, dtype=np.int32)[present // 7]
            b_indicator = np.eye(7, dtype=np.int32)[present % 7]
            # Joint multinomial counts exactly implement paired-item resampling,
            # without allocating a 20,000 x 2,800 item index array.
            weights = rng.multinomial(int(mask.sum()), probabilities, size=iterations)
            draws_a += weights @ a_indicator
            draws_b += weights @ b_indicator
        assert np.all(draws_a.sum(1) == n) and np.all(draws_b.sum(1) == n)
        joint = np.bincount(la * 7 + lb, minlength=49).reshape(7, 7)
        for old in range(7):
            for new in range(7):
                transitions.append({"benchmark": benchmark, "model": model, "partition": partition,
                                    "reference_policy": policy, "condition_a": a, "condition_b": b,
                                    "label_a": old, "label_b": new, "count": int(joint[old, new])})
        incidence_draws = (draws_b[:, 1:] - draws_a[:, 1:]) / n
        incidence_actual = (actual_b[1:] - actual_a[1:]) / n
        error_draws = ratio(draws_b) - ratio(draws_a)
        error_actual = ratio(actual_b) - ratio(actual_a)
        for denominator, draws, actual in (("all_evaluated_items", incidence_draws, incidence_actual),
                                           ("condition_specific_incorrect_responses", error_draws, error_actual)):
            for index in range(6):
                label = index + 1
                values = draws[:, index]
                values = values[np.isfinite(values)]
                if not len(values):
                    raise ValueError("No finite bootstrap ratios")
                estimate = float(actual[index])
                ci = np.quantile(values, [.025, .975]) * 100
                p = (1 + np.count_nonzero(np.abs(values - estimate) >= abs(estimate) - 1e-12)) / (len(values) + 1)
                family = "ld_primary_12" if (a, b) == ("neutral", "warm") and label == 2 else (
                    "kd_supporting_12" if (a, b) == ("neutral", "warm") and label == 1 else "all_labels_exploratory_192")
                if denominator != "all_evaluated_items":
                    family = "error_share_secondary_" + family
                if partition != "Overall":
                    family = "mmlu_partition_exploratory_288"
                if policy != "as_scored":
                    family = "finance_reference_sensitivity_144"
                wrong_a, wrong_b = int(actual_a[1:].sum()), int(actual_b[1:].sum())
                den_a, den_b = (n, n) if denominator == "all_evaluated_items" else (wrong_a, wrong_b)
                count_draws = draws_b[:, label] - draws_a[:, label]
                count_ci = np.quantile(count_draws, [.025, .975])
                row = {"benchmark": benchmark, "model": model, "partition": partition,
                       "reference_policy": policy, "condition_a": a, "condition_b": b,
                       "label": label, "label_name": LABELS[label], "denominator": denominator,
                       "n_pairs": n, "count_a": int(actual_a[label]), "count_b": int(actual_b[label]),
                       "difference_b_minus_a_count": int(actual_b[label] - actual_a[label]),
                       "paired_ci95_low_count": float(count_ci[0]), "paired_ci95_high_count": float(count_ci[1]),
                       "denominator_a": den_a, "denominator_b": den_b,
                       "a_percent": actual_a[label] / den_a * 100, "b_percent": actual_b[label] / den_b * 100,
                       "difference_b_minus_a_pp": estimate * 100, "paired_ci95_low_pp": float(ci[0]),
                       "paired_ci95_high_pp": float(ci[1]), "p_bootstrap_centered": float(p),
                       "valid_bootstrap_draws": len(values), "multiplicity_family": family,
                       "bootstrap_unit": "within_category_item_pair" if benchmark == "mmlu" else "item_pair",
                       "operational_failures_a": sum(int(r[f"{a}_legacy_operational_failure"]) for r in group),
                       "operational_failures_b": sum(int(r[f"{b}_legacy_operational_failure"]) for r in group)}
                result.append(row)
    return result, transitions


def summary_text(accuracy_rows, label_rows, meta):
    primary = [r for r in label_rows if r["denominator"] == "all_evaluated_items" and r["label"] == 2
               and (r["condition_a"], r["condition_b"]) == ("neutral", "warm")]
    lines = ["# 대응 통계 검정 요약", "", "정확도: 양측 exact McNemar. 에러 라벨 개수 차이: 같은 문항쌍을 20,000회 재표집한 paired bootstrap(시드 42), percentile 95% CI 및 centered-null 양측 p. MMLU-Pro는 14개 카테고리 내 재표집. p는 아래 사전에 고정한 retrospective family별 Holm 보정.", "",
             "| 벤치마크 | 모델 | Base→Neutral Δ정확도(pp) | McNemar Holm p | Neutral LD | Warm LD | ΔLD 개수 | 95% CI (개수) | Bootstrap Holm p |",
             "|---|---|---:|---:|---:|---:|---:|---|---:|"]
    for bench in BENCHMARKS:
        for model in MODELS:
            acc = next(r for r in accuracy_rows if r["benchmark"] == bench and r["model"] == model and r["condition_a"] == "base" and r["condition_b"] == "neutral")
            ld = next(r for r in primary if r["benchmark"] == bench and r["model"] == model)
            lines.append(f"| {bench} | {model} | {acc['difference_b_minus_a_pp']:+.2f} | {acc['p_holm']:.4g} | {ld['count_a']} | {ld['count_b']} | {ld['difference_b_minus_a_count']:+d} | [{ld['paired_ci95_low_count']:.1f}, {ld['paired_ci95_high_count']:.1f}] | {ld['p_holm']:.4g} |")
    decreases = [r for r in accuracy_rows if r["condition_a"] == "base" and r["condition_b"] == "neutral" and r["difference_b_minus_a_pp"] < 0]
    gains = [r for r in primary if r["difference_b_minus_a_pp"] > 0]
    significant_gains = [r for r in gains if r["p_holm"] < .05]
    lines += ["", f"- Base 대비 Neutral 정확도는 12개 모델–벤치마크 비교 중 {len(decreases)}개에서 하락했고, 그중 {sum(r['p_holm'] < .05 for r in decreases)}개가 SFT 36개 family Holm p<.05였다.",
              f"- Neutral 대비 Warm의 LD 개수는 {len(gains)}/12 비교에서 증가했다. 그러나 12개 LD family 보정 후 유의한 증가는 {len(significant_gains)}개였다. 방향성 반복과 개별 통계적 유의를 구분해야 한다.",
              "- 논문 제시는 절대 오류 개수와 개수 차이 CI가 우선이다. 고정된 동일 N에서 차이를 N으로 나눈 라벨 발생률 검정과 p는 동일하다. 오답 내 구성비는 조건별 오답 분모가 다르므로 별도 secondary로 구분했다.",
              "- 비유의 결과는 동일성·동등성·비열등성의 증거가 아니다. 고정 모델의 문항 불확실성만 반영하며 학습 시드 변동은 반영하지 않는다.",
              "- 과거 judge API 실패 10건은 기존 정책대로 Other에 남겼다. Finance의 참조 정답 충돌 문항은 원 채점 유지가 primary이며, 충돌 제외 sensitivity도 별도 산출했다.",
              "- Bootstrap p는 null에 중심을 옮긴 근사 검정이며 minimum Monte Carlo p는 1/20,001이다. CI는 개별 차이의 marginal interval이다.", "", "## Multiplicity families", ""]
    for key, value in PLAN.items():
        lines.append(f"- {key}: {value}")
    lines += ["", "## 모델별 세 벤치마크 LD 개수의 기술적 합계", "", "| 모델 | Neutral LD 합계 | Warm LD 합계 | Warm−Neutral |", "|---|---:|---:|---:|"]
    for model in MODELS:
        selected = [r for r in primary if r["model"] == model]
        a, b = sum(r["count_a"] for r in selected), sum(r["count_b"] for r in selected)
        lines.append(f"| {model} | {a} | {b} | {b-a:+d} |")
    lines += ["", "모델별 합계는 방향을 요약한 기술통계이다. 서로 다른 벤치마크를 하나의 모수로 합친 검정도, 동일 문항을 공유하는 4개 모델을 독립 표본처럼 합친 검정도 시행하지 않았다.",
              "", "## 정확도 주요 해석", "", "Warm−Neutral 정확도의 SFT family 보정 후 유의한 차이는 Phi-4 MMLU-Pro 개선뿐이었다. 다른 비유의 비교는 동일함을 증명하지 않는다."]
    for r in accuracy_rows:
        if r["condition_b"] == "dpo" and r["p_holm"] < .05:
            lines.append(f"- {r['benchmark']} {r['model']} {r['condition_a']}→dpo: {r['difference_b_minus_a_pp']:+.2f} pp, exact McNemar Holm p={r['p_holm']:.4g}.")
    return "\n".join(lines) + "\n"


def pooled_counts(rows):
    """Descriptive totals only: do not treat shared benchmark items across models as independent."""
    results = []
    for model in MODELS:
        group = [r for r in rows if r["model"] == model]
        for label in (1, 2):
            a = sum(int(r["neutral_legacy_label"]) == label for r in group)
            b = sum(int(r["warm_legacy_label"]) == label for r in group)
            results.append({"model": model, "label": label, "label_name": LABELS[label],
                            "benchmarks": "mmlu;finance;medical", "n_items": len(group),
                            "neutral_count": a, "warm_count": b, "warm_minus_neutral_count": b - a,
                            "interpretation": "descriptive aggregate, no pooled inferential test"})
    return results


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--iterations", type=int, default=20000)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()
    HERE.mkdir(parents=True, exist_ok=True)
    (HERE / "analysis_plan.json").write_text(json.dumps(PLAN, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    manifest = json.loads((OLD / "prepared/manifest.json").read_text(encoding="utf-8"))
    rows = read_csv(OLD / "prepared/accuracy_paired.csv")
    reconciliation, sources = validate(rows, manifest)
    acc, labels, transitions, sensitivity_acc, sensitivity_labels, partition_labels = [], [], [], [], [], []
    for bench in BENCHMARKS:
        for model in MODELS:
            group = [r for r in rows if r["benchmark"] == bench and r["model"] == model]
            acc.extend(accuracy(group, bench, model, args.iterations, args.seed))
            lr, tr = label_contrasts(group, bench, model, args.iterations, args.seed)
            labels.extend(lr)
            transitions.extend(tr)
            if bench == "finance":
                clean = [r for r in group if not int(r["reference_conflict"])]
                sensitivity_acc.extend(accuracy(clean, bench, model, args.iterations, args.seed, "exclude_reference_conflicts"))
                sens, _ = label_contrasts(clean, bench, model, args.iterations, args.seed, policy="exclude_reference_conflicts")
                sensitivity_labels.extend(sens)
            if bench == "mmlu":
                for partition in ("Specific", "General"):
                    sub = [r for r in group if r["knowledge_type"] == partition]
                    detail, _ = label_contrasts(sub, bench, model, args.iterations, args.seed, partition=partition)
                    partition_labels.extend(detail)
            print(f"Complete: {bench}/{model}", flush=True)
    adjust(acc, "p_exact_mcnemar")
    adjust(sensitivity_acc, "p_exact_mcnemar")
    adjust(labels)
    adjust(sensitivity_labels)
    adjust(partition_labels)
    incidence = [r for r in labels if r["denominator"] == "all_evaluated_items"]
    shares = [r for r in labels if r["denominator"] != "all_evaluated_items"]
    assert len(acc) == 60 and len(incidence) == len(shares) == 216
    meta = {"iterations": args.iterations, "seed": args.seed, "numpy_version": np.__version__,
            "script_sha256": sha(Path(__file__)), "prepared_input_sha256": sha(OLD / "prepared/accuracy_paired.csv"),
            "sources_sha256": sources, "historical_judge_context": "Full-context, researcher-confirmed; no rerun",
            "pair_count": len(rows), "model_condition_benchmark_cells": len(reconciliation),
            "operational_failures_all_conditions": sum(sum(int(r[f"{c}_legacy_operational_failure"]) for c in CONDITIONS) for r in rows),
            "finance_reference_conflict_items_per_model": {m: sum(int(r["reference_conflict"]) for r in rows if r["benchmark"] == "finance" and r["model"] == m) for m in MODELS},
            "analysis_plan": PLAN, "file_counts": {"accuracy": len(acc), "label_incidence": len(incidence), "label_error_share": len(shares), "partition_labels": len(partition_labels)}}
    key_counts = [r for r in incidence if (r["condition_a"], r["condition_b"]) == ("neutral", "warm") and r["label"] in (1, 2)]
    for name, data in (("accuracy_mcnemar.csv", acc), ("label_count_bootstrap.csv", incidence),
                       ("key_kd_ld_count_bootstrap.csv", key_counts), ("pooled_model_kd_ld_counts.csv", pooled_counts(rows)),
                       ("label_incidence_bootstrap.csv", incidence),
                       ("label_error_share_bootstrap.csv", shares), ("label_transitions.csv", transitions),
                       ("paper_count_reconciliation.csv", reconciliation), ("finance_sensitivity_accuracy.csv", sensitivity_acc),
                       ("finance_sensitivity_labels.csv", sensitivity_labels), ("mmlu_partition_labels.csv", partition_labels)):
        write_csv(HERE / name, data)
    (HERE / "analysis_manifest.json").write_text(json.dumps(meta, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    (HERE / "summary_ko.md").write_text(summary_text(acc, labels, meta), encoding="utf-8")
    print(json.dumps(meta["file_counts"], ensure_ascii=False), flush=True)


if __name__ == "__main__":
    main()
