"""Reproduce released numeric results without withheld human annotations.

Historical research scripts are preserved. This entry point uses their paired
statistics and KL summaries, expands Steering coverage to all twelve cells,
and does not call a model, an API, or a human-agreement routine.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
import math
from pathlib import Path
import sys

sys.dont_write_bytecode = True
ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "scripts"))
from verify_submission import verify


def read(path):
    with Path(path).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def save(path, rows):
    if not rows:
        raise ValueError(f"No output rows: {path.name}")
    with Path(path).open("x", encoding="utf-8", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def load_module(name, path):
    spec = importlib.util.spec_from_file_location(name, path)
    module = importlib.util.module_from_spec(spec)
    sys.modules[name] = module
    spec.loader.exec_module(module)
    return module


def calculate(root=ROOT):
    root = Path(root)
    steering = load_module("released_steering", root / "analysis/steering_statistics_reference.py")
    historical = defaultdict(list)
    for row in read(root / "data/derived/accuracy_paired.csv"):
        historical[(row["benchmark"], row["model"])].append(row)
    current = defaultdict(list)
    for row in read(root / "data/derived/dpo_steering.csv"):
        current[(row["benchmark"], row["model"])].append(row)
    expected = {(b, m) for b in ("mmlu", "finance", "medical")
                for m in ("qwen", "phi4", "llama", "gemma")}
    if set(historical) != expected or set(current) != expected:
        raise ValueError("Expected twelve historical and twelve Steering cells")

    profiles, comparisons = [], []
    for benchmark, model in sorted(expected):
        old = {r["question_id"]: r for r in historical[(benchmark, model)]}
        new = {r["question_id"]: r for r in current[(benchmark, model)]}
        if len(old) != len(historical[(benchmark, model)]) or len(new) != len(current[(benchmark, model)]):
            raise ValueError("Duplicate question identifiers")
        if old.keys() != new.keys():
            raise ValueError("Historical and Steering question identifiers differ")
        ids = sorted(old)
        labels = [int(new[k]["label"]) for k in ids]
        if any(x not in range(7) for x in labels):
            raise ValueError("Unexpected error-label ID")
        if any(int(new[k]["is_correct"]) != int(new[k]["label"] == "0") for k in ids):
            raise ValueError("Correctness and Steering labels disagree")
        for condition in ("base", "neutral", "warm", "dpo", "dpo_steering"):
            counts = Counter(labels if condition == "dpo_steering" else
                             [int(old[k][condition + "_legacy_label"]) for k in ids])
            profiles.append(dict(benchmark=benchmark, model=model, condition=condition,
                                 n=len(ids), correct=counts[0], errors=len(ids) - counts[0],
                                 accuracy_pct=100 * counts[0] / len(ids),
                                 **{f"label_{k}": counts[k] for k in range(1, 7)}))
        strata = [old[k]["category"] if benchmark == "mmlu" else "all" for k in ids]
        for condition in ("warm", "dpo"):
            comparisons.append(steering.comparison(
                model, benchmark, condition,
                [int(old[k][condition + "_legacy_label"]) for k in ids], labels, strata))
    steering.holm(comparisons, "p_mcnemar", "p_mcnemar_holm24")
    for label in (1, 2, 5):
        steering.holm([l for c in comparisons for l in c["labels"] if l["label"] == label],
                      "p_bootstrap", "p_bootstrap_holm24")

    paired_accuracy, paired_labels = [], []
    for comparison in comparisons:
        paired_accuracy.append({k: comparison[k] for k in (
            "model", "benchmark", "old_condition", "n", "old_correct", "new_correct",
            "old_acc", "new_acc", "delta_pp", "corrected_total", "newly_wrong_total",
            "p_mcnemar", "p_mcnemar_holm24")})
        paired_accuracy[-1]["holm_family_n"] = 24
        for label in comparison["labels"]:
            row = dict(model=comparison["model"], benchmark=comparison["benchmark"],
                       old_condition=comparison["old_condition"], n=comparison["n"])
            row.update({k: label[k] for k in ("label", "old_count", "new_count", "delta",
                                            "p_bootstrap", "p_bootstrap_holm24")})
            row.update(ci95_low=label["ci95"][0], ci95_high=label["ci95"][1], holm_family_n=24)
            paired_labels.append(row)

    dora = read(root / "data/derived/dora_error_labels_summary.csv")
    dora_logic = {m: sum(int(r["c2_n"]) for r in dora
                         if r["model"] == m and r["benchmark"] in ("mmlu", "finance", "medical"))
                  for m in ("qwen", "phi4", "llama", "gemma")}
    reductions = []
    for model in ("qwen", "phi4", "llama", "gemma"):
        totals = {c: sum(r["label_2"] for r in profiles if r["model"] == model and r["condition"] == c)
                  for c in ("warm", "dpo", "dpo_steering")}
        reductions.append(dict(model=model, n_model_question_pairs=5105,
                               dpo_steering_logic=totals["dpo_steering"],
                               versus_warm_sft_lora=totals["dpo_steering"] - totals["warm"],
                               versus_warm_dpo_lora=totals["dpo_steering"] - totals["dpo"],
                               versus_warm_sft_dora=totals["dpo_steering"] - dora_logic[model]))

    methods = read(root / "data/derived/method_comparison.csv")
    references = {r["model"]: r for r in methods if r["method"] == "Warm-SFT (LoRA)"}
    sensitivity = []
    largest_score_difference = 0.0
    for row in methods:
        ref = references[row["model"]]
        mean_accuracy = sum(float(row[b]) for b in ("mmlu_pro", "finance", "medical")) / 3
        reference_accuracy = sum(float(ref[b]) for b in ("mmlu_pro", "finance", "medical")) / 3
        warmth_ratio = max(float(row["sociot"]), 0) / float(ref["sociot"])
        accuracy_ratio = mean_accuracy / reference_accuracy
        scores = {f"lambda_{weight:g}": 100 * warmth_ratio ** weight * accuracy_ratio ** (1 - weight)
                  for weight in (0, .25, .5, .75, 1)}
        largest_score_difference = max(largest_score_difference,
                                      abs(scores["lambda_0.5"] - float(row["score_reported"])))
        sensitivity.append(dict(model=row["model"], method=row["method"],
                                mean_accuracy_pct=mean_accuracy, **scores))
    if largest_score_difference > .03:
        raise ValueError("Composite scores disagree with the manuscript's rounded inputs")

    kl = defaultdict(list)
    for row in read(root / "data/kl/response_sensitivity.csv"):
        if row["direction"] == "base_to_warm" and row["peak_position_pct"]:
            kl[(row["condition"], row["error_group"], row["variant"])].append(float(row["peak_position_pct"]))
    peaks = [dict(condition=k[0], error_group=k[1], variant=k[2], n=len(v),
                  mean_peak_position_pct=sum(v) / len(v)) for k, v in sorted(kl.items())]
    totals = {}
    for condition in ("warm", "dpo"):
        labels = [l for c in comparisons if c["old_condition"] == condition
                  for l in c["labels"] if l["label"] == 2]
        totals[condition] = dict(logical_error_difference=sum(l["delta"] for l in labels),
                                reductions=sum(l["delta"] < 0 for l in labels),
                                significant_holm=sum(l["delta"] < 0 and l["p_bootstrap_holm24"] < .05
                                                     for l in labels))
    report = dict(model_benchmark_cells=12, steering_comparisons=len(comparisons),
                  evaluated_steering_rows=sum(len(v) for v in current.values()),
                  steering_error_rows=sum(r["errors"] for r in profiles if r["condition"] == "dpo_steering"),
                  bootstrap_iterations=20000, seed=42, totals=totals,
                  method_comparison_rows=len(methods), composite_score_max_rounding_difference=largest_score_difference,
                  human_annotation="Withheld. No human votes or agreement results are generated.",
                  interpretation="DoRA comparisons are aggregate only. Raw generation and new full-vocabulary KL passes are not replayed.")
    return dict(profiles=profiles, comparisons=comparisons, paired_accuracy=paired_accuracy,
                paired_labels=paired_labels, logical_error_reduction=reductions,
                score_sensitivity=sensitivity, kl_peak_means=peaks, report=report)


def write_outputs(output, results):
    output = Path(output)
    output.mkdir(parents=True, exist_ok=False)
    for key in ("profiles", "paired_accuracy", "paired_labels", "logical_error_reduction", "score_sensitivity", "kl_peak_means"):
        save(output / (key + ".csv"), results[key])
    for filename, key in (("dpo_steering_tests.json", "comparisons"), ("verification.json", "report")):
        with (output / filename).open("x", encoding="utf-8") as handle:
            json.dump(results[key], handle, ensure_ascii=False, indent=2)
            handle.write("\n")


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--output", type=Path, required=True)
    args = parser.parse_args()
    if args.output.resolve().is_relative_to(ROOT):
        raise ValueError("Keep generated outputs outside the extracted release")
    verify()
    results = calculate()
    write_outputs(args.output, results)
    print(json.dumps(results["report"], indent=2))


if __name__ == "__main__":
    main()
