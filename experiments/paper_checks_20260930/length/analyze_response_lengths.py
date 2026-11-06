"""Offline response-length audit of frozen Base/Neutral/Warm outputs.

No model/API calls, no source edits, and no tokenizer-based estimates.
Run from any directory: python analyze_response_lengths.py
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter, defaultdict
from pathlib import Path

import numpy as np

MODELS = ("qwen", "phi4", "llama", "gemma")
CONDITIONS = ("base", "neutral", "warm")
BENCHMARKS = {"mmlu": 2800, "finance": 1032, "medical": 1273}
DISPLAY = {"qwen": "Qwen", "phi4": "Phi-4", "llama": "Llama", "gemma": "Gemma"}
ROOT = Path(__file__).resolve().parents[3]
HERE = Path(__file__).resolve().parent


def sha256(path):
    with path.open("rb") as stream:
        return hashlib.file_digest(stream, "sha256").hexdigest()


def read_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as stream:
        return list(csv.DictReader(stream))


def write_csv(path, rows):
    fields = list(dict.fromkeys(key for row in rows for key in row))
    with path.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fields)
        writer.writeheader()
        writer.writerows(rows)


def bool01(value):
    normalized = str(value).strip().lower()
    if normalized in {"true", "1", "1.0"}:
        return 1
    if normalized in {"false", "0", "0.0"}:
        return 0
    raise ValueError(f"Unknown Boolean value: {value!r}")


def condition(value):
    value = value.strip().lower()
    return {"original": "base", "neutral-sft": "neutral", "warm-sft": "warm"}.get(value, value)


def numeric(value):
    return float(value) if str(value).strip() else None


def statistics(values):
    if not values:
        return {"n": 0, "mean": None, "median": None, "p25": None, "p75": None,
                "p90": None, "p95": None, "max": None}
    vector = np.asarray(values, dtype=float)
    quantiles = np.quantile(vector, [.25, .5, .75, .90, .95], method="linear")
    return {"n": len(vector), "mean": float(vector.mean()), "median": float(quantiles[1]),
            "p25": float(quantiles[0]), "p75": float(quantiles[2]), "p90": float(quantiles[3]),
            "p95": float(quantiles[4]), "max": float(vector.max())}


def summarize(group, benchmark, model, phase, subset="all"):
    row = {"benchmark": benchmark, "model": model, "condition": phase, "subset": subset,
           "n": len(group), "blank_n": sum(r["blank"] for r in group)}
    row["blank_percent"] = 100 * row["blank_n"] / len(group) if group else None
    for field in ("chars", "words", "generated_tokens"):
        available = [r[field] for r in group if r[field] is not None]
        for key, value in statistics(available).items():
            row[f"{field}_{key}"] = value
    known_flags = [r["hit_max_new_tokens"] for r in group if r["hit_max_new_tokens"] is not None]
    row["hit_max_flag_n"] = len(known_flags)
    row["hit_max_n"] = sum(known_flags) if known_flags else None
    row["hit_max_percent"] = 100 * sum(known_flags) / len(known_flags) if known_flags else None
    return row


def matched_summary(left, right, benchmark, model, a, b, subset):
    assert len(left) == len(right)
    result = {"benchmark": benchmark, "model": model, "condition_a": a, "condition_b": b,
              "subset": subset, "n_pairs": len(left)}
    for metric in ("chars", "words", "generated_tokens"):
        paired = [(ra[metric], rb[metric]) for ra, rb in zip(left, right)
                  if ra[metric] is not None and rb[metric] is not None]
        av = np.array([p[0] for p in paired], dtype=float)
        bv = np.array([p[1] for p in paired], dtype=float)
        result[f"{metric}_n_pairs"] = len(paired)
        if not paired:
            for key in ("a_mean", "b_mean", "difference_mean", "difference_mean_percent", "a_median", "b_median", "paired_difference_median"):
                result[f"{metric}_{key}"] = None
            continue
        result.update({f"{metric}_a_mean": float(av.mean()), f"{metric}_b_mean": float(bv.mean()),
                       f"{metric}_difference_mean": float((bv - av).mean()),
                       f"{metric}_difference_mean_percent": float(100 * (bv.mean() / av.mean() - 1)) if av.mean() else None,
                       f"{metric}_a_median": float(np.median(av)), f"{metric}_b_median": float(np.median(bv)),
                       f"{metric}_paired_difference_median": float(np.median(bv - av))})
    return result


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=HERE)
    args = parser.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    out.mkdir(parents=True, exist_ok=True)
    prepared = root / "experiments/judge_paired_20260928/prepared"
    manifest_path = prepared / "manifest.json"
    manifest = json.loads(manifest_path.read_text(encoding="utf-8-sig"))
    audited = {}

    def register(rel, required_hash=True):
        path = root / rel
        actual = sha256(path)
        if required_hash:
            assert rel in manifest["sources"], f"Source not archived: {rel}"
            assert actual == manifest["sources"][rel]["sha256"], f"Source changed: {rel}"
        audited[rel] = {"sha256": actual, "bytes": path.stat().st_size,
                        "matches_archived_manifest": required_hash}
        return path

    paired_path = prepared / "accuracy_paired.csv"
    assert sha256(paired_path) == manifest["prepared_files"][paired_path.name]
    paired = read_csv(paired_path)
    expected = {(r["benchmark"], r["model"], r["question_id"]): r for r in paired}
    inventory_path = prepared / "judge_inventory.csv"
    mapping_path = prepared / "judge_mapping.csv"
    for path in (inventory_path, mapping_path):
        assert sha256(path) == manifest["prepared_files"][path.name]
    inventory = read_csv(inventory_path)
    mapping = read_csv(mapping_path)
    keys_to_label = {}
    label_paths = {r["legacy_label_csv"] for r in mapping if r["condition"] in CONDITIONS}
    for rel in sorted(label_paths):
        for row in read_csv(register(rel)):
            phase = condition(row["model_type"])
            if phase not in CONDITIONS:
                continue
            # The manifest's row mapping is the authoritative model/benchmark map.
            keys_to_label.setdefault(rel, {})[(phase, row["question_id"])] = row
    labels, label_audits = {}, Counter()
    for row in mapping:
        if row["condition"] not in CONDITIONS:
            continue
        original = keys_to_label[row["legacy_label_csv"]][(row["condition"], row["question_id"])]
        label = int(original["error_category"])
        assert label == int(row["legacy_label"])
        key = (row["benchmark"], row["model"], row["condition"], row["question_id"])
        assert key not in labels
        labels[key] = (label, int(row["legacy_operational_failure"]), row["response_sha256"])
        label_audits[row["legacy_label_csv"]] += 1

    sources = sorted({r["source_csv"] for r in inventory if r["condition"] in CONDITIONS})
    source_map = {r["source_csv"]: (r["benchmark"], r["model"]) for r in inventory}
    records, seen = [], set()
    settings = []
    caps = {}
    for bench in ("finance", "medical"):
        for model in MODELS:
            rel = f"extra_benchmarks/{bench}/{model}/manifest.json"
            generation = json.loads(register(rel, False).read_text(encoding="utf-8-sig"))
            caps[(bench, model)] = int(generation["max_new_tokens"])
            settings.append({"benchmark": bench, "model": model, "source": rel,
                             "max_new_tokens": generation["max_new_tokens"],
                             "temperature": generation["temperature"], "top_p": generation["top_p"]})
    for rel in sources:
        bench, model = source_map[rel]
        for row in read_csv(register(rel)):
            phase = condition(row["model_type"])
            if phase not in CONDITIONS:
                continue
            qid = row["question_id"]
            key = (bench, model, phase, qid)
            assert key not in seen, f"Duplicate response key: {key}"
            seen.add(key)
            archival = expected[(bench, model, qid)]
            correct = bool01(row["is_correct"])
            assert correct == bool01(archival[phase])
            text = row["answer_text"]
            response_hash = hashlib.sha256(text.encode()).hexdigest()
            if correct:
                assert key not in labels
                label, operational = 0, 0
            else:
                label, operational, archived_hash = labels[key]
                assert archived_hash == response_hash, f"Response differs from label mapping: {key}"
            assert label == int(archival[f"{phase}_legacy_label"])
            assert operational == int(archival[f"{phase}_legacy_operational_failure"])
            generated_tokens = numeric(row.get("generated_tokens", ""))
            flag = bool01(row["hit_max_new_tokens"]) if row.get("hit_max_new_tokens", "").strip() else None
            if generated_tokens is not None:
                assert generated_tokens >= 0 and generated_tokens.is_integer()
                assert flag == int(generated_tokens >= caps[(bench, model)]), f"Token flag mismatch: {key}"
            records.append({"benchmark": bench, "model": model, "condition": phase, "question_id": qid,
                            "knowledge_type": archival["knowledge_type"], "is_correct": correct,
                            "error_label": label, "judge_operational_failure": operational,
                            "chars": len(text), "words": len(text.split()), "blank": int(not text.strip()),
                            "generated_tokens": generated_tokens, "hit_max_new_tokens": flag,
                            "response_sha256": response_hash, "source_csv": rel})
    assert len(records) == sum(BENCHMARKS.values()) * len(MODELS) * len(CONDITIONS)
    groups = defaultdict(list)
    indexes = {}
    for row in records:
        groups[(row["benchmark"], row["model"], row["condition"])].append(row)
    id_audits = []
    for benchmark, n in BENCHMARKS.items():
        for model in MODELS:
            ids = {qid for (b, m, qid) in expected if b == benchmark and m == model}
            for phase in CONDITIONS:
                group = groups[(benchmark, model, phase)]
                index = {r["question_id"]: r for r in group}
                assert set(index) == ids and len(index) == len(group) == n
                indexes[(benchmark, model, phase)] = index
            id_audits.append({"benchmark": benchmark, "model": model, "n": n,
                              "all_three_conditions_match_prepared_ids": True})

    summaries, comparisons = [], []
    for benchmark in BENCHMARKS:
        for model in MODELS:
            for phase in CONDITIONS:
                group = groups[(benchmark, model, phase)]
                subsets = {"all": group, "nonblank": [r for r in group if not r["blank"]],
                           "correct": [r for r in group if r["is_correct"]],
                           "incorrect": [r for r in group if not r["is_correct"]],
                           "KD": [r for r in group if r["error_label"] == 1],
                           "LD": [r for r in group if r["error_label"] == 2]}
                for subset, selected in subsets.items():
                    summaries.append(summarize(selected, benchmark, model, phase, subset))
            for a, b in (("base", "neutral"), ("neutral", "warm"), ("base", "warm")):
                ia, ib = indexes[(benchmark, model, a)], indexes[(benchmark, model, b)]
                ids = sorted(ia)
                criteria = {"all": ids,
                            "both_nonblank": [q for q in ids if not ia[q]["blank"] and not ib[q]["blank"]],
                            "persistent_wrong": [q for q in ids if not ia[q]["is_correct"] and not ib[q]["is_correct"]],
                            "persistent_KD": [q for q in ids if ia[q]["error_label"] == ib[q]["error_label"] == 1],
                            "persistent_LD": [q for q in ids if ia[q]["error_label"] == ib[q]["error_label"] == 2]}
                for subset, selected in criteria.items():
                    comparisons.append(matched_summary([ia[q] for q in selected], [ib[q] for q in selected],
                                                       benchmark, model, a, b, subset))
    sociot_rows, sociot_audit = [], []
    for model in MODELS:
        rel = f"Final_Results/sociot_responses_{model}_3way.csv"
        response_groups = defaultdict(list)
        for row in read_csv(register(rel)):
            phase = condition(row["model_type"])
            if phase in CONDITIONS:
                text = row["response"]
                response_groups[phase].append({"question_id": row["id"], "chars": len(text), "words": len(text.split()),
                                               "generated_tokens": None, "blank": int(not text.strip()),
                                               "hit_max_new_tokens": None})
        idsets = [{r["question_id"] for r in response_groups[p]} for p in CONDITIONS]
        assert all(len(response_groups[p]) == len(idsets[i]) == 1500 for i, p in enumerate(CONDITIONS))
        assert idsets[0] == idsets[1] == idsets[2]
        for phase in CONDITIONS:
            for subset in ("all", "nonblank"):
                selected = response_groups[phase] if subset == "all" else [r for r in response_groups[phase] if not r["blank"]]
                sociot_rows.append(summarize(selected, "sociot_prompts", model, phase, subset))
        sociot_audit.append({"model": model, "n": 1500, "three_condition_id_sets_equal": True,
                             "source": rel, "max_new_tokens": 300, "cap_source": "paper/current/appendix.tex; no per-response flags"})

    write_csv(out / "all_item_lengths.csv", records)
    write_csv(out / "length_summary.csv", summaries)
    write_csv(out / "matched_length_changes.csv", comparisons)
    write_csv(out / "sociot_length_summary.csv", sociot_rows)
    result = {"knowledge_response_n": len(records), "knowledge_all_item_cells": 36,
              "character_definition": "Python len(raw decoded answer_text): Unicode code points including whitespace",
              "word_definition": "len(answer_text.split()): whitespace-separated units, not tokenizer tokens",
              "blank_definition": "not answer_text.strip(); blank retained as 0 characters if genuinely empty",
              "summary_quantile_method": "numpy linear empirical quantiles",
              "native_token_definition": "generated_tokens recorded by original evaluator; no re-tokenization",
              "label_provenance": "Original saved labels verified against prepared legacy mapping; researcher confirmed these as full-context results. No new judge calls.",
              "script_sha256": sha256(Path(__file__)),
              "manifest_sha256": sha256(manifest_path), "prepared_accuracy_sha256": sha256(paired_path),
              "prepared_mapping_sha256": sha256(mapping_path), "sources": audited,
              "id_alignment_audit": id_audits, "generation_settings": settings, "sociot_id_audit": sociot_audit,
              "limitations": ["Cross-model token counts are not comparable because tokenizers differ.",
                              "MMLU native token counts and cap flags are absent. The paper records 2048 new tokens for Qwen/Phi-4/Llama and 1024 for Gemma; direct censoring checks are impossible there.",
                              "Finance/Medical use 2048 new tokens and stored hit_max_new_tokens flags. Cap-hit responses are length-censored, not fully observed desired lengths.",
                              "KD/LD subsets differ across conditions; their unpaired means are descriptive only. Persistent-wrong/KD/LD matched subsets are secondary, outcome-conditioned comparisons.",
                              "Length differences are not evidence of reasoning depth or a causal explanation of LD.",
                              "SocioT generation uses a different 300-new-token cap and temperature; not poolable with knowledge benchmarks."]}
    (out / "length_audit.json").write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding="utf-8")
    all_summaries = {(r["benchmark"], r["model"], r["condition"]): r for r in summaries if r["subset"] == "all"}
    table = ["| 벤치마크 | 모델 | Base 평균/중앙값 | Neutral 평균/중앙값 | Warm 평균/중앙값 | Warm−Neutral 평균 변화 |",
             "| --- | --- | ---: | ---: | ---: | ---: |"]
    for benchmark in BENCHMARKS:
        for model in MODELS:
            rows = [all_summaries[(benchmark, model, p)] for p in CONDITIONS]
            change = next(r for r in comparisons if r["benchmark"] == benchmark and r["model"] == model
                          and r["condition_a"] == "neutral" and r["condition_b"] == "warm" and r["subset"] == "all")
            cells = [f"{r['chars_mean']:.1f}/{r['chars_median']:.0f}" for r in rows]
            table.append(f"| {benchmark} | {DISPLAY[model]} | {' | '.join(cells)} | {change['chars_difference_mean']:+.1f} ({change['chars_difference_mean_percent']:+.1f}%) |")
    nw = [r for r in comparisons if r["subset"] == "all" and r["condition_a"] == "neutral" and r["condition_b"] == "warm"]
    persistent_ld = [r for r in comparisons if r["subset"] == "persistent_LD" and r["condition_a"] == "neutral" and r["condition_b"] == "warm"]
    tex_table = [r"\begin{table}[t]", r"\centering\small", r"\setlength{\tabcolsep}{3pt}",
                 r"\begin{tabular}{llrrr}", r"\toprule", r"Benchmark & Model & Base & Neutral & Warm\\\midrule"]
    for benchmark in BENCHMARKS:
        for i, model in enumerate(MODELS):
            rows = [all_summaries[(benchmark, model, p)] for p in CONDITIONS]
            values = " & ".join(f"{r['chars_mean']:.1f}/{r['chars_median']:.1f}" for r in rows)
            name = {"mmlu": "MMLU-Pro", "finance": "Finance", "medical": "Medical"}[benchmark] if i == 0 else ""
            tex_table.append(f"{name} & {DISPLAY[model]} & {values} " + r"\\")
        if benchmark != "medical":
            tex_table.append(r"\midrule")
    tex_table += [r"\bottomrule\end{tabular}",
                  r"\caption{Mean/median response length in Unicode characters, including whitespace. All items, including correct and blank responses, are retained. Comparisons are within family and benchmark because generation limits differ.}",
                  r"\label{tab:response-length}\end{table}"]
    cap_table = ["| 벤치마크 | 모델 | Base native token 평균 / 상한률 | Neutral 평균 / 상한률 | Warm 평균 / 상한률 |",
                 "| --- | --- | ---: | ---: | ---: |"]
    for benchmark in ("finance", "medical"):
        for model in MODELS:
            rows = [all_summaries[(benchmark, model, p)] for p in CONDITIONS]
            cells = [f"{r['generated_tokens_mean']:.1f} / {r['hit_max_percent']:.2f}%" for r in rows]
            cap_table.append(f"| {benchmark} | {DISPLAY[model]} | {' | '.join(cells)} |")
    report = ["# 출력 답변 길이 분석", "", "전체 문항(정답 및 오답 포함)을 기준으로 세 조건의 저장된 decoded continuation을 조사했습니다. 문항 ID·원본 SHA-256·정오답·오답 라벨 연결을 보관된 prepared manifest와 대조했습니다. 원본 데이터와 논문은 수정하지 않았습니다.",
              "", "## 주 결과: 문자 수", "", *table, "",
              f"Warm은 Neutral보다 평균 답변이 긴 셀이 {sum(r['chars_difference_mean'] > 0 for r in nw)}/12개입니다. 이는 문자 길이의 기술통계이며 논리 오류 증가의 원인이나 추론 깊이를 입증하지 않습니다.",
              "", "## 저장된 token 수와 생성 상한", "", *cap_table,
              "", f"두 조건에서 모두 LD인 동일 문항만 남겼을 때도 Warm의 평균 문자 수가 증가한 경우는 {sum(r['chars_difference_mean'] > 0 for r in persistent_ld)}/12개입니다. Qwen은 MMLU-Pro에서 869.4→847.9(n=286), Finance에서 840.0→809.6(n=63)으로 감소했습니다. 이는 LD 증가와 일률적인 장문화가 동행하지 않는다는 보조 관찰입니다. 이 집합은 두 조건의 결과에 따라 선정되었으므로 전체 평가 문항에 대한 인과 효과로 일반화하지 않습니다.",
              "", "## 측정·해석", "",
              "문자 수는 공백·줄바꿈을 포함한 Unicode 문자 수입니다. 단어 수는 공백 단위 분할이며 tokenizer token 수와 다릅니다. 출력에 저장된 native generated_tokens는 Finance/Medical에만 있어 별도로 정리했고 새 토크나이저나 모델은 내려받지 않았습니다.",
              "", "Gemma의 빈 답변도 주 분석에서 보존했습니다. nonblank 조건별 통계 및 두 조건 모두 nonblank인 paired 비교를 별도로 저장했습니다. KD/LD 라벨 답변, 양쪽 모두 오답, 양쪽 모두 KD/LD인 문항은 보조 분석입니다. 서로 다른 오답 집합의 평균 길이를 직접 인과 비교하면 안 됩니다.",
              "", "Finance/Medical은 모든 모델에서 최대 2,048 new tokens이며 실제 상한 도달률을 함께 저장했습니다. MMLU-Pro는 논문에 기록된 상한이 Qwen/Phi-4/Llama 2,048, Gemma 1,024이고 원본에는 token/cap 플래그가 없어 상한 도달을 직접 검증할 수 없습니다. SocioT 답변은 300 new tokens로 생성한 별도 환경이므로 본문의 전문가 벤치마크 길이와 합산하지 않습니다.",
              "", "## 논문용 문장 제안", "", "```tex",
              "We additionally audit the length of stored responses across Base, Neutral-SFT, and Warm-SFT using all evaluated items, including correct and blank responses. We report Unicode character counts and, where recorded, native generated-token counts. These descriptive comparisons do not identify reasoning depth or establish that response length explains the change in logical-error labels.",
              "Warm-SFT responses are longer than Neutral-SFT responses in only six of twelve model--benchmark cells, with mixed within-family directions. Thus, the observed increase in logical-error counts is not uniformly accompanied by longer answers; the length audit is descriptive rather than a causal account of the label changes.",
              "```", "", "## Appendix용 표 제안", "", "```tex", *tex_table, "```", "", "## 결과 파일", "", "- length_summary.csv: 벤치마크·모델·조건별 전체/nonblank/정답/오답/KD/LD 분포", "- matched_length_changes.csv: Base–Neutral, Neutral–Warm, Base–Warm matched 길이 변화 및 persistent subset", "- sociot_length_summary.csv: 별도 1,500문항 warmth 출력 길이", "- all_item_lengths.csv: 답변 전문 없이 길이·ID·정오답·라벨·응답 해시", "- length_audit.json: 원본 해시·문항 정렬 검증·generation 설정·해석 제한", ""]
    (out / "README_ko.md").write_text("\n".join(report), encoding="utf-8")
    print(json.dumps({"output": str(out), "knowledge_responses": len(records),
                      "warm_longer_than_neutral_cells": sum(r["chars_difference_mean"] > 0 for r in nw),
                      "total_cells": len(nw)}, ensure_ascii=False))


if __name__ == "__main__":
    main()
