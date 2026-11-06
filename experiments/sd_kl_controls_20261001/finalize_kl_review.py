#!/usr/bin/env python3
"""Verify the confirmed P1/P2 outputs and write a concise research summary.

No inference or API calls. Independently re-computes positional masks, numeric
summary means and eight headline bootstrap intervals from archived arrays.
Does not edit paper sources or the archived model results.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import csv
import math
from pathlib import Path
import sys

import numpy as np

import kl_local as kl
import review_assist as assist

ROOT = kl.DEFAULT_REVIEW
OUT = ROOT / "analysis"
MODEL_LABELS = {"qwen": "Qwen", "phi4": "Phi-4", "llama": "Llama", "gemma": "Gemma"}
GROUPS = (("base", "KD"), ("base", "LD"), ("warm", "KD"), ("warm", "LD"))


def read_csv(name):
    with (OUT / name).open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def number(value):
    return None if value in (None, "") else float(value)


def close(actual, expected, label):
    value = number(actual)
    if expected is None:
        if value is not None:
            raise ValueError(f"Expected missing value: {label}")
    elif value is None or not math.isclose(value, expected, rel_tol=1e-10, abs_tol=1e-10):
        raise ValueError(f"Numeric mismatch: {label}: {value} != {expected}")


def confirmed_final(item, record):
    if item["final_mode"] == "none":
        return None
    if item["final_mode"] == "custom":
        final = item["final"]
        return final["char_start"], final["char_end"]
    return kl.final_answer_span(record["reference"]["answer_text"], record["reference"].get("options"))


def independent_metrics(record, direction, variant, decision):
    tokens = record["tokens"]
    final = confirmed_final(decision, record)
    spans = [(s["char_start"], s["char_end"]) for s in decision["spans"] if s["exclude"]]
    overlap = lambda token, pair: token["char_start"] < pair[1] and pair[0] < token["char_end"]
    final_indices = [i for i, t in enumerate(tokens) if final and overlap(t, final)]
    style_indices = [i for i, t in enumerate(tokens) if any(overlap(t, span) for span in spans)]
    excluded = set()
    if variant != "raw":
        excluded.update(final_indices)
    if variant == "exclude_final_answer_and_style":
        excluded.update(style_indices)
    retained = [i for i in range(len(tokens)) if i not in excluded]
    values = [t["kl_" + direction] for t in tokens]
    total = math.fsum(values)
    kept_sum = math.fsum(values[i] for i in retained)
    peak = max(retained, key=lambda i: values[i]) if retained else None
    return {"response_tokens": len(tokens), "eligible_tokens": len(retained),
        "retained_fraction": len(retained) / len(tokens),
        "mean_token_kl_nats": kept_sum / len(retained) if retained else None,
        "peak_kl_nats": values[peak] if peak is not None else None,
        "peak_token_index": peak + 1 if peak is not None else None,
        "peak_position_pct": 100 * (peak + 1) / len(tokens) if peak is not None else None,
        "eligible_kl_mass_fraction": kept_sum / total if total > 0 else None,
        "final_answer_kl_mass_fraction": math.fsum(values[i] for i in final_indices) / total if total > 0 else None,
        "style_kl_mass_fraction": math.fsum(values[i] for i in style_indices) / total if total > 0 else None,
        "eligible_kl_centroid_pct": math.fsum(100 * (i + 1) / len(tokens) * values[i]
             for i in retained) / kept_sum if kept_sum > 0 else None}


def independent_bootstrap(rows, summary, iterations):
    values = defaultdict(list)
    for row in rows:
        value = number(row[summary["metric"]])
        if value is not None:
            values[row["model"]].append(value)
    seed_key = [summary["condition"], summary["error_group"], summary["direction"]]
    seed = int(kl.json_hash([42, summary["scope"], seed_key, summary["metric"]])[:16], 16)
    rng = np.random.default_rng(seed)
    total = sum(len(v) for v in values.values())
    draws = np.zeros(iterations)
    for model in sorted(values):
        data = np.asarray(values[model], dtype=float)
        indices = rng.integers(0, len(data), size=(iterations, len(data)))
        draws += data[indices].mean(axis=1) * len(data) / total
    return np.quantile(draws, [0.025, 0.975])


def verify():
    manifest = kl.read_json(OUT / "manifest.json")
    if manifest["review_status"] != "author_review_complete" or manifest["config"]["iterations"] != 20000:
        raise ValueError("Expected fully confirmed analysis with 20,000 bootstrap iterations")
    if any(kl.file_hash(OUT / name) != digest for name, digest in manifest["outputs"].items()):
        raise ValueError("An analysis output hash changed")
    records, provenance = kl.load_records(kl.DEFAULT_SOURCE)
    if kl.json_hash(provenance) != manifest["source_digest"]:
        raise ValueError("Original results changed")
    source_data = kl.read_json(assist.DEFAULT_OUT / "candidates.json")
    approval = kl.read_json(assist.DEFAULT_OUT / "kl_review_approval.json")
    decisions = assist.validated_approval(source_data, approval, records)
    receipt = kl.read_json(assist.DEFAULT_OUT / "apply_receipt.json")
    if receipt["approval_sha256"] != kl.file_hash(assist.DEFAULT_OUT / "kl_review_approval.json"):
        raise ValueError("Applied approval hash changed")
    for name, digest in receipt["before"].items():
        if kl.file_hash(assist.DEFAULT_OUT / "original_input_snapshot" / name) != digest:
            raise ValueError("Original editable-input backup changed")
    backup = assist.DEFAULT_OUT / "analysis_before_confirmation"
    old_manifest = kl.read_json(backup / "manifest.json")
    for name, digest in old_manifest["outputs"].items():
        if kl.file_hash(backup / name) != digest:
            raise ValueError("Prior analysis backup changed")
    reviews = kl.read_reviews(ROOT / "peak_review.csv", records)
    styles, reviewed_items, finals = kl.read_styles(ROOT / "style_spans.json", records)
    for record in records:
        item = decisions[record["_key"]]
        for peak in item["peaks"]:
            review = reviews[record["_key"], peak["direction"]]
            if not review["reviewed"] or review["category"] != peak["category"]:
                raise ValueError("Applied peak category does not match confirmed decision")
        if finals[record["_key"]] != confirmed_final(item, record):
            raise ValueError("Applied final-answer mask does not match confirmed decision")
        expected = {(s["char_start"], s["char_end"], s["quote"]) for s in item["spans"] if s["exclude"]}
        actual = {(s["char_start"], s["char_end"], s["quote"]) for s in styles[record["_key"]]}
        if actual != expected:
            raise ValueError("Applied style mask does not match confirmed decision")
    raw = read_csv("response_sensitivity.csv")
    summaries = read_csv("group_summary.csv")
    index = {r["_key"]: r for r in records}
    seen = set()
    for row in raw:
        key = (row["model"] + ":" + row["item_id"], row["direction"], row["variant"])
        if key in seen:
            raise ValueError("Duplicate sensitivity row")
        seen.add(key)
        expected = independent_metrics(index[key[0]], key[1], key[2], decisions[key[0]])
        for metric, value in expected.items():
            close(row[metric], value, f"{key}/{metric}")
    if len(seen) != len(records) * 2 * 3:
        raise ValueError("Sensitivity coverage changed")
    # Independent checks of all numeric group means, including model balancing.
    for s in summaries:
        matched = [r for r in raw if all(r[k] == s[k] for k in
            ("condition", "error_group", "direction", "variant")) and
            (s["scope"].startswith("pooled") or r["model"] == s["scope"])]
        values = defaultdict(list)
        for r in matched:
            value = number(r[s["metric"]])
            if value is not None:
                values[r["model"]].append(value)
        flat = [v for vs in values.values() for v in vs]
        mean = (math.fsum(math.fsum(v) / len(v) for v in values.values()) / len(values)
                if s["scope"] == "pooled_model_balanced" and values else
                math.fsum(flat) / len(flat) if flat else None)
        if int(s["n"]) != len(flat) or int(s["source_responses"]) != len(matched):
            raise ValueError("Group sample-count mismatch")
        close(s["mean"], mean, "summary mean")
    ci_checks = 0
    for s in summaries:
        if (s["scope"] == "pooled_response_weighted" and s["direction"] == "base_to_warm" and
            s["metric"] == "peak_position_pct" and s["variant"] in ("raw", "exclude_final_answer_and_style")):
            selected = [r for r in raw if all(r[k] == s[k] for k in
                ("condition", "error_group", "direction", "variant"))]
            low, high = independent_bootstrap(selected, s, 20000)
            close(s["ci_low"], low, "headline CI low")
            close(s["ci_high"], high, "headline CI high")
            ci_checks += 1
    verification = {"source_digest": manifest["source_digest"],
        "approval_sha256": receipt["approval_sha256"], "review_origin": "assistant_draft_confirmed_by_author",
        "independent_three_annotator_span_validation": False,
        "confirmed_responses": len(records), "confirmed_peaks": len(reviews), "reviewed_style_items": reviewed_items,
        "backup_inputs_verified": True, "backup_analysis_verified": True,
        "source_archive_unchanged": True, "applied_decisions_match_approval": True,
        "sensitivity_rows_independently_recomputed": len(raw),
        "summary_means_independently_recomputed": len(summaries),
        "headline_ci_intervals_independently_recomputed": ci_checks,
        "numeric_mismatches": 0, "bootstrap_iterations": 20000,
        "current_input_hashes": assist.input_hashes(ROOT),
        "note": "prepare rewrites the validated peak CSV after apply; current hashes are recorded here, "
                "while apply_receipt preserves the immediate post-apply hashes.",
        "analysis_manifest_sha256": kl.file_hash(OUT / "manifest.json")}
    return summaries, read_csv("peak_role_counts.csv"), verification


def summary_row(rows, condition, error, variant, scope="pooled_response_weighted",
                direction="base_to_warm", metric="peak_position_pct"):
    matched = [r for r in rows if r["scope"] == scope and r["condition"] == condition and
        r["error_group"] == error and r["variant"] == variant and
        r["direction"] == direction and r["metric"] == metric]
    if len(matched) != 1:
        raise ValueError("Missing/duplicate report cell")
    return matched[0]


def ci(row):
    if not row["mean"]:
        return "n.a."
    return f"{float(row['mean']):.2f} [{float(row['ci_low']):.2f}, {float(row['ci_high']):.2f}]"


def report(rows, counts, verification):
    lines = ["# KL 보강 최종 결과", "",
        "확인한 후보 분류와 제외 구간을 적용해 P1/P2를 재집계했습니다. "
        "원본 모델 결과·응답·토큰 확률은 변경하지 않았습니다. 논문 파일은 수정하지 않았습니다.", "",
        "## 핵심 결과", "",
        "최종 정답 문장과 식별된 비필수 긍정적 표현을 제외한 뒤에도, "
        "Base∥Warm KL 방향에서는 Warm 생성 응답의 최대 KL 위치가 Base 생성 응답보다 "
        "평균적으로 뒤쪽에 있습니다. KD에서는 4개 모델 모두, LD에서는 Qwen·Phi-4·Gemma에서 "
        "같은 방향이며 Llama는 반대입니다. Reverse KL의 응답 가중 전체 평균에서는 "
        "같은 후반 이동이 나타나지 않습니다.", "",
        "따라서 이 결과는 선택된 오류 응답의 **방향·모델 의존적인 피크 위치 차이**입니다. "
        "전체 분포가 후반으로 이동했다거나 이 변화가 LD 증가를 일으켰다는 근거는 아닙니다.", "",
        "## 분석 범위와 확인", "",
        "- 대상: Base-KD 21개, Base-LD 38개, Warm-KD 33개, Warm-LD 48개; 총 140개 응답.",
        "- 양방향 피크 280개와 응답별 제외 구간 확인 완료. AI 후보에 대한 저자 검토이며, "
        "새로운 3인 독립 span annotation으로 보고하지 않습니다.",
        "- P2는 저장된 KL 배열의 집계 위치만 제외합니다. 모델 입력·앞선 맥락·원래 토큰 위치는 유지합니다.",
        "- 평균 위치는 원래 응답의 1-based token index / 원래 응답 토큰 수 × 100입니다.",
        "- Bootstrap 20,000회, seed 42. 합산 CI는 모델 내 응답을 재표집하는 층화 percentile 방식입니다.",
        "- 서로 다른 선택 문항 집단의 요약이므로 Base–Warm item-paired 효과나 검정으로 해석하지 않습니다.",
        "- 수정 전 입력과 분석 결과는 assisted_review의 별도 백업에 보존했습니다.", "",
        "## P1: 원본 최대 KL 위치의 기능 분류", "",
        "| 방향 | 내용·추론 | 서식·구두점 | 최종 정답 문장 | 문체·긍정적 표현 | 합계 |",
        "|---|---:|---:|---:|---:|---:|"]
    for direction in kl.DIRECTIONS:
        labels = ("content_reasoning", "format", "final_answer", "warmth_style")
        nums = [sum(int(r["responses"]) for r in counts if r["direction"] == direction and r["peak_role"] == label)
                for label in labels]
        total = sum(nums)
        title = "Base∥Warm" if direction == "base_to_warm" else "Warm∥Base"
        lines.append(f"| {title} | " + " | ".join(f"{n} ({100*n/total:.1f}%)" for n in nums) + f" | {total} |")
    lines += ["", "분류는 관찰된 피크 위치의 기능이며, KL을 유발한 후보 단어·오류 원인을 분류한 것이 아닙니다. "
        "KL 자체는 해당 위치의 전체 next-token 어휘 분포를 비교합니다.", "",
        "문체 후보는 4개 응답에 있고 3개만 제외했습니다. Llama E108의 thoughtful은 "
        "Warm∥Base 피크 분류에서 문체로 확인했지만, 승인 파일의 제외 체크가 꺼져 있어 P2에서는 유지했습니다. "
        "최종 정답 구간은 129개 자동 지정 및 Gemma E125의 명시적 None of the above 문장 1개입니다. "
        "나머지 Llama 10개에는 명시적 최종 정답 형식이 없어 추론 결론을 임의로 제외하지 않았습니다.", "",
        "## P2: Base∥Warm 최대 KL 위치 평균 (%)", "",
        "응답에 동일 가중치를 부여한 합산 평균과 95% CI입니다. %는 정확도가 아니라 응답 내 위치입니다.", "",
        "| 그룹 | n | 원본 | 최종 문장 제외 | 최종 문장·문체 제외 |",
        "|---|---:|---:|---:|---:|"]
    for condition, error in GROUPS:
        values = [summary_row(rows, condition, error, v) for v in kl.VARIANTS]
        lines.append(f"| {condition.title()}-{error} | {values[0]['n']} | " + " | ".join(ci(r) for r in values) + " |")
    full = "exclude_final_answer_and_style"
    contrasts = {}
    for error in ("KD", "LD"):
        b, w = (summary_row(rows, c, error, full) for c in ("base", "warm"))
        contrasts[error] = float(w["mean"]) - float(b["mean"])
    lines += ["", f"전체 마스킹 후 Warm−Base의 평균 위치 차이는 KD {contrasts['KD']:+.2f} 위치 %p, "
        f"LD {contrasts['LD']:+.2f} 위치 %p입니다. 이 차이는 기술적인 점 추정치이며 paired 효과가 아닙니다.", "",
        "피크를 제거하면 다음으로 큰 KL 위치가 선택되므로, 추가 문체 제외 후 피크가 더 뒤로 갈 수도 있습니다. "
        "Warm-LD가 최종 문장 제외 46.41%에서 문체까지 제외한 48.25%로 바뀌는 것은 이 재선택 때문입니다.", "",
        "## 모델별 P2 결과", "",
        "| 모델 | 오류 | Base n | Warm n | Base 평균 [95% CI] | Warm 평균 [95% CI] | Warm−Base 위치 %p |",
        "|---|---|---:|---:|---:|---:|---:|"]
    later = Counter()
    for model in kl.MODELS:
        for error in ("KD", "LD"):
            b, w = (summary_row(rows, c, error, full, scope=model) for c in ("base", "warm"))
            delta = float(w["mean"]) - float(b["mean"])
            later[error] += int(delta > 0)
            lines.append(f"| {MODEL_LABELS[model]} | {error} | {b['n']} | {w['n']} | {ci(b)} | {ci(w)} | {delta:+.2f} |")
    lines += ["", f"KD는 {later['KD']}/4 모델, LD는 {later['LD']}/4 모델에서 Warm 평균 피크가 더 뒤쪽입니다. "
        "Llama LD는 원본에서는 Base 41.93%, Warm 46.27%였으나 마스킹 후 Base 39.45%, Warm 30.80%로 "
        "방향이 바뀝니다. 따라서 네 모델 모두에서 일관된 LD 후반 이동이라고 서술하지 않습니다.", "",
        "모델별 집단은 n=3–24로 작으며 구간이 넓습니다. 각 평균의 CI만 비교해 차이의 유의성을 주장하지 않습니다.", "",
        "## 가중치·KL 방향 점검", "",
        "| 방향·가중치 | Base-KD | Warm-KD | Base-LD | Warm-LD |",
        "|---|---:|---:|---:|---:|"]
    for direction, scope in (("base_to_warm", "pooled_model_balanced"),
                             ("warm_to_base", "pooled_response_weighted"),
                             ("warm_to_base", "pooled_model_balanced")):
        label = ("Base∥Warm" if direction == "base_to_warm" else "Warm∥Base") + (" · 모델 동일 가중" if scope.endswith("model_balanced") else " · 응답 동일 가중")
        ordered = (("base", "KD"), ("warm", "KD"), ("base", "LD"), ("warm", "LD"))
        lines.append(f"| {label} | " + " | ".join(ci(summary_row(rows, c, e, full, scope, direction)) for c, e in ordered) + " |")
    lines += ["", "Base∥Warm의 합산 방향은 모델 동일 가중에서도 유지됩니다. 그러나 reverse KL의 응답 가중 합산에서는 "
        "Warm 평균이 Base보다 뒤로 가지 않습니다. Reverse KL의 모델 동일 가중 결과와도 규모가 다르므로 "
        "이를 KL 방향과 가중치에 불변인 일반 효과로 묘사하지 않습니다.", "",
        "## 피크와 전체 KL 질량 위치의 구분", "",
        "| 그룹 | 마스킹 후 평균 토큰 KL (nats) [95% CI] | KL 가중 위치 중심 (%) [95% CI] |",
        "|---|---:|---:|"]
    for condition, error in GROUPS:
        mean = summary_row(rows, condition, error, full, metric="mean_token_kl_nats")
        center = summary_row(rows, condition, error, full, metric="eligible_kl_centroid_pct")
        mean_text = f"{float(mean['mean']):.4f} [{float(mean['ci_low']):.4f}, {float(mean['ci_high']):.4f}]"
        lines.append(f"| {condition.title()}-{error} | {mean_text} | {ci(center)} |")
    lines += ["", "KL 가중 위치 중심의 Warm−Base 차이는 KD 약 +4.05 위치 %p, LD 약 +1.12 위치 %p로, "
        "최대 피크 위치 차이보다 작습니다. 따라서 피크 결과를 전체 KL 질량의 큰 후반 이동으로 바꾸어 말하지 않습니다.", "",
        "## 본문에 사용할 수 있는 표현", "",
        "최종 정답 문장과 식별된 비필수 문체 표현을 제외한 뒤에도, Base∥Warm 방향의 평균 최대 KL 위치는 "
        "선택된 Warm 오류 응답에서 Base 오류 응답보다 뒤쪽에 나타났다. 이 경향은 KD에서는 네 모델 모두, "
        "LD에서는 세 모델에서 관찰되었으나, 모델 및 KL 방향에 따라 달랐다. 이는 오류 응답의 분포 차이를 "
        "보완적으로 기술하며, 논리 오류 증가의 인과적 기제를 입증하지는 않는다.", "",
        "정답 응답 대조군이나 Neutral 비교를 추가하지 않았으므로 오류 특이성, warmth 특정성, "
        "초반 지식 적용→후반 논리 연결이라는 단계 해석은 확인되지 않았습니다.", "",
        "## 검증", "",
        f"원본 및 백업 해시 확인, 승인 결정 일치 확인, 응답 민감도 {verification['sensitivity_rows_independently_recomputed']}행·"
        f"집단 평균 {verification['summary_means_independently_recomputed']}행·주요 CI {verification['headline_ci_intervals_independently_recomputed']}개를 "
        "별도 계산으로 검증했습니다. 수치 불일치 0건입니다.", ""]
    return "\n".join(lines)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    summaries, roles, verification = verify()
    text = report(summaries, roles, verification)
    report_path = ROOT / "FINAL_RESULTS_ko.md"
    kl.atomic_text(report_path, text)
    verification["report_path"] = str(report_path)
    verification["report_sha256"] = kl.file_hash(report_path)
    kl.write_json(ROOT / "final_verification.json", verification)
    print(f"Verified: {verification['sensitivity_rows_independently_recomputed']} sensitivity rows, "
          f"{verification['summary_means_independently_recomputed']} group means, "
          f"{verification['headline_ci_intervals_independently_recomputed']} headline CIs; no mismatches.")
    print(f"Summary: {report_path}")


if __name__ == "__main__":
    main()
