#!/usr/bin/env python3
"""Prepare AI-drafted P1/P2 decisions; apply only explicitly confirmed decisions.

No model, API or server calls. Original KL arrays and the active human-review
files are never modified by build. apply checks hashes, retains a byte-for-byte
backup, and refuses partial or stale approval. Python >=3.10 and NumPy required.
"""
from __future__ import annotations

import argparse
from collections import Counter
import copy
import csv
import json
from pathlib import Path
import subprocess
import sys
import tempfile

import kl_local as kl

HERE = Path(__file__).resolve().parent
DEFAULT_OUT = kl.DEFAULT_REVIEW / "assisted_review"
TOOL = "kl_assisted_review"
VERSION = 1
# Reviewed in context by the assistant. These are nonessential positive appraisal
# candidates, not an exhaustive or validated interpersonal-warmth detector.
STYLE_PROPOSALS = {
    "phi4:E104": [("a great way to understand how it works", True,
        "비필수 긍정적 평가 표현. 핵심 커널 정의는 유지합니다.")],
    "phi4:E106": [("fascinating", True,
        "대상에 대한 긍정적 수식어. Kinsey 척도의 설명은 유지합니다.")],
    "phi4:E184": [("really interesting", True,
        "도입부의 비필수 평가 표현. 유전학 내용은 유지합니다.")],
    "llama:E108": [("thoughtful", False,
        "심리학적 판단과 문체 수식이 겹칩니다. 기본값은 제외하지 않음.")],
}
TERMINAL_PROPOSALS = {
    "gemma:E125": "Therefore, the correct answer is None of the above."
}
CATEGORY_KO = {
    "final_answer": "최종 정답 문장", "warmth_style": "문체·긍정적 평가 표현",
    "content_reasoning": "내용·추론", "format": "서식·구두점", "uncertain": "판단 보류",
}


def exact_span(text, quote):
    if not quote or text.count(quote) != 1:
        raise ValueError(f"Proposed quote must occur exactly once: {quote!r}")
    a = text.index(quote)
    return {"char_start": a, "char_end": a + len(quote), "quote": quote}


def input_hashes(review):
    return {name: kl.file_hash(review / name)
            for name in ("peak_review.csv", "style_spans.json")}


def loose_reviews(path, records):
    """Retain incomplete TRUE rows, but never treat them as category approval."""
    with path.open(encoding="utf-8-sig", newline="") as handle:
        rows = list(csv.DictReader(handle))
    expected = {(r["_key"], d): r for r in records for d in kl.DIRECTIONS}
    result = {}
    for row in rows:
        key = (f"{row['model']}:{row['item_id']}", row["direction"])
        if key not in expected or key in result:
            raise ValueError(f"Unknown/duplicate review row: {key}")
        if row["result_sha256"] != expected[key]["result_sha256"]:
            raise ValueError(f"Review hash mismatch: {key}")
        if row["category"] and row["category"] not in kl.CATEGORIES:
            raise ValueError(f"Invalid existing category: {key}")
        kl.parse_bool(row["reviewed"])
        result[key] = row
    if set(result) != set(expected):
        raise ValueError("Review CSV coverage mismatch")
    return result


def proposed_record(record, reviews, style_entry):
    s, text = record["summary"], record["reference"]["answer_text"]
    key = record["_key"]
    automatic = kl.final_answer_span(text, record["reference"].get("options"))
    final = ({"char_start": automatic[0], "char_end": automatic[1],
              "quote": text[automatic[0]:automatic[1]]} if automatic else None)
    mode, flags = "automatic", []
    # Existing valid edits take precedence over suggestions. Confirmation of the
    # new complete response-level decision is still required in the interface.
    override = style_entry.get("final_answer_override")
    if override is not None:
        mode = "custom" if kl.parse_bool(override.get("exclude", True)) else "none"
        final = ({k: override[k] for k in ("char_start", "char_end", "quote")}
                 if mode == "custom" else None)
        flags.append("기존 최종 문장 지정")
    elif key in TERMINAL_PROPOSALS:
        mode = "custom"
        final = exact_span(text, TERMINAL_PROPOSALS[key])
        flags.append("A–J 형식이 아닌 명시적 최종 문장: 제외 후보 확인")
    elif automatic is None:
        flags.append("명시적 최종 정답 형식 없음: 추론 결론을 임의로 제외하지 않음")
    spans = copy.deepcopy(style_entry.get("spans", []))
    for span in spans:
        span["exclude"] = kl.parse_bool(span.get("exclude", False))
    for quote, exclude, reason in STYLE_PROPOSALS.get(key, []):
        candidate = exact_span(text, quote)
        if not any(x["char_start"] == candidate["char_start"] and
                   x["char_end"] == candidate["char_end"] for x in spans):
            spans.append({**candidate, "exclude": exclude, "reason": reason,
                          "source": "assistant_contextual_candidate", "reviewed": False})
        flags.append("문체 구간 후보: " + quote)
    selected = [span for span in spans if span["exclude"]]
    final_tuple = (final["char_start"], final["char_end"]) if final else None
    peaks = []
    for direction in kl.DIRECTIONS:
        token = max(record["tokens"], key=lambda t: t["kl_" + direction])
        old = reviews[key, direction]
        category = kl.token_role(token, final_tuple, selected)
        reason = {
            "final_answer": "표시된 토큰 위치가 최종 정답 문장과 겹침.",
            "warmth_style": "표시된 위치가 제외 후보인 비필수 평가 표현과 겹침.",
            "content_reasoning": "문맥상 설명·계산·논거에 포함된 위치. 논리 오류 지점이라는 뜻은 아님.",
            "format": "관찰된 토큰 조각이 구두점·공백. KL은 이 위치의 전체 어휘 분포를 비교함.",
            "uncertain": "문자 조각만으로 기능을 결정하기 어려움.",
        }[category]
        # Math delimiters/operators encode content, unlike sentence punctuation.
        context = text[max(0, token["char_start"] - 35):token["char_end"] + 35]
        piece = token.get("token_text", "")
        if category == "format" and any(c in piece for c in "$=+-*/<>\\"):
            category = "content_reasoning"
            reason = "수식 구분·연산 기호가 포함됨. 단순 서식으로 제외하지 않음."
            flags.append("수식 기호 피크: 주변 계산 내용 확인")
        if piece and piece.isspace():
            flags.append("공백 토큰 피크: 전체 분포 변화와 관찰 토큰 의미를 구분")
        if key == "llama:E108" and direction == "warm_to_base":
            category, reason = "uncertain", "thoughtful은 심리학적 평가와 문체가 겹쳐 기본적으로 보류."
        if old["category"]:
            category, reason = old["category"], "기존 사용자가 입력한 분류를 보존함."
        peaks.append({"direction": direction, "token_index": token["token_index"],
            "token_text": piece, "char_start": token["char_start"], "char_end": token["char_end"],
            "relative_position_pct": 100 * token["token_index"] / len(record["tokens"]),
            "kl_nats": token["kl_" + direction], "context": context,
            "category": category, "reason": reason, "old_reviewed": kl.parse_bool(old["reviewed"]),
            "old_category": old["category"], "old_notes": old.get("notes", "")})
    return {"key": key, "model": s["model"], "item_id": s["item_id"],
        "condition": s["condition"], "error_group": s["error_group"],
        "result_sha256": record["result_sha256"], "response_sha256": s["response_sha256"],
        "source_relative_path": record["_source_relative_path"],
        "question": record["reference"].get("question", ""),
        "options": record["reference"].get("options", []), "answer_text": text,
        "response_tokens": len(record["tokens"]), "peaks": peaks, "spans": spans,
        "automatic_final": ({"char_start": automatic[0], "char_end": automatic[1],
             "quote": text[automatic[0]:automatic[1]]} if automatic else None),
        "final": final, "final_mode": mode, "flags": list(dict.fromkeys(flags)),
        "confirmed": False, "notes": ""}


def provisional_analysis(records, candidates, output, iterations):
    rows, roles = [], Counter()
    for record, candidate in zip(records, candidates):
        spans = [span for span in candidate["spans"] if span["exclude"]]
        final = candidate["final"]
        final_tuple = (final["char_start"], final["char_end"]) if final else None
        for peak in candidate["peaks"]:
            direction = peak["direction"]
            roles[(candidate["model"], candidate["condition"], candidate["error_group"],
                   direction, peak["category"])] += 1
            for variant in kl.VARIANTS:
                metrics, _ = kl.response_metrics(record, direction, variant, spans, final_tuple)
                rows.append({"model": candidate["model"], "item_id": candidate["item_id"],
                    "condition": candidate["condition"], "error_group": candidate["error_group"],
                    "direction": direction, "variant": variant,
                    "review_status": "assistant_draft_not_human_confirmed", **metrics})
    kl.write_csv(output / "draft_response_sensitivity.csv", rows)
    summaries = kl.group_summary(rows, ("peak_position_pct", "mean_token_kl_nats"), iterations, 42)
    for row in summaries:
        row["review_status"] = "assistant_draft_not_human_confirmed"
    kl.write_csv(output / "draft_group_summary.csv", summaries)
    kl.write_csv(output / "draft_peak_role_counts.csv", [dict(zip(
        ("model", "condition", "error_group", "direction", "category"), key),
        responses=count, review_status="assistant_draft_not_human_confirmed")
        for key, count in sorted(roles.items())])
    return summaries


def build(args):
    records, provenance = kl.load_records(args.source)
    review, output = args.review.resolve(), args.output.resolve()
    if output == review or review not in output.parents:
        raise ValueError("Use a dedicated subdirectory of the review folder")
    hashes = input_hashes(review)
    if output.exists():
        manifest = kl.read_json(output / "manifest.json")
        if manifest.get("tool") == TOOL and manifest.get("input_hashes") == hashes:
            print(f"Existing draft preserved: {output / 'review.html'}")
            return
        raise FileExistsError("Draft exists for different inputs. Use a new --output subdirectory.")
    reviews = loose_reviews(review / "peak_review.csv", records)
    style_doc = kl.read_json(review / "style_spans.json")
    kl.read_styles(review / "style_spans.json", records)
    candidates = [proposed_record(r, reviews, style_doc["items"][r["_key"]]) for r in records]
    data = {"tool": TOOL, "schema_version": VERSION, "source_digest": kl.json_hash(provenance),
            "input_hashes": hashes, "category_labels": CATEGORY_KO,
            "review_status": "assistant_draft_not_human_confirmed", "items": candidates}
    data["candidate_digest"] = kl.json_hash(data)
    output.mkdir(parents=True)
    snapshot = output / "original_input_snapshot"
    snapshot.mkdir()
    for name in hashes:
        (snapshot / name).write_bytes((review / name).read_bytes())
    kl.write_json(output / "candidates.json", data)
    template = (HERE / "review_template.html").read_text(encoding="utf-8")
    embedded = json.dumps(data, ensure_ascii=False, allow_nan=False).replace("<", "\\u003c")
    kl.atomic_text(output / "review.html", template.replace("__DATA_JSON__", embedded))
    peak_rows = [{"model": item["model"], "item_id": item["item_id"],
        "condition": item["condition"], "error_group": item["error_group"],
        "direction": peak["direction"], "peak_token": peak["token_text"],
        "peak_position_pct": peak["relative_position_pct"], "proposed_category": peak["category"],
        "rationale": peak["reason"], "needs_attention": bool(item["flags"]),
        "attention_reason": "; ".join(item["flags"]), "human_confirmed": False,
        "previous_reviewed_flag": peak["old_reviewed"], "previous_category": peak["old_category"]}
        for item in candidates for peak in item["peaks"]]
    kl.write_csv(output / "draft_peak_categories.csv", peak_rows)
    summaries = provisional_analysis(records, candidates, output, args.bootstrap_iterations)
    manifest = {"tool": TOOL, "schema_version": VERSION, "source": str(args.source.resolve()),
        "review_dir": str(review), "source_digest": data["source_digest"],
        "candidate_digest": data["candidate_digest"], "input_hashes": hashes,
        "responses": len(candidates), "peak_decisions": len(peak_rows),
        "attention_responses": sum(bool(item["flags"]) for item in candidates),
        "proposed_style_items": sum(bool(item["spans"]) for item in candidates),
        "proposed_excluded_style_items": sum(any(s["exclude"] for s in item["spans"]) for item in candidates),
        "automatic_terminal_items": sum(item["automatic_final"] is not None for item in candidates),
        "custom_terminal_candidates": sum(item["final_mode"] == "custom" for item in candidates),
        "previous_incomplete_review_flags": sum(p["old_reviewed"] and not p["old_category"]
            for item in candidates for p in item["peaks"]),
        "human_confirmed_responses": 0, "active_inputs_modified": False,
        "review_status": data["review_status"], "bootstrap_iterations": args.bootstrap_iterations,
        "policy": "Token-location roles, not causes. Nonessential appraisal is a style sensitivity, "
                  "not a validated warmth detector. No semantic answer-conclusion masks invented. "
                  "Masked statistics retain original token positions and model context."}
    kl.write_json(output / "manifest.json", manifest)
    table = ["# KL 후보 검토 요약", "", "자동 분류와 문맥 검토를 결합한 AI 초안입니다. 인간 확인 전입니다.", "",
             f"- 응답 {len(candidates)}개 / 방향별 피크 {len(peak_rows)}개",
             f"- 우선 확인 응답 {manifest['attention_responses']}개",
             "- 기존 입력 파일과 원본 KL는 변경하지 않았습니다.", "",
             "## Forward KL 최대 위치 평균 (%) — AI 후보 마스킹에 따른 잠정값", "",
             "| 그룹 | 원본 | 최종 문장 제외 | 최종 문장+문체 후보 제외 |", "|---|---:|---:|---:|"]
    for condition, error in (("base", "KD"), ("base", "LD"), ("warm", "KD"), ("warm", "LD")):
        values = [next(x for x in summaries if x["scope"] == "pooled_response_weighted" and
                  x["direction"] == "base_to_warm" and x["condition"] == condition and
                  x["error_group"] == error and x["variant"] == variant and x["metric"] == "peak_position_pct")
                  for variant in kl.VARIANTS]
        table.append(f"| {condition.title()}-{error} | " + " | ".join(
            f"{v['mean']:.2f} [{v['ci_low']:.2f}, {v['ci_high']:.2f}]" for v in values) + " |")
    table += ["", "95% 구간은 모델 내 응답 재표집 후 합치는 층화 bootstrap입니다. "
              "서로 다른 문항·길이의 선택 표본이므로 인과 효과나 Base–Warm paired 효과가 아닙니다.", "",
              "## 간단한 검토", "", "1. review.html을 브라우저에서 열고 '확인할 후보'부터 봅니다.",
              "2. 분류/마스크가 맞으면 응답별 확인, 일반 후보까지 검토한 뒤 전체 확인도 가능합니다.",
              "3. '확인 파일 저장'으로 kl_review_approval.json을 내려받습니다.",
              "4. 아래 명령은 확인된 결정을 반영하고 최종 로컬 분석을 실행합니다.", "", "```powershell",
              'python .\\experiments\\sd_kl_controls_20261001\\review_assist.py apply --approval "C:\\Users\\researcher\\Downloads\\kl_review_approval.json" --analyze',
              "```", "", "확인 없는 입력, 일부만 확인된 입력, 원본 검토 파일이 달라진 입력은 반영하지 않습니다."]
    kl.atomic_text(output / "README_ko.md", "\n".join(table) + "\n")
    print(json.dumps({k: manifest[k] for k in ("responses", "peak_decisions", "attention_responses",
        "proposed_style_items", "previous_incomplete_review_flags", "review_status")}, ensure_ascii=False))
    print(f"Review: {output / 'review.html'}")


def check_span(span, text):
    a, b = span.get("char_start"), span.get("char_end")
    if type(a) is not int or type(b) is not int or not 0 <= a < b <= len(text):
        raise ValueError("Invalid span bounds; offsets use Unicode code points")
    if text[a:b] != span.get("quote"):
        raise ValueError("Span quote mismatch")


def validated_approval(data, approval, records):
    if approval.get("tool") != TOOL or approval.get("schema_version") != VERSION:
        raise ValueError("Unknown approval file")
    for field in ("candidate_digest", "source_digest", "input_hashes"):
        if approval.get(field) != data[field]:
            raise ValueError(f"Approval draft mismatch: {field}")
    if approval.get("confirmation_statement") != "I reviewed and confirmed these proposed decisions.":
        raise ValueError("Explicit confirmation statement missing")
    decisions = approval.get("items")
    if not isinstance(decisions, list) or len(decisions) != len(records):
        raise ValueError("Approval must retain all response decisions")
    by_key = {x["key"]: x for x in decisions}
    if len(by_key) != len(records) or set(by_key) != {r["_key"] for r in records}:
        raise ValueError("Unknown/duplicate response decision")
    for record in records:
        x, text = by_key[record["_key"]], record["reference"]["answer_text"]
        if x.get("confirmed") is not True:
            raise ValueError(f"Response not explicitly confirmed: {record['_key']}")
        if x.get("result_sha256") != record["result_sha256"] or x.get("response_sha256") != record["summary"]["response_sha256"]:
            raise ValueError("Approval source hash mismatch")
        peaks = x.get("peaks", [])
        if len(peaks) != 2 or {p.get("direction") for p in peaks} != set(kl.DIRECTIONS):
            raise ValueError("Keep both KL directions exactly once")
        if any(p.get("category") not in kl.CATEGORIES for p in peaks):
            raise ValueError("Invalid peak category")
        if not isinstance(x.get("notes", ""), str):
            raise ValueError("Notes must be text")
        if x.get("final_mode") not in ("automatic", "custom", "none"):
            raise ValueError("Invalid final-answer decision")
        if x["final_mode"] == "custom":
            check_span(x["final"], text)
        seen = set()
        for span in x.get("spans", []):
            check_span(span, text)
            if type(span.get("exclude")) is not bool:
                raise ValueError("Style exclude must be boolean")
            pair = (span["char_start"], span["char_end"])
            if pair in seen:
                raise ValueError("Duplicate style span")
            seen.add(pair)
    return by_key


def apply(args):
    output = args.output.resolve()
    data, manifest = kl.read_json(output / "candidates.json"), kl.read_json(output / "manifest.json")
    if manifest.get("tool") != TOOL:
        raise ValueError("Unknown draft owner")
    unhashed = {k: v for k, v in data.items() if k != "candidate_digest"}
    if kl.json_hash(unhashed) != data.get("candidate_digest") or data["candidate_digest"] != manifest["candidate_digest"]:
        raise ValueError("Candidate document was changed")
    source, review = Path(manifest["source"]), Path(manifest["review_dir"])
    records, provenance = kl.load_records(source)
    if kl.json_hash(provenance) != data["source_digest"]:
        raise ValueError("Source archive changed")
    if input_hashes(review) != data["input_hashes"]:
        raise ValueError("Active review files changed since draft creation; preserve and rebuild in a new directory")
    approval = kl.read_json(args.approval)
    confirmed = validated_approval(data, approval, records)
    old_reviews = loose_reviews(review / "peak_review.csv", records)
    styles = kl.read_json(review / "style_spans.json")
    rows = []
    for record in records:
        item = confirmed[record["_key"]]
        for peak in item["peaks"]:
            row = copy.deepcopy(old_reviews[record["_key"], peak["direction"]])
            row.update(category=peak["category"], reviewed=True)
            row["notes"] = " | ".join(filter(None, [row.get("notes", ""), item.get("notes", ""),
                "Assistant draft explicitly confirmed by user; see assisted_review/approved_decisions.json"]))
            rows.append(row)
        entry = styles["items"][record["_key"]]
        entry.update(reviewed=True, review_origin="assistant_draft_human_confirmed")
        entry["spans"] = [{**{k: span[k] for k in ("char_start", "char_end", "quote", "exclude")},
            "source": "assistant_draft_human_confirmed", "reviewed": True} for span in item["spans"]]
        mode = item["final_mode"]
        entry["final_answer_override"] = (None if mode == "automatic" else
            {"exclude": False, "reviewed": True} if mode == "none" else
            {**{k: item["final"][k] for k in ("char_start", "char_end", "quote")},
             "exclude": True, "reviewed": True, "review_origin": "assistant_draft_human_confirmed"})
    styles["assisted_review_provenance"] = {"candidate_digest": data["candidate_digest"],
        "approval_sha256": kl.file_hash(args.approval), "origin": "assistant_draft_human_confirmed"}
    # Validate both prospective files before replacing either active input.
    with tempfile.TemporaryDirectory(prefix="validate_kl_", dir=output) as temp:
        temp = Path(temp)
        kl.write_csv(temp / "peak_review.csv", rows, kl.REVIEW_FIELDS)
        kl.write_json(temp / "style_spans.json", styles)
        kl.read_reviews(temp / "peak_review.csv", records)
        kl.read_styles(temp / "style_spans.json", records)
        if input_hashes(review) != data["input_hashes"]:
            raise ValueError("Concurrent edit detected; active files unchanged")
        original = {name: (review / name).read_bytes() for name in data["input_hashes"]}
        kl.write_json(output / "approved_decisions.json", approval)
        try:
            for name in original:
                kl.atomic_text(review / name, (temp / name).read_text(encoding="utf-8"))
        except BaseException:
            for name, content in original.items():
                (review / name).write_bytes(content)
            raise
    kl.write_json(output / "apply_receipt.json", {"tool": TOOL, "approval_sha256": kl.file_hash(args.approval),
        "before": data["input_hashes"], "after": input_hashes(review),
        "source_digest": data["source_digest"], "confirmed_responses": len(records),
        "status": "assistant_draft_human_confirmed", "backup": str(output / "original_input_snapshot")})
    print(f"Applied explicitly confirmed decisions for {len(records)} responses.")
    if args.analyze:
        subprocess.run([sys.executable, "-B", str(HERE / "kl_local.py"), "prepare",
            "--source", str(source), "--output", str(review), "--rebuild"], check=True)
        subprocess.run([sys.executable, "-B", str(HERE / "kl_local.py"), "analyze",
            "--review", str(review), "--rebuild", "--require-reviewed",
            "--bootstrap-iterations", str(args.bootstrap_iterations)], check=True)


def main():
    if hasattr(sys.stdout, "reconfigure"):
        sys.stdout.reconfigure(encoding="utf-8")
    parser = argparse.ArgumentParser(description=__doc__)
    sub = parser.add_subparsers(dest="command", required=True)
    b = sub.add_parser("build")
    b.add_argument("--source", type=Path, default=kl.DEFAULT_SOURCE)
    b.add_argument("--review", type=Path, default=kl.DEFAULT_REVIEW)
    b.add_argument("--output", type=Path, default=DEFAULT_OUT)
    b.add_argument("--bootstrap-iterations", type=int, default=20000)
    a = sub.add_parser("apply")
    a.add_argument("--output", type=Path, default=DEFAULT_OUT)
    a.add_argument("--approval", type=Path, required=True)
    a.add_argument("--analyze", action="store_true")
    a.add_argument("--bootstrap-iterations", type=int, default=20000)
    args = parser.parse_args()
    if args.bootstrap_iterations < 100:
        parser.error("bootstrap iterations must be >=100")
    (build if args.command == "build" else apply)(args)


if __name__ == "__main__":
    main()
