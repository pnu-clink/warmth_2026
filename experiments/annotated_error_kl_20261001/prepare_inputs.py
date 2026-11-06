"""Prepare fixed Base/Warm KD/LD responses with human-majority and judge agreement.

Offline only. Source annotations and evaluations are never modified. No KL or
generation is run. Llama uses the archived responses actually shown to annotators.
"""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
from collections import Counter
from pathlib import Path

ROOT = Path(__file__).resolve().parents[2]
MODELS = ("qwen", "phi4", "llama", "gemma")
DISPLAY = {"qwen": "Qwen", "phi4": "Phi-4", "llama": "Llama", "gemma": "Gemma"}
ERRORS = {1: "Knowledge Deficit", 2: "Logical Derailment"}
LABEL_SOURCES = {
    "qwen": "result/results_mmlu_pro_strict_qwen_200percat_FULL_LABELED.csv",
    "phi4": "result/results_mmlu_pro_strict_phi4_200percat_FULL_LABELED.csv",
    "llama": "result/old_llama/results_mmlu_pro_strict_llama_200percat_FULL_LABELED.csv",
    "gemma": "result/results_mmlu_pro_strict_gemma_12b_200percat_FULL_LABELED.csv",
}
HUMAN = "annotation/analysis_20260930/task2_annotators_123_labels_confidence.csv"
JUDGE = "annotation/task2_LLMKEY.csv"
QUESTION = "peft_handoff/data/eval/mmlu_pro_questions.json"
TEMPLATES = tuple(f"annotation/task2_annotator{i}.csv" for i in (1, 2, 3))


def sha_text(value: str) -> str:
    return hashlib.sha256(value.encode("utf-8")).hexdigest()


def file_sha(path: Path) -> str:
    with path.open("rb") as handle:
        return hashlib.file_digest(handle, "sha256").hexdigest()


def load_csv(path: Path) -> list[dict[str, str]]:
    with path.open(encoding="utf-8-sig", newline="") as handle:
        return list(csv.DictReader(handle))


def unique(rows, fields, source):
    result = {}
    for ordinal, row in enumerate(rows, 1):
        key = tuple(str(row[f]).strip() for f in fields)
        if any(not v for v in key) or key in result:
            raise ValueError(f"Missing/duplicate key in {source}: {key}")
        result[key] = (row, ordinal)
    return result


def selection_decision(labels: list[int], judge_label: int):
    if len(labels) != 3 or any(v not in range(1, 7) for v in labels):
        raise ValueError("Exactly three valid human labels (1..6) are required.")
    if judge_label not in range(1, 7):
        raise ValueError("Judge label must be in 1..6.")
    label, votes = Counter(labels).most_common(1)[0]
    if votes < 2:
        return None, votes, "excluded_no_human_majority"
    if label != judge_label:
        return label, votes, "excluded_human_judge_disagreement"
    if label not in ERRORS:
        return label, votes, "excluded_non_KD_LD"
    return label, votes, "selected"


def write_csv(path, rows):
    if not rows:
        raise ValueError(f"No rows for {path}")
    with path.open("x", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def build(root: Path, out: Path):
    if out.exists():
        raise FileExistsError(f"Refusing to overwrite {out}; choose another --output.")
    sources = (HUMAN, JUDGE, QUESTION, *TEMPLATES, *LABEL_SOURCES.values())
    before = {s: {"sha256": file_sha(root / s), "bytes": (root / s).stat().st_size}
              for s in sources}
    human = unique(load_csv(root / HUMAN), ("item_id",), HUMAN)
    judge = unique(load_csv(root / JUDGE), ("item_id",), JUDGE)
    templates = [unique(load_csv(root / s), ("item_id",), s) for s in TEMPLATES]
    if any(set(mapping) != set(judge) for mapping in (human, *templates)):
        raise ValueError("Human, Judge and response-template item-ID sets differ.")
    questions = unique(json.loads((root / QUESTION).read_text(encoding="utf-8-sig")),
                       ("question_id",), QUESTION)
    originals = {m: unique(load_csv(root / s), ("question_id", "model_type"), s)
                 for m, s in LABEL_SOURCES.items()}
    selected, audit, annotation_count = [], [], Counter()
    observation_keys = set()
    for key, (j, judge_ordinal) in judge.items():
        item_id = key[0]
        h, human_ordinal = human[key]
        model = j["model"].strip().lower()
        original_type = j["model_type"].strip()
        if model not in MODELS or original_type not in {"original", "warm"}:
            raise ValueError(f"Unsupported model/condition: {item_id}")
        condition = "base" if original_type == "original" else "warm"
        qid = j["question_id"].strip()
        observation = (model, condition, qid)
        if observation in observation_keys:
            raise ValueError(f"Duplicate response observation: {observation}")
        observation_keys.add(observation)
        template_rows = [mapping[key][0] for mapping in templates]
        shown = template_rows[0]
        fields = ("question_id", "category", "correct_answer", "extracted_choice", "answer_text")
        if any(any(t[f] != shown[f] for f in fields) for t in template_rows):
            raise ValueError(f"Annotators were shown different stimuli: {item_id}")
        q, question_ordinal = questions[(qid,)]
        original, original_ordinal = originals[model][(qid, original_type)]
        if any(original[f] != shown[f] for f in fields):
            raise ValueError(f"Annotation/source response differs: {item_id}")
        judge_label = int(j["llm_error_category"])
        if int(original["error_category"]) != judge_label:
            raise ValueError(f"LLMKEY/original judge label differs: {item_id}")
        if not (j["category"] == shown["category"] == q["category"]):
            raise ValueError(f"Category differs: {item_id}")
        if shown["correct_answer"] != q["correct_answer"]:
            raise ValueError(f"Reference answer differs: {item_id}")
        if shown["extracted_choice"] == shown["correct_answer"] or int(original["is_correct"]) != 0:
            raise ValueError(f"Annotated error is actually correct: {item_id}")
        options = q["options"]
        if not q["question"].strip() or not options or any(not o.strip() for o in options):
            raise ValueError(f"Missing question/options: {item_id}")
        labels = [int(h[f"annotator{i}_label"]) for i in (1, 2, 3)]
        confidences = [int(h[f"annotator{i}_confidence"]) for i in (1, 2, 3)]
        if any(c not in range(1, 6) for c in confidences):
            raise ValueError(f"Invalid confidence (expected 1..5): {item_id}")
        majority, votes, decision = selection_decision(labels, judge_label)
        annotation_count[model] += 1
        response_hash = sha_text(shown["answer_text"])
        origin = "archived_llama_annotation_run" if model == "llama" else "current_annotation_response_run"
        common = {
            "item_id": item_id, "model": model, "condition": condition,
            "question_id": qid, "category": q["category"],
            "judge_label": judge_label, "human_majority_label": majority,
            "majority_votes": votes, "human_unanimous": votes == 3,
            "annotator1_label": labels[0], "annotator2_label": labels[1], "annotator3_label": labels[2],
            "annotator1_confidence": confidences[0], "annotator2_confidence": confidences[1],
            "annotator3_confidence": confidences[2], "origin_run": origin,
            "response_sha256": response_hash, "response_empty": not shown["answer_text"].strip(),
            "source_response_csv": LABEL_SOURCES[model],
            "source_response_ordinal_1based": original_ordinal,
        }
        audit.append({**common, "selection_decision": decision})
        if decision != "selected":
            continue
        if not shown["answer_text"].strip():
            raise ValueError(f"Selected KD/LD response is blank: {item_id}")
        selected.append({
            **common, "schema_version": 1, "benchmark": "mmlu_pro",
            "model_display": DISPLAY[model], "original_model_type": original_type,
            "condition_display": "Base" if condition == "base" else "Warm-SFT",
            "error_label": ERRORS[majority], "error_group": "KD" if majority == 1 else "LD",
            "knowledge_type": q.get("knowledge_type"),
            "question": q["question"], "options": options,
            "correct_answer": shown["correct_answer"], "extracted_choice": shown["extracted_choice"],
            "is_correct": False, "answer_text": shown["answer_text"],
            "judge_reason": j["llm_reason"], "original_judge_reason": original.get("reason"),
            "original_judge_analysis": original.get("step_by_step_analysis"),
            "response_characters": len(shown["answer_text"]),
            "response_whitespace_words": len(shown["answer_text"].split()),
            "source_human_csv": HUMAN, "source_human_ordinal_1based": human_ordinal,
            "source_llmkey_csv": JUDGE, "source_llmkey_ordinal_1based": judge_ordinal,
            "source_question_json": QUESTION, "source_question_ordinal_1based": question_ordinal,
            "source_stimulus_csv": TEMPLATES[0],
            "response_token_count": None, "logical_error_span": None,
        })
    order = lambda r: (MODELS.index(r["model"]), r["condition"] != "base",
                       r["human_majority_label"] or 9, int(r["item_id"][1:]))
    selected.sort(key=order)
    audit.sort(key=lambda r: int(r["item_id"][1:]))
    index_fields = (*audit[0].keys(), "error_group", "error_label", "knowledge_type",
                    "response_characters", "response_whitespace_words")
    index = [{f: ("selected" if f == "selection_decision" else r[f]) for f in index_fields}
             for r in selected]
    summary = []
    for model in MODELS:
        for condition in ("base", "warm"):
            for label in (1, 2):
                group = [r for r in selected if (r["model"], r["condition"], r["human_majority_label"])
                         == (model, condition, label)]
                summary.append({"model": model, "condition": condition,
                                "error_group": "KD" if label == 1 else "LD", "count": len(group),
                                "unanimous_count": sum(r["human_unanimous"] for r in group)})
    if any(file_sha(root / s) != before[s]["sha256"] for s in sources):
        raise ValueError("A source changed during preparation; aborting.")
    out.mkdir(parents=True, exist_ok=False)
    with (out / "inputs.jsonl").open("x", encoding="utf-8", newline="\n") as handle:
        for row in selected:
            handle.write(json.dumps(row, ensure_ascii=False) + "\n")
    write_csv(out / "selected_items.csv", index)
    write_csv(out / "group_summary.csv", summary)
    write_csv(out / "selection_audit.csv", audit)
    output_files = {p.name: {"sha256": file_sha(p), "bytes": p.stat().st_size}
                    for p in sorted(out.iterdir()) if p.is_file()}
    manifest = {
        "schema_version": 1, "prepared_date": "2026-10-01", "benchmark": "mmlu_pro",
        "selection_rule": "At least 2 of 3 human annotators agree with Judge on KD=1 or LD=2.",
        "confidence_threshold": None, "sampling": "All eligible annotated responses; no resampling.",
        "models": list(MODELS), "conditions": ["base", "warm"],
        "annotation_responses": len(audit), "source_verified_per_model": dict(annotation_count),
        "selected_responses": len(selected),
        "selected_unanimous_responses": sum(r["human_unanimous"] for r in selected),
        "selected_by_error": dict(Counter(r["error_group"] for r in selected)),
        "selection_decisions": dict(Counter(r["selection_decision"] for r in audit)),
        "group_summary": summary, "sources": before, "outputs": output_files,
        "sources_unchanged": True, "api_calls": 0, "model_inference_performed": False,
        "llama_included": True,
        "llama_provenance": "Use old_llama responses exactly as annotated, not current Llama responses matched by question ID alone. Included per user direction; generation configuration review is outside this preparation.",
        "limits": [
            "Response-level annotations do not locate the first incorrect reasoning step or token.",
            "No Neutral/DPO responses are selected or assigned human labels.",
            "Question/options are restored; historical rendered/tokenized generation prompts are not reconstructed here.",
            "Counts describe this annotation subset, not whole-benchmark error prevalence.",
            "Base and Warm groups are not automatically item-paired; keep each response as its own reference trajectory.",
        ],
    }
    with (out / "manifest.json").open("x", encoding="utf-8") as handle:
        json.dump(manifest, handle, ensure_ascii=False, indent=2)
        handle.write("\n")
    print(json.dumps({"output": str(out), "selected": len(selected),
                      "by_error": manifest["selected_by_error"], "groups": summary}, ensure_ascii=False, indent=2))


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=ROOT)
    parser.add_argument("--output", type=Path, default=Path(__file__).parent / "prepared")
    args = parser.parse_args()
    build(args.root.resolve(), args.output.resolve())
