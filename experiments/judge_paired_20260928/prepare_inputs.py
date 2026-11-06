"""Prepare full-context judging and item-paired inputs. Offline; no source edits."""
from __future__ import annotations

import argparse
import csv
import hashlib
import json
import re
import unicodedata
from collections import Counter, defaultdict
from pathlib import Path

MODELS = ("qwen", "phi4", "llama", "gemma")
CONDITIONS = ("base", "neutral", "warm", "dpo")
PRESSURES = ("doubt", "authority", "wrong_suggest")
EXPECTED = {"mmlu": 2800, "finance": 1032, "medical": 1273}
ROOT = Path(__file__).resolve().parents[2]


def digest(value):
    return hashlib.sha256(value).hexdigest()


def file_hash(path):
    with path.open("rb") as f:
        return hashlib.file_digest(f, "sha256").hexdigest()


def load_csv(path):
    with path.open(encoding="utf-8-sig", newline="") as f:
        return list(csv.DictReader(f))


def write_csv(path, rows, fields=None):
    if fields is None:
        fields = list(rows[0]) if rows else []
    with path.open("x", encoding="utf-8-sig", newline="") as f:
        w = csv.DictWriter(f, fieldnames=fields)
        w.writeheader()
        w.writerows(rows)


def bool01(value):
    s = str(value).strip().lower()
    if s in {"1", "1.0", "true"}:
        return 1
    if s in {"0", "0.0", "false"}:
        return 0
    raise ValueError(f"Invalid boolean: {value!r}")


def condition(value):
    s = str(value).strip().lower()
    s = {"original": "base", "warm-sft": "warm", "warm-dpo": "dpo", "neutral-sft": "neutral"}.get(s, s)
    if s not in CONDITIONS:
        raise ValueError(f"Unknown condition: {s}")
    return s


def record_id(benchmark, model, phase, qid):
    return digest(json.dumps([benchmark, model, phase, qid], ensure_ascii=False).encode())[:24]


def normalized_stem(prompt):
    pieces = re.split(r"\n\s*A\)\s*", prompt, maxsplit=1)
    if len(pieces) != 2:
        raise ValueError("Cannot recover SycoBench question stem before A)")
    return " ".join(unicodedata.normalize("NFKC", pieces[0]).casefold().split())


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--output", type=Path, default=Path(__file__).parent / "prepared")
    args = ap.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists():
        raise SystemExit(f"Refusing to overwrite {out}; choose a new --output.")
    out.mkdir(parents=True)
    sources = {}

    def register(path):
        rel = path.relative_to(root).as_posix()
        if rel not in sources:
            sources[rel] = {"sha256": file_hash(path), "bytes": path.stat().st_size}
        return rel

    qpath = root / "peft_handoff/data/eval/mmlu_pro_questions.json"
    register(qpath)
    questions = json.loads(qpath.read_text(encoding="utf-8-sig"))
    qmap = {str(q["question_id"]): q for q in questions}
    assert len(qmap) == len(questions) == 2800
    assert set(Counter(q["category"] for q in questions).values()) == {200}
    provenance = root / "peft_handoff/data/eval/mmlu_pro_questions.provenance.json"
    register(provenance)
    assert json.loads(provenance.read_text(encoding="utf-8-sig"))["sha256"] == file_hash(qpath)

    specs = []
    for model in MODELS:
        stem = "gemma_12b" if model == "gemma" else model
        specs.append(("mmlu", model, root / f"result/results_mmlu_pro_strict_{stem}_200percat.csv"))
        specs.append(("mmlu", model, root / f"results_mmlu_pro_strict_{stem}_200percat_NEUTRAL.csv"))
        specs.append(("mmlu", model, root / f"result/results_mmlu_pro_strict_{model}_200percat_DPO.csv"))
        for bench in ("finance", "medical"):
            specs.append((bench, model, root / f"extra_benchmarks/{bench}/{model}/predictions.csv"))

    old_labels = {}
    for model in MODELS:
        stem = "gemma_12b" if model == "gemma" else model
        paths = [
            ("mmlu", root / f"results_mmlu_pro_strict_{stem}_200percat_NEUTRAL_LABELED.csv"),
            ("mmlu", root / f"result/results_mmlu_pro_strict_{model}_200percat_DPO_FULL_LABELED.csv"),
            ("mmlu", root / ("results_mmlu_pro_strict_llama_200percat_LABELED.csv" if model == "llama" else f"result/results_mmlu_pro_strict_{stem}_200percat_FULL_LABELED.csv")),
            *[(b, root / f"extra_benchmarks/{b}/{model}/predictions_FULL_LABELED.csv") for b in ("finance", "medical")],
        ]
        for bench, path in paths:
            src = register(path)
            for row in load_csv(path):
                key = (bench, model, condition(row["model_type"]), row["question_id"])
                if key in old_labels:
                    raise ValueError(f"Duplicate old label: {key}")
                label = int(row["error_category"])
                assert 1 <= label <= 6
                old_labels[key] = (row, src)

    wide, inventory, provenance_rows = {}, [], []
    refs, conflicts, blocked = {}, [], []
    judge_counts = Counter()
    old_failures = 0
    used_old_labels = set()
    ready_path, held_path = out / "judge_ready.jsonl", out / "judge_reference_review.jsonl"
    with ready_path.open("x", encoding="utf-8") as ready, held_path.open("x", encoding="utf-8") as held:
        for bench, model, path in specs:
            src = register(path)
            counts = Counter()
            for ordinal, row in enumerate(load_csv(path), 1):
                phase, qid = condition(row["model_type"]), row["question_id"]
                y = bool01(row["is_correct"])
                gold = row["correct_answer"].strip().upper()
                if bench == "mmlu":
                    q = qmap[qid]
                    assert gold == q["correct_answer"] and row["category"] == q["category"], (model, phase, qid)
                    question, options, knowledge = q["question"], q["options"], q["knowledge_type"]
                else:
                    question, options, knowledge = row["question"], json.loads(row["options_json"]), ""
                if not isinstance(options, list) or not 2 <= len(options) <= 26:
                    raise ValueError(f"Invalid choices: {bench} {qid}")
                if not question.strip() or any(not isinstance(o, str) or not o.strip() for o in options):
                    raise ValueError(f"Missing question/options: {bench} {qid}")
                if len(gold) != 1 or not 0 <= ord(gold) - 65 < len(options):
                    raise ValueError(f"Gold not in options: {bench} {qid}")
                pred = row["extracted_choice"].strip().upper()
                assert y == int(pred == gold), (src, ordinal, "stored correctness mismatch")
                reference = (question, tuple(options), gold, row["category"])
                if (bench, qid) in refs:
                    assert reference == refs[(bench, qid)], (bench, qid, "reference differs across runs")
                refs[(bench, qid)] = reference
                conflict = bool01(row.get("reference_conflict", "False"))
                key = (bench, model, phase, qid)
                wid = (bench, model, qid)
                target = wide.setdefault(wid, {"benchmark": bench, "model": model, "question_id": qid,
                    "category": row["category"], "knowledge_type": knowledge, "correct_answer": gold,
                    "question_options_sha256": digest(json.dumps(reference, ensure_ascii=False).encode()),
                    "reference_conflict": conflict})
                assert target["reference_conflict"] == conflict
                if phase in target:
                    raise ValueError(f"Duplicate raw key: {key}")
                target[phase] = y
                rid = record_id(*key)
                target[f"{phase}_record_id"] = rid
                counts[(phase, y)] += 1
                if y:
                    continue
                if key not in old_labels:
                    raise ValueError(f"Missing legacy label: {key}")
                old, old_src = old_labels[key]
                used_old_labels.add(key)
                assert old["correct_answer"] == gold and old["extracted_choice"] == row["extracted_choice"], key
                if "answer_text" in old:
                    assert old["answer_text"] == row["answer_text"], key
                operational = old.get("step_by_step_analysis", "").strip().lower() == "error"
                old_failures += int(operational)
                target[f"{phase}_legacy_label"] = int(old["error_category"])
                target[f"{phase}_legacy_operational_failure"] = int(operational)
                payload = {"record_id": rid, "question": question,
                    "options": dict(zip("ABCDEFGHIJKLMNOPQRSTUVWXYZ", options)),
                    "correct_answer": gold, "correct_answer_text": options[ord(gold) - 65],
                    "model_response": row["answer_text"]}
                # record_id is local bookkeeping only and is never sent to the judge.
                payload["input_sha256"] = digest(json.dumps(payload, ensure_ascii=False, sort_keys=True).encode())
                stream = held if conflict else ready
                stream.write(json.dumps(payload, ensure_ascii=False) + "\n")
                status = "reference_review" if conflict else "ready"
                judge_counts[(bench, model, phase, status)] += 1
                provenance_rows.append({"record_id": rid, "benchmark": bench, "model": model,
                    "condition": phase, "question_id": qid, "source_csv": src,
                    "source_record_ordinal_1based": ordinal, "input_sha256": payload["input_sha256"],
                    "legacy_label": int(old["error_category"]), "legacy_operational_failure": int(operational),
                    "legacy_label_csv": old_src, "status": status,
                    "response_sha256": digest(row["answer_text"].encode()),
                    "response_empty": int(not row["answer_text"].strip())})
                if conflict:
                    conflicts.append({"record_id": rid, "benchmark": bench, "model": model, "condition": phase,
                        "question_id": qid, "scoring_answer": gold, "source_answer": row.get("source_answer", "")})
            for phase in CONDITIONS:
                n = counts[(phase, 0)] + counts[(phase, 1)]
                if n:
                    assert n == EXPECTED[bench], (src, phase, n)
                    inventory.append({"benchmark": bench, "model": model, "condition": phase,
                        "n": n, "correct": counts[(phase, 1)], "incorrect": counts[(phase, 0)],
                        "judge_ready": judge_counts[(bench, model, phase, "ready")],
                        "judge_reference_review": judge_counts[(bench, model, phase, "reference_review")],
                        "source_csv": src})
    assert used_old_labels == set(old_labels), "Unexpected legacy rows not present in raw errors"
    for row in wide.values():
        assert all(c in row for c in CONDITIONS), (row["benchmark"], row["model"], row["question_id"])
        for c in CONDITIONS:
            if row[c] == 1:
                row[f"{c}_legacy_label"] = 0
                row[f"{c}_legacy_operational_failure"] = 0
    for bench in EXPECTED:
        idsets = [{r["question_id"] for r in wide.values() if r["benchmark"] == bench and r["model"] == m} for m in MODELS]
        assert all(ids == idsets[0] for ids in idsets) and len(idsets[0]) == EXPECTED[bench]
    order = {m: i for i, m in enumerate(MODELS)}
    wide_rows = sorted(wide.values(), key=lambda r: (list(EXPECTED).index(r["benchmark"]), order[r["model"]], r["question_id"]))
    fields = ["benchmark", "model", "question_id", "category", "knowledge_type", "correct_answer",
        "question_options_sha256", "reference_conflict"]
    fields += [name for c in CONDITIONS for name in (c, f"{c}_record_id", f"{c}_legacy_label", f"{c}_legacy_operational_failure")]
    write_csv(out / "accuracy_paired.csv", wide_rows, fields)
    write_csv(out / "judge_inventory.csv", inventory)
    write_csv(out / "judge_mapping.csv", provenance_rows)
    write_csv(out / "reference_review.csv", conflicts, ["record_id", "benchmark", "model", "condition", "question_id", "scoring_answer", "source_answer"])

    syco, syco_sources, stems = {}, [], {}
    for model in MODELS:
        path = root / f"extra_benchmarks/sycobench/{model}/responses.jsonl"
        register(path)
        with path.open(encoding="utf-8-sig") as f:
            for line in f:
                item = json.loads(line)
                phase, qid, result = condition(item["model_type"]), item["question_id"], item["result"]
                variant = int(result["variant_id"])
                assert variant in (0, 1, 2) and item["key"] == [phase, qid, variant]
                prompt = result["baseline"]["question_prompt"]
                stem = normalized_stem(prompt)
                if qid in stems:
                    assert stems[qid] == stem
                stems[qid] = stem
                key = (model, qid, variant)
                target = syco.setdefault(key, {"model": model, "question_id": qid, "variant_id": variant,
                    "cluster_id": digest(stem.encode())[:24], "domain": result["domain"], "difficulty": result["difficulty"]})
                assert f"{phase}_initial" not in target, key
                initial = bool01(result["baseline"]["correct"])
                pressure = [bool01(result["perturbations"][p]["correct"]) for p in PRESSURES]
                target[f"{phase}_initial"] = initial
                target[f"{phase}_pra_all"] = int(initial and all(pressure))
                target[f"{phase}_pressure_correct"] = sum(pressure)
                target[f"{phase}_flips"] = initial * (3 - sum(pressure))
                target[f"{phase}_gold"] = result["correct"]
    for row in syco.values():
        assert all(f"{c}_initial" in row for c in CONDITIONS)
        assert len({row[f"{c}_gold"] for c in CONDITIONS}) == 1
    assert all(sum(r["model"] == m for r in syco.values()) == 1800 for m in MODELS)
    write_csv(out / "sycobench_paired.csv", list(syco.values()))
    syco_clusters = len(set(stems.values()))

    # Only named response-level scores count. Means/SDs cannot recover pairing.
    socio = []
    for folder in (root / "Final_Results", root / "CSV_files"):
        for path in sorted(folder.glob("*sociot*.csv")):
            with path.open(encoding="utf-8-sig", newline="") as f:
                reader = csv.DictReader(f)
                fields = reader.fieldnames or []
                scores = [v for v in fields if "warmth" in v.lower() and "std" not in v.lower()]
                if "id" not in fields:
                    continue
                counts = Counter()
                missing = Counter()
                ids = set()
                for r in reader:
                    c = condition(r["model_type"])
                    assert (c, r["id"]) not in ids
                    ids.add((c, r["id"]))
                    counts[c] += 1
                    if scores and not r[scores[0]].strip():
                        missing[c] += 1
                register(path)
                for c, n in counts.items():
                    socio.append({"file": path.relative_to(root).as_posix(), "condition": c, "n": n,
                        "score_column": scores[0] if scores else "", "missing_scores": missing[c] if scores else n,
                        "paired_ready": bool(scores)})
    write_csv(out / "sociot_score_inventory.csv", socio)
    summary = {"knowledge_observations": sum(r["n"] for r in inventory), "paired_item_model_rows": len(wide),
        "model_condition_benchmark_cells": len(inventory), "incorrect_responses": len(provenance_rows),
        "judge_ready": sum(r["status"] == "ready" for r in provenance_rows),
        "judge_reference_review": len(conflicts), "legacy_operational_failures": old_failures,
        "sycobench_paired_trial_rows": len(syco), "sycobench_unique_question_ids": len(stems),
        "sycobench_normalized_stem_clusters": syco_clusters, "sources": sources,
        "notes": ["All original files preserved. No API calls or generation.",
            "MMLU questions/options from locally reconstructed archival snapshot; original HF revision unknown.",
            "Finance conflicting reference fields quarantined for judging; accuracy keeps original scoring for primary analysis.",
            "Legacy labels are audit-only and must not fill missing full-context labels.",
            "Syco cluster stem: NFKC, casefold, whitespace normalization before the A) option.",
            "Prepared analysis plan is retrospective, not preregistration."]}
    summary["prepared_files"] = {p.name: file_hash(p) for p in out.iterdir() if p.is_file()}
    (out / "manifest.json").write_text(json.dumps(summary, ensure_ascii=False, indent=2) + "\n", encoding="utf-8")
    print(json.dumps({k: v for k, v in summary.items() if k not in {"sources", "prepared_files", "notes"}}, indent=2))


if __name__ == "__main__":
    main()
