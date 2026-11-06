"""Offline validation, full-vocabulary KL math and crash-safe result storage."""
from __future__ import annotations

from collections import Counter
import csv
import hashlib
import json
import os
from pathlib import Path
import statistics

MODELS = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5_7b"),
    "phi4": ("microsoft/phi-4", "phi4_14b"),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "llama3_1_8b"),
    "gemma": ("google/gemma-4-12b-it", "gemma_12b"),
}
REGIONS = ("early", "middle", "late")


def file_hash(path):
    # hashlib.file_digest requires Python 3.11; retain Python 3.10 compatibility.
    digest = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def text_hash(text):
    return hashlib.sha256(text.encode("utf-8")).hexdigest()


def json_hash(value):
    return text_hash(json.dumps(value, ensure_ascii=False, sort_keys=True, separators=(",", ":"), allow_nan=False))


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_json(path, value):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="\n") as handle:
        json.dump(value, handle, ensure_ascii=False, indent=2, allow_nan=False)
        handle.write("\n")
        handle.flush()
        os.fsync(handle.fileno())
    os.replace(temp, path)


def load_inputs(path, manifest_path):
    manifest = read_json(manifest_path)
    if file_hash(path) != manifest["outputs"]["inputs.jsonl"]["sha256"]:
        raise ValueError("Prepared input hash differs from its manifest; do not mix or edit records.")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["selected_responses"]:
        raise ValueError("Input count differs from the preparation manifest.")
    seen, observations = set(), set()
    for r in rows:
        item = r["item_id"]
        if (not isinstance(item, str) or not item.startswith("E") or not item[1:].isdigit()
                or item in seen or r["model"] not in MODELS or r["condition"] not in ("base", "warm")):
            raise ValueError(f"Invalid/duplicate item metadata: {item}")
        seen.add(item)
        key = (r["model"], r["condition"], r["question_id"])
        if key in observations:
            raise ValueError(f"Duplicate response observation: {key}")
        observations.add(key)
        label = r["judge_label"]
        votes = [r[f"annotator{i}_label"] for i in (1, 2, 3)]
        if (label not in (1, 2) or label != r["human_majority_label"] or votes.count(label) < 2
                or votes.count(label) != r["majority_votes"]
                or r["error_group"] != ("KD" if label == 1 else "LD")):
            raise ValueError(f"Human/Judge selection invalid: {item}")
        if not r["answer_text"].strip() or text_hash(r["answer_text"]) != r["response_sha256"]:
            raise ValueError(f"Response is blank or changed: {item}")
        if (not r["question"].strip() or not 2 <= len(r["options"]) <= 10
                or any(not isinstance(o, str) or not o.strip() for o in r["options"])
                or r["correct_answer"] not in "ABCDEFGHIJ"[:len(r["options"])]
                or r["extracted_choice"] == r["correct_answer"]):
            raise ValueError(f"Question/reference fields invalid: {item}")
    return rows, manifest


def format_prompt(question, options):
    """Exact text of the local archived MMLU-Pro formatter; no gold answer included."""
    alphabet = "ABCDEFGHIJ"
    options_text = "\n".join(f"{alphabet[i]}. {opt}" for i, opt in enumerate(options))
    instruction = "You are an expert taking a multiple-choice test."
    user_input = f'''[Question]
{question}

[Options]
{options_text}

[Strict Constraints]
Write your explanation as a SINGLE, cohesive paragraph.
Do NOT use any XML tags, bullet points, numbered lists, or line breaks in your explanation.
At the very end of your paragraph, you MUST conclude with exactly this sentence:
"Therefore, the correct answer is [Option Letter]."

Response:
'''
    return instruction, user_input


def tokenize_reference(row, tokenizer, owner, model_key):
    instruction, user = format_prompt(row["question"], row["options"])
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user}]
    rendering = "native_system_user"
    try:
        prompt = owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
    except Exception as exc:
        # Gemma can reject system roles. Keep the instruction, as in the archived evaluator.
        if (model_key != "gemma" or type(exc).__name__ not in {"TemplateError", "ValueError"}
                or not any(word in str(exc).lower() for word in ("system", "role", "alternate"))):
            raise
        prompt = owner.apply_chat_template(
            [{"role": "user", "content": instruction + "\n\n" + user}],
            tokenize=False, add_generation_prompt=True)
        rendering = "system_folded_into_user"
    # Preserve the archived three-model tokenizer default; Gemma explicitly used False.
    prompt_specials = model_key != "gemma"
    prompt_ids = list(tokenizer(prompt, add_special_tokens=prompt_specials)["input_ids"])
    offsets = None
    if getattr(tokenizer, "is_fast", False):
        encoded = tokenizer(row["answer_text"], add_special_tokens=False, return_offsets_mapping=True)
        offsets = [list(pair) for pair in encoded["offset_mapping"]]
    else:
        encoded = tokenizer(row["answer_text"], add_special_tokens=False)
    response_ids = list(encoded["input_ids"])
    if not prompt_ids or not response_ids:
        raise ValueError(f"Empty prompt/response after tokenization: {row['item_id']}")
    if offsets is not None and (len(offsets) != len(response_ids)
                               or any(not 0 <= a <= b <= len(row["answer_text"]) for a, b in offsets)):
        raise ValueError(f"Invalid response offsets: {row['item_id']}")
    return {"prompt_ids": prompt_ids, "response_ids": response_ids, "offsets": offsets,
            "rendered_prompt": prompt, "rendering": rendering,
            "prompt_add_special_tokens": prompt_specials,
            "input_ids": prompt_ids + response_ids[:-1]}


def response_logit_slice(logit_length, prompt_length, response_length):
    """Input is prompt + y[:-1]. Logit at P-1 predicts the FIRST response token."""
    if min(prompt_length, response_length) < 1:
        raise ValueError("Nonempty prompt and response required.")
    if logit_length == prompt_length + response_length - 1:
        return slice(prompt_length - 1, prompt_length - 1 + response_length)
    if logit_length == response_length:
        return slice(0, response_length)  # Explicit suffix-logit optimization.
    raise ValueError(f"Unexpected logit length {logit_length}; do not guess the causal alignment.")


def region(index, length):
    return "early" if index < length // 3 else "middle" if index < (2 * length) // 3 else "late"


def distribution_metrics(base_logits, warm_logits, target_ids):
    """Stable full-vocabulary float64 math on a CPU chunk, without top-k approximation."""
    import numpy as np
    b = np.asarray(base_logits, dtype=np.float64)
    w = np.asarray(warm_logits, dtype=np.float64)
    ids = np.asarray(target_ids, dtype=np.int64)
    if b.ndim != 2 or b.shape != w.shape or len(ids) != b.shape[0] or not b.shape[1]:
        raise ValueError("Logit shapes/target lengths must match.")
    if not np.isfinite(b).all() or not np.isfinite(w).all() or np.any(ids < 0) or np.any(ids >= b.shape[1]):
        raise ValueError("Nonfinite logits or target outside vocabulary.")
    def log_softmax(x):
        shifted = x - x.max(axis=1, keepdims=True)
        return shifted - np.log(np.exp(shifted).sum(axis=1, keepdims=True))
    lp, lq = log_softmax(b), log_softmax(w)
    delta = lp - lq
    bw = (np.exp(lp) * delta).sum(axis=1)
    wb = (np.exp(lq) * -delta).sum(axis=1)
    if not np.isfinite(bw).all() or not np.isfinite(wb).all() or min(bw.min(), wb.min()) < -1e-9:
        raise FloatingPointError("Invalid KL; refusing to conceal a numerical error.")
    arange = np.arange(len(ids))
    lb, lw = lp[arange, ids], lq[arange, ids]
    return {"kl_base_to_warm": np.maximum(bw, 0).tolist(),
            "kl_warm_to_base": np.maximum(wb, 0).tolist(),
            "base_logprob": lb.tolist(), "warm_logprob": lw.tolist(),
            "delta_logprob_warm_minus_base": (lw - lb).tolist(),
            "base_probability": np.exp(lb).tolist(), "warm_probability": np.exp(lw).tolist(),
            "base_top1_id": b.argmax(axis=1).tolist(), "warm_top1_id": w.argmax(axis=1).tolist()}


METRICS = ("kl_base_to_warm", "kl_warm_to_base", "base_logprob", "warm_logprob",
           "delta_logprob_warm_minus_base")


def summarize_response(row, tokenized, tokens):
    length = len(tokenized["response_ids"])
    if len(tokens) != length or not tokens:
        raise ValueError("Token output count differs from the fixed response.")
    result = {k: row[k] for k in ("item_id", "model", "condition", "question_id", "category",
              "knowledge_type", "error_group", "human_unanimous", "majority_votes", "origin_run", "response_sha256")}
    result.update(response_tokens=length, prompt_tokens=len(tokenized["prompt_ids"]),
                  teacher_forcing_input_tokens=len(tokenized["input_ids"]),
                  response_characters=len(row["answer_text"]), response_whitespace_words=len(row["answer_text"].split()))
    for metric in METRICS:
        result["mean_" + metric] = statistics.fmean(t[metric] for t in tokens)
        if metric.startswith("kl_"):
            result["sum_" + metric] = sum(t[metric] for t in tokens)
            peak = max(tokens, key=lambda t: t[metric])
            result["max_" + metric] = peak[metric]
            result["max_" + metric + "_token_index"] = peak["token_index"]
            for reg in REGIONS:
                vals = [t[metric] for t in tokens if t["region"] == reg]
                result[f"{reg}_mean_{metric}"] = statistics.fmean(vals) if vals else None
    result["mean_base_nll"] = -result["mean_base_logprob"]
    result["mean_warm_nll"] = -result["mean_warm_logprob"]
    result["top1_changed_fraction"] = statistics.fmean(t["base_top1_id"] != t["warm_top1_id"] for t in tokens)
    return result


class RunStore:
    """Atomic per-response commits. OS lock is released automatically after a crash."""
    def __init__(self, directory, manifest, resume):
        self.directory = Path(directory)
        self.manifest = manifest
        if self.directory.exists() and not resume:
            raise FileExistsError(f"Output exists: {self.directory}. Use --resume or a NEW --output.")
        if resume and not (self.directory / "manifest.json").is_file():
            raise FileNotFoundError("--resume requires an existing run manifest.")
        self.directory.mkdir(parents=True, exist_ok=True)
        self.lock = (self.directory / ".run.lock").open("a+b")
        try:
            if os.name == "nt":
                import msvcrt
                self.lock.seek(0)
                if not self.lock.read(1):
                    self.lock.write(b"0")
                    self.lock.flush()
                self.lock.seek(0)
                msvcrt.locking(self.lock.fileno(), msvcrt.LK_NBLCK, 1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
            path = self.directory / "manifest.json"
            if path.exists():
                if read_json(path) != manifest:
                    raise ValueError("Resume manifest differs: input/code/adapter/runtime/settings must remain fixed.")
            else:
                atomic_json(path, manifest)
            (self.directory / "items").mkdir(exist_ok=True)
            self.rows = {}
            expected = set(manifest["item_ids"])
            for path in (self.directory / "items").glob("*.json"):
                record = read_json(path)
                uid = record["summary"]["item_id"]
                if uid not in expected or path.stem != uid or uid in self.rows:
                    raise ValueError("Unexpected/duplicate saved item.")
                if record.get("run_fingerprint") != json_hash(manifest):
                    raise ValueError("Saved item belongs to a different run.")
                if record["result_sha256"] != json_hash({"summary": record["summary"], "tokens": record["tokens"],
                                                         "reference": record["reference"]}):
                    raise ValueError(f"Saved result changed: {uid}")
                if len(record["tokens"]) != record["summary"]["response_tokens"]:
                    raise ValueError(f"Saved token count differs: {uid}")
                self.rows[uid] = record
        except BaseException:
            self.lock.close()
            raise

    def close(self):
        self.lock.close()

    def save(self, record):
        uid = record["summary"]["item_id"]
        if uid in self.rows or uid not in self.manifest["item_ids"]:
            raise ValueError(f"Duplicate/unexpected save: {uid}")
        record["run_fingerprint"] = json_hash(self.manifest)
        record["result_sha256"] = json_hash({k: record[k] for k in ("summary", "tokens", "reference")})
        atomic_json(self.directory / "items" / (uid + ".json"), record)
        self.rows[uid] = record

    def export(self):
        ordered = [self.rows[uid] for uid in self.manifest["item_ids"] if uid in self.rows]
        if not ordered:
            return
        def write(name, iterator, fields):
            path = self.directory / name
            temp = path.with_name(path.name + ".tmp")
            with temp.open("w", encoding="utf-8-sig", newline="") as handle:
                writer = csv.DictWriter(handle, fieldnames=fields)
                writer.writeheader()
                writer.writerows(iterator)
            os.replace(temp, path)
        summaries = [r["summary"] for r in ordered]
        write("response_metrics.csv", iter(summaries), list(summaries[0]))
        meta = ("item_id", "model", "condition", "error_group", "question_id", "human_unanimous")
        tokens = ({**{k: r["summary"][k] for k in meta}, **t} for r in ordered for t in r["tokens"])
        write("token_metrics.csv", tokens, [*meta, *ordered[0]["tokens"][0]])
        groups = []
        for condition in ("base", "warm"):
            for error in ("KD", "LD"):
                for subset in ("majority_and_judge", "unanimous_and_judge"):
                    selected = [s for s in summaries if s["condition"] == condition and s["error_group"] == error
                                and (subset == "majority_and_judge" or s["human_unanimous"])]
                    planned = [r for r in self.manifest["selected_groups"] if r["condition"] == condition
                               and r["error_group"] == error and (subset == "majority_and_judge" or r["human_unanimous"])]
                    row = {"model": self.manifest["model"], "reference_condition": condition,
                           "error_group": error, "subset": subset,
                           "saved_responses": len(selected), "expected_responses": len(planned),
                           "complete": len(selected) == len(planned),
                           "mean_response_tokens": statistics.fmean(s["response_tokens"] for s in selected) if selected else None,
                           "mean_response_characters": statistics.fmean(s["response_characters"] for s in selected) if selected else None}
                    names = ["mean_" + m for m in METRICS] + [f"{reg}_mean_{m}" for reg in REGIONS
                                                             for m in ("kl_base_to_warm", "kl_warm_to_base")]
                    for name in names:
                        vals = [s[name] for s in selected if s[name] is not None]
                        row[name + "_response_equalweight"] = statistics.fmean(vals) if vals else None
                    groups.append(row)
        write("summary_by_group.csv", iter(groups), list(groups[0]))
