#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""Shared source embedded into each of the three standalone server scripts."""
from __future__ import annotations

import argparse
import contextlib
import csv
import hashlib
import importlib.metadata
import json
import math
import os
from pathlib import Path
import re
import sys

MODEL_REGISTRY = {
    "qwen": ("Qwen/Qwen2.5-7B-Instruct", "qwen2_5_7b"),
    "llama": ("meta-llama/Llama-3.1-8B-Instruct", "llama3_1_8b"),
    "phi4": ("microsoft/phi-4", "phi4_14b"),
    "gemma": ("google/gemma-4-12b-it", "gemma_12b"),
}
CONDITIONS = ("base", "neutral", "warm", "dpo")
CONDITION_NAMES = {"base": "Base", "neutral": "Neutral-SFT", "warm": "Warm-SFT", "dpo": "Warm-DPO"}


def json_dump(value):
    return json.dumps(value, ensure_ascii=False, sort_keys=True, allow_nan=False)


def sha256_file(path):
    digest = hashlib.sha256()
    with Path(path).open("rb") as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b""):
            digest.update(block)
    return digest.hexdigest()


def atomic_json(path, value):
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json_dump(value) + "\n", encoding="utf-8")
    temp.replace(path)


def stable_seed(seed, *parts):
    data = json_dump([seed, *parts]).encode("utf-8")
    return int.from_bytes(hashlib.sha256(data).digest()[:4], "big")


def parse_args(benchmark):
    parser = argparse.ArgumentParser(description=f"{benchmark}: Base, Neutral-SFT, Warm-SFT, Warm-DPO")
    parser.add_argument("--model", required=True, choices=MODEL_REGISTRY)
    parser.add_argument("--root", type=Path, default=Path("/workspace/warm_models"))
    parser.add_argument("--output-dir", type=Path)
    parser.add_argument("--limit", type=int, help="Diagnostic prefix only; written to a separate smoke directory")
    parser.add_argument("--check-data", action="store_true", help="Validate dataset only; no model/GPU loaded")
    parser.add_argument("--resume", action="store_true", help="Resume only if the saved inputs/settings match")
    parser.add_argument("--seed", type=int, default=0 if benchmark == "sycobench" else 42)
    parser.add_argument("--max-new-tokens", type=int, default=128 if benchmark == "sycobench" else 2048)
    args = parser.parse_args()
    if (args.limit is not None and args.limit < 1) or args.max_new_tokens < 1:
        parser.error("--limit and --max-new-tokens must be positive")
    args.root = args.root.expanduser().resolve()
    if args.output_dir is None:
        suffix = args.model if args.limit is None else f"{args.model}_smoke_{args.limit}"
        args.output_dir = args.root / "results" / "extra_benchmarks" / benchmark / suffix
    args.output_dir = args.output_dir.expanduser().resolve()
    # Set before importing datasets/transformers, preserving the user's cache layout.
    os.environ["HF_HOME"] = str(args.root / "hf_cache")
    os.environ["HF_DATASETS_CACHE"] = str(args.root / "hf_cache" / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(args.root / "hf_cache" / "models")
    return args


def inspect_adapters(args):
    expected_base, suffix = MODEL_REGISTRY[args.model]
    paths, fingerprints, base_ids, revisions = {}, {}, set(), set()
    for condition in CONDITIONS[1:]:
        folder = args.root / f"lora_{condition}_adapter_{suffix}"
        config_file = folder / "adapter_config.json"
        if not config_file.is_file():
            raise FileNotFoundError(f"Final adapter not found: {config_file}. Finish training/saving first.")
        config = json.loads(config_file.read_text(encoding="utf-8"))
        base_id = config.get("base_model_name_or_path")
        if not isinstance(base_id, str) or not base_id:
            raise ValueError(f"Missing Base model ID in {config_file}")
        base_ids.add(base_id)
        revisions.add(config.get("revision"))
        # Adapter-disabled inference must really represent the original Base.
        if config.get("peft_type") != "LORA" or config.get("bias", "none") != "none":
            raise ValueError(f"Expected bias-free LoRA: {config_file}")
        if config.get("modules_to_save"):
            raise ValueError(f"modules_to_save requires an explicit Base-control review: {config_file}")
        if config.get("init_lora_weights", True) not in (True, False, "gaussian"):
            raise ValueError(f"Nonstandard initialization requires a Base-weight isolation review: {config_file}")
        weights = sorted(folder.glob("adapter_model*.safetensors"))
        if not weights:
            raise FileNotFoundError(f"No final safetensors adapter weights in {folder}")
        paths[condition] = folder
        fingerprints[condition] = {
            "path": str(folder), "config_sha256": sha256_file(config_file),
            "weights_sha256": {path.name: sha256_file(path) for path in weights},
        }
    if len(base_ids) != 1:
        raise ValueError(f"The adapters name different Base checkpoints: {sorted(base_ids)}")
    if len(revisions) != 1:
        raise ValueError("The adapter configs name different Base revisions")
    base_id = next(iter(base_ids))
    if args.model != "gemma" and base_id != expected_base:
        raise ValueError(f"Unexpected Base checkpoint: {base_id}; expected {expected_base}")
    if args.model == "gemma" and "gemma" not in base_id.lower():
        raise ValueError(f"Gemma adapter references a non-Gemma Base: {base_id}")
    return base_id, paths, fingerprints


def role_compatible_messages(messages):
    """Compatibility fallback for templates rejecting system/adjacent user roles."""
    result, pending_system = [], []
    for message in messages:
        role, content = message["role"], message["content"]
        if role == "system":
            if result:
                raise ValueError("Only initial system messages can be folded into a user turn")
            pending_system.append(content)
            continue
        if pending_system:
            if role != "user":
                raise ValueError("A folded system message must precede a user turn")
            content = "\n\n".join(pending_system + [content])
            pending_system = []
        if result and result[-1]["role"] == role:
            result[-1]["content"] += "\n\n" + content
        else:
            result.append({"role": role, "content": content})
    if pending_system:
        raise ValueError("Missing user message after system instructions")
    return result


def render_chat(template_owner, messages):
    try:
        text = template_owner.apply_chat_template(messages, tokenize=False, add_generation_prompt=True)
        return text, "native"
    except Exception as exc:
        if type(exc).__name__ not in {"TemplateError", "ValueError"}:
            raise
        if not any(word in str(exc).lower() for word in ("role", "alternate", "alternation", "system")):
            raise
        converted = role_compatible_messages(messages)
        text = template_owner.apply_chat_template(converted, tokenize=False, add_generation_prompt=True)
        return text, "native_with_role_compatibility"


class LocalModel:
    def __init__(self, args, base_id, adapter_paths):
        import torch
        import transformers
        from peft import PeftModel
        from transformers import AutoConfig, AutoTokenizer
        if not torch.cuda.is_available():
            raise RuntimeError("CUDA GPU required for model evaluation; --check-data is CPU-only")
        self.torch, self.args = torch, args
        hf = {"trust_remote_code": True, "cache_dir": str(args.root / "hf_cache" / "models")}
        adapter_config = json.loads((adapter_paths["neutral"] / "adapter_config.json").read_text(encoding="utf-8"))
        if adapter_config.get("revision"):
            hf["revision"] = adapter_config["revision"]
        config = AutoConfig.from_pretrained(base_id, **hf)
        revision = getattr(config, "_commit_hash", None)
        if revision:
            hf["revision"] = revision
        if args.model == "gemma":
            from transformers import AutoProcessor
            self.template_owner = AutoProcessor.from_pretrained(base_id, **hf)
            self.tokenizer = self.template_owner.tokenizer
            loader = getattr(transformers, "AutoModelForMultimodalLM", None)
            if loader is None:
                loader = getattr(transformers, "AutoModelForImageTextToText", None)
            if loader is None:
                raise RuntimeError("Use the same Transformers environment as Gemma training; multimodal loader missing")
        else:
            self.tokenizer = AutoTokenizer.from_pretrained(base_id, **hf)
            self.template_owner = self.tokenizer
            loader = transformers.AutoModelForCausalLM
        base = loader.from_pretrained(base_id, config=config, torch_dtype=torch.bfloat16,
                                      device_map="auto", attn_implementation="sdpa", **hf)
        self.model = PeftModel.from_pretrained(base, str(adapter_paths["neutral"]),
                                              adapter_name="neutral", is_trainable=False)
        for condition in ("warm", "dpo"):
            self.model.load_adapter(str(adapter_paths[condition]), adapter_name=condition, is_trainable=False)
        self.model.eval()
        self.model.requires_grad_(False)
        self.terminators = self._terminators()
        self.pad_id = self.tokenizer.pad_token_id
        if self.pad_id is None:
            self.pad_id = self.terminators[0]
        text_config = getattr(config, "text_config", config)
        self.context_length = getattr(text_config, "max_position_embeddings", None)
        template = getattr(self.template_owner, "chat_template", None) or getattr(self.tokenizer, "chat_template", None)
        self.runtime = {
            "base_id": base_id, "resolved_revision": revision, "loader": loader.__name__,
            "chat_template_sha256": hashlib.sha256(json_dump(template).encode()).hexdigest(),
            "versions": {name: importlib.metadata.version(name) for name in ("torch", "transformers", "peft", "datasets", "accelerate")},
            "generation_config": self.model.generation_config.to_dict(),
            "terminators": self.terminators, "tokenization": "native template then add_special_tokens=False",
            "role_compatibility": "native first; merge only on template role error; recorded per response",
        }

    def _terminators(self):
        ids = []
        for eos in (self.tokenizer.eos_token_id, self.model.generation_config.eos_token_id):
            ids.extend(eos if isinstance(eos, (list, tuple)) else [eos])
        token = {"qwen": "<|im_end|>", "llama": "<|eot_id|>"}.get(self.args.model)
        if token and token in self.tokenizer.get_vocab():
            ids.append(self.tokenizer.convert_tokens_to_ids(token))
        ids = list(dict.fromkeys(value for value in ids if isinstance(value, int) and value >= 0))
        if not ids:
            raise ValueError("No valid EOS token IDs")
        return ids

    @contextlib.contextmanager
    def condition(self, name):
        if name == "base":
            with self.model.disable_adapter():
                self.model.eval()
                yield
        else:
            self.model.set_adapter(name)
            self.model.eval()
            self.model.requires_grad_(False)
            yield

    def generate(self, messages, seed):
        from transformers import set_seed
        set_seed(seed)
        text, rendering = render_chat(self.template_owner, messages)
        inputs = self.tokenizer(text, return_tensors="pt", add_special_tokens=False)
        n_input = int(inputs["input_ids"].shape[-1])
        if self.context_length and n_input + self.args.max_new_tokens > self.context_length:
            raise ValueError(f"Context overflow ({n_input}+{self.args.max_new_tokens}>{self.context_length}); no text was truncated")
        device = self.model.get_input_embeddings().weight.device
        inputs = {key: value.to(device) for key, value in inputs.items()}
        kwargs = {"max_new_tokens": self.args.max_new_tokens, "do_sample": BENCHMARK != "sycobench",
                  "pad_token_id": self.pad_id, "eos_token_id": self.terminators}
        if kwargs["do_sample"]:
            kwargs.update(temperature=0.1, top_p=0.9)
        with self.torch.inference_mode():
            output = self.model.generate(**inputs, **kwargs)
        continuation = output[0, n_input:]
        n_output = int(continuation.shape[-1])
        ended = n_output > 0 and int(continuation[-1]) in self.terminators
        return {"text": self.tokenizer.decode(continuation, skip_special_tokens=True),
                "input_tokens": n_input, "generated_tokens": n_output,
                "hit_max_new_tokens": n_output >= self.args.max_new_tokens and not ended,
                "generation_seed": seed, "chat_rendering": rendering}


def strict_messages(item):
    options = "\n".join(f"{chr(65 + i)}. {value}" for i, value in enumerate(item["options"]))
    user = f'''[Question]
{item["question"]}

[Options]
{options}

[Strict Constraints]
Write your explanation as a SINGLE, cohesive paragraph.
Do NOT use any XML tags, bullet points, numbered lists, or line breaks in your explanation.
At the very end of your paragraph, you MUST conclude with exactly this sentence:
"Therefore, the correct answer is [Option Letter]."

Response:
'''
    return [{"role": "system", "content": "You are an expert taking a multiple-choice test."},
            {"role": "user", "content": user}]


def score_mcq(text, correct, n_options):
    cleaned = re.sub(r"<think>.*?(?:</think>|$)", "", text, flags=re.DOTALL).strip()
    match = re.search(r"correct answer is\s*([A-J])", cleaned, re.IGNORECASE)
    if match:
        choice, method = match.group(1).upper(), "Regex_Strict"
    else:
        fallback = re.findall(r"\b([A-J])\b", cleaned.upper())
        choice, method = (fallback[-1], "Regex_Fallback") if fallback else ("Format_Error", "Failed")
    valid = choice in "ABCDEFGHIJ"[:n_options]
    if not valid:
        choice, method = "Format_Error", "Failed"
    return {"extracted_choice": choice, "is_correct": int(choice == correct), "eval_method": method,
            "strict_final_sentence": bool(re.search(r"Therefore, the correct answer is [A-J]\.\s*$", cleaned))}


def wilson(correct, n):
    if n == 0:
        return [None, None]
    z, p = 1.959963984540054, correct / n
    denom = 1 + z * z / n
    centre = (p + z * z / (2 * n)) / denom
    half = z * math.sqrt(p * (1 - p) / n + z * z / (4 * n * n)) / denom
    return [max(0.0, centre - half), min(1.0, centre + half)]


class RunStore:
    def __init__(self, directory, manifest, resume):
        self.directory, self.records, self.manifest = directory, {}, manifest
        manifest_path = directory / "manifest.json"
        journal = directory / "responses.jsonl"
        if directory.exists() and any(directory.iterdir()):
            if not resume:
                raise FileExistsError(f"Output exists: {directory}. Use --resume or a different --output-dir.")
            if not manifest_path.is_file() or json.loads(manifest_path.read_text(encoding="utf-8")) != manifest:
                raise ValueError("Resume inputs/settings/adapter hashes differ; choose a new output directory")
            if journal.exists():
                for line in journal.read_text(encoding="utf-8").splitlines():
                    record = json.loads(line)
                    key = tuple(record["key"])
                    if key in self.records:
                        raise ValueError("Duplicate key in resume journal")
                    self.records[key] = record
            if self.records and not (directory / "runtime.json").is_file():
                raise ValueError("Existing responses have no runtime.json; cannot verify their model environment")
        directory.mkdir(parents=True, exist_ok=True)
        if not manifest_path.exists():
            atomic_json(manifest_path, manifest)
        self.stream = journal.open("a", encoding="utf-8", newline="\n")

    def append(self, key, result):
        record = {"key": list(key), "model_type": key[0], "question_id": key[1], "result": result}
        self.stream.write(json_dump(record) + "\n")
        self.stream.flush()
        os.fsync(self.stream.fileno())
        self.records[key] = record

    def check_runtime(self, runtime):
        path = self.directory / "runtime.json"
        if self.records and not path.exists():
            raise ValueError("Existing responses require their original runtime.json")
        if path.exists() and json.loads(path.read_text(encoding="utf-8")) != runtime:
            raise ValueError("Model revision/template/library runtime changed since the saved run")
        if not path.exists():
            atomic_json(path, runtime)

    def close(self):
        self.stream.close()


def export_mcq(store):
    rows = [record["result"] for record in store.records.values()]
    if rows:
        path = store.directory / "predictions.csv"
        temp = path.with_suffix(".csv.tmp")
        with temp.open("w", encoding="utf-8-sig", newline="") as stream:
            writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
            writer.writeheader()
            writer.writerows(rows)
        temp.replace(path)
    summary = {}
    for condition in CONDITIONS:
        subset = [row for row in rows if row["model_type"] == condition]
        n, correct = len(subset), sum(row["is_correct"] for row in subset)
        expected = store.manifest["evaluation_items"]
        summary[condition] = {"n": n, "expected_n": expected, "complete": n == expected,
                              "correct": correct, "accuracy": correct / n if n else None,
                              "wilson_95_ci": wilson(correct, n),
                              "format_failures": sum(row["eval_method"] == "Failed" for row in subset),
                              "fallback_answers": sum(row["eval_method"] == "Regex_Fallback" for row in subset),
                              "hit_max_new_tokens": sum(row["hit_max_new_tokens"] for row in subset)}
        if store.manifest.get("benchmark") == "finance":
            alternate = sum(row["is_correct_source_answer"] for row in subset)
            summary[condition].update(
                reference_conflicts=sum(row["reference_conflict"] for row in subset),
                source_answer_accuracy=alternate / n if n else None,
                primary_reference="choices[gold]",
            )
    atomic_json(store.directory / "summary.json", summary)
    export_summary_csv(store.directory, summary)
    return summary


def export_summary_csv(directory, summary):
    rows = [{"model_type": condition, **values} for condition, values in summary.items()]
    path = directory / "summary.csv"
    temp = path.with_suffix(".csv.tmp")
    with temp.open("w", encoding="utf-8-sig", newline="") as stream:
        writer = csv.DictWriter(stream, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows({key: json_dump(value) if isinstance(value, (list, dict)) else value
                         for key, value in row.items()} for row in rows)
    temp.replace(path)


BENCHMARK = "medical"

"""Pinned, fail-closed normalizers for the two specialist MCQ benchmarks.

Only ``load_data`` needs Hugging Face datasets; normalization is pure stdlib.
The prompt builder must use question/options, never correct_answer or metadata.
"""


import hashlib
import json
import re
from collections import Counter
from numbers import Integral
from typing import Any, Iterable, Mapping


DATASET_SPECS = {
    "finance": {
        "dataset_path": "Salesforce/FinEval",
        "dataset_config": "CFA-Easy",
        "revision": "a911ec4dbd635228f216349c2f4eda8354ed15cc",
        "split": "test",
        "expected_count": 1032,
        "option_labels": "ABC",
        "license": "cc-by-nc-4.0; research release; underlying-source terms also apply",
        "source_url": "https://huggingface.co/datasets/Salesforce/FinEval",
    },
    "medical": {
        "dataset_path": "GBaker/MedQA-USMLE-4-options",
        "dataset_config": "default",
        "revision": "0fb93dd23a7339b6dcd27e241cb9b5eca62d4d18",
        "split": "test",
        "expected_count": 1273,
        "option_labels": "ABCD",
        "license": "cc-by-4.0 (Hugging Face distribution card)",
        "source_url": "https://huggingface.co/datasets/GBaker/MedQA-USMLE-4-options",
    },
}


def _nonblank(value: Any, field: str) -> str:
    if not isinstance(value, str) or not value.strip():
        raise ValueError(f"{field} must be a non-empty string")
    return value.strip()


def _source_index(index: int) -> int:
    if isinstance(index, bool) or not isinstance(index, Integral) or index < 0:
        raise ValueError("source_index must be a non-negative integer")
    return int(index)


def normalize_finance_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Decode CFA-Easy's embedded option text; ``choices`` holds labels only.

    The pinned release has exactly three labels A/B/C. No answer text, label,
    rationale, or query suffix is appended to the question/options.
    """
    index = _source_index(index)
    if not isinstance(row, Mapping):
        raise ValueError("CFA-Easy row must be a mapping")
    if row.get("choices") != ["A", "B", "C"]:
        raise ValueError("CFA-Easy choices must be the label list A, B, C")
    text = _nonblank(row.get("text"), "text")
    if text.count("CHOICES:") != 1:
        raise ValueError("CFA-Easy text must contain exactly one CHOICES: marker")
    question, option_block = text.split("CHOICES:", 1)
    question = re.sub(r"^Q:\s*", "", question.strip()).rstrip().removesuffix(",").strip()
    question = _nonblank(question, "question")
    # The comma/newline belongs to the delimiter, not the previous option.
    markers = list(re.finditer(r"(?:^|,|\n)\s*([A-Z]):\s*", option_block))
    if [match.group(1) for match in markers] != ["A", "B", "C"]:
        raise ValueError("CFA-Easy embedded option labels must be exactly A, B, C")
    if option_block[: markers[0].start()].strip():
        raise ValueError("Unexpected text before first CFA-Easy option")
    options = [
        _nonblank(
            option_block[match.end(): markers[i + 1].start() if i + 1 < len(markers) else None],
            f"option {match.group(1)}",
        )
        for i, match in enumerate(markers)
    ]
    if re.search(r"\bAnswer\s*:", text, flags=re.IGNORECASE):
        raise ValueError("Unexpected answer marker in CFA-Easy text; refusing possible leakage")
    gold = row.get("gold")
    if isinstance(gold, bool) or not isinstance(gold, Integral) or not 0 <= gold < 3:
        raise ValueError("CFA-Easy gold must be a zero-based index in 0..2")
    answer = _nonblank(row.get("answer"), "answer").upper()
    if answer not in {"A", "B", "C"}:
        raise ValueError("CFA-Easy answer must be an A/B/C label")
    # Match the source CFA evaluator's Classification.process_results, which
    # scores against choices[gold], not doc_to_target's redundant answer field.
    # normalize_rows records source disagreements; nothing is relabeled/dropped.
    gold_label = "ABC"[int(gold)]
    return {
        "question_id": f"finance:CFA-Easy:test:{index:06d}",
        "category": "CFA-Easy",
        "question": question,
        "options": options,
        "correct_answer": gold_label,
        "source_answer": answer,
        "reference_conflict": answer != gold_label,
        "source_index": index,
    }


def normalize_medical_row(row: Mapping[str, Any], index: int) -> dict[str, Any]:
    """Preserve the official A/B/C/D mapping and verify its redundant answer."""
    index = _source_index(index)
    if not isinstance(row, Mapping):
        raise ValueError("MedQA row must be a mapping")
    option_map = row.get("options")
    if not isinstance(option_map, Mapping) or set(option_map) != set("ABCD"):
        raise ValueError("MedQA options must contain exactly A, B, C, D")
    options = [_nonblank(option_map[label], f"option {label}") for label in "ABCD"]
    answer_idx = _nonblank(row.get("answer_idx"), "answer_idx")
    if answer_idx not in {"A", "B", "C", "D"}:
        raise ValueError("MedQA answer_idx must be A, B, C, or D")
    answer = _nonblank(row.get("answer"), "answer")
    if answer != options["ABCD".index(answer_idx)]:
        raise ValueError("MedQA answer text and answer_idx disagree")
    category = _nonblank(row.get("meta_info"), "meta_info")
    if category not in {"step1", "step2&3"}:
        raise ValueError("Unexpected MedQA meta_info category")
    return {
        "question_id": f"medical:MedQA-USMLE-4-options:test:{index:06d}",
        "category": category,
        "question": _nonblank(row.get("question"), "question"),
        "options": options,
        "correct_answer": answer_idx,
        "source_index": index,
    }


def normalize_rows(
    benchmark: str,
    rows: Iterable[Mapping[str, Any]],
    *,
    require_full_count: bool = True,
) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Validate every row before limiting; fail rather than silently dropping any."""
    if benchmark not in DATASET_SPECS:
        raise ValueError("benchmark must be 'finance' or 'medical'")
    normalizer = normalize_finance_row if benchmark == "finance" else normalize_medical_row
    items: list[dict[str, Any]] = []
    source_hasher = hashlib.sha256()
    answer_case_normalizations: list[int] = []
    reference_disagreements: list[dict[str, Any]] = []
    for index, row in enumerate(rows):
        try:
            item = normalizer(row, index)
        except (ValueError, KeyError, TypeError) as exc:
            raise ValueError(f"{benchmark} source row {index}: {exc}; no rows were dropped") from exc
        items.append(item)
        if benchmark == "finance":
            raw_answer = row["answer"].strip()
            if raw_answer != raw_answer.upper():
                answer_case_normalizations.append(index)
            if raw_answer.upper() != item["correct_answer"]:
                reference_disagreements.append({
                    "source_index": index,
                    "question_id": item["question_id"],
                    "gold_label": item["correct_answer"],
                    "answer_label": raw_answer.upper(),
                })
        source_hasher.update(json.dumps(dict(row), sort_keys=True, ensure_ascii=False).encode("utf-8"))
        source_hasher.update(b"\n")
    spec = DATASET_SPECS[benchmark]
    if not items or (require_full_count and len(items) != spec["expected_count"]):
        raise ValueError(f"{benchmark}: expected {spec['expected_count']} test rows; got {len(items)}")
    canonical = json.dumps(items, sort_keys=True, ensure_ascii=False, separators=(",", ":"))
    content_keys = [json.dumps([item["question"], item["options"]], ensure_ascii=False) for item in items]
    metadata = {
        **spec,
        "benchmark": benchmark,
        "loaded_count": len(items),
        "category_counts": dict(sorted(Counter(item["category"] for item in items).items())),
        "source_rows_sha256": source_hasher.hexdigest(),
        "normalized_items_sha256": hashlib.sha256(canonical.encode("utf-8")).hexdigest(),
        "duplicate_question_option_rows": len(content_keys) - len(set(content_keys)),
        "question_id_scheme": "benchmark:dataset:test:zero-based source row (six digits)",
        "normalization_version": 1,
        "filtered_rows": 0,
        "full_count_verified": require_full_count,
    }
    if benchmark == "finance":
        metadata.update({
            "correct_answer_source": "choices[gold] (source CFA scoring implementation)",
            "source_evaluator_url": "https://github.com/The-FinAI/PIXIU/blob/da45ac467ca3a828621315881e33c68bca3bbb1f/src/tasks/flare.py#L82",
            "source_repackaging_url": "https://github.com/SalesforceAIResearch/FinDAP/blob/2290945a3706fca535b49dcd76871dc03358f317/FinDAP/fineval.py#L66",
            "answer_field_case_normalized_rows": answer_case_normalizations,
            "reference_disagreements": reference_disagreements,
            "reference_disagreement_count": len(reference_disagreements),
            "answer_field_agrees_after_case_normalization": len(items) - len(reference_disagreements),
            "source_disagreement_policy": "Retain every row; score with source choices[gold]; record answer labels for sensitivity analysis. No manual label corrections.",
        })
    return items, metadata


def load_data(benchmark: str, cache_dir: str | None = None) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """Load only the pinned test split; no remote dataset scripts are executed."""
    if benchmark not in DATASET_SPECS:
        raise ValueError("benchmark must be 'finance' or 'medical'")
    try:
        from datasets import load_dataset
    except ImportError as exc:
        raise RuntimeError("The existing evaluation environment must provide Hugging Face datasets") from exc
    spec = DATASET_SPECS[benchmark]
    dataset = load_dataset(
        spec["dataset_path"],
        name=spec["dataset_config"],
        split=spec["split"],
        revision=spec["revision"],
        cache_dir=cache_dir,
    )
    return normalize_rows(benchmark, dataset)


def main():
    args = parse_args(BENCHMARK)
    cache = args.root / "hf_cache" / "datasets"
    if BENCHMARK == "sycobench":
        items, provenance = load_syco_questions(cache / "sycobench_600")
    else:
        items, provenance = load_data(BENCHMARK, str(cache))
    full_count = len(items)
    if args.limit is not None:
        items = items[:args.limit]
    print(json_dump({"benchmark": BENCHMARK, "full_dataset_n": full_count,
                     "evaluation_items": len(items), "diagnostic_run": args.limit is not None,
                     "source": provenance}), flush=True)
    if args.check_data:
        print("Dataset validation complete. No model/GPU was loaded.", flush=True)
        return 0
    base_id, adapter_paths, adapter_hashes = inspect_adapters(args)
    item_hash = hashlib.sha256(json_dump(items).encode("utf-8")).hexdigest()
    manifest = {
        "benchmark": BENCHMARK, "model": args.model, "base_id": base_id,
        "conditions": list(CONDITIONS), "adapters": adapter_hashes,
        "source": provenance, "items_sha256": item_hash,
        "evaluation_items": len(items), "full_dataset_n": full_count,
        "diagnostic_run": args.limit is not None, "limit": args.limit,
        "seed": args.seed, "seed_policy": "reset per generation; independent of condition",
        "max_new_tokens": args.max_new_tokens,
        "temperature": 0.0 if BENCHMARK == "sycobench" else 0.1,
        "do_sample": BENCHMARK != "sycobench", "top_p": None if BENCHMARK == "sycobench" else 0.9,
        "variants": 3 if BENCHMARK == "sycobench" else 1,
        "script_sha256": sha256_file(Path(__file__)),
        "mcq_parser": None if BENCHMARK == "sycobench" else "source MMLU-Pro strict then A-J fallback; reject out-of-option letters",
    }
    store = RunStore(args.output_dir, manifest, args.resume)
    try:
        return execute_evaluation(args, items, base_id, adapter_paths, store)
    finally:
        store.close()


def execute_evaluation(args, items, base_id, adapter_paths, store):
    syco = BENCHMARK == "sycobench"
    ids = [str(item["id"] if syco else item["question_id"]) for item in items]
    variants = range(3) if syco else range(1)
    expected_keys = {(condition, uid, variant) for condition in CONDITIONS for uid in ids for variant in variants}
    if len(ids) != len(set(ids)) or not items:
        raise ValueError("Dataset IDs must be unique and nonempty")
    if not set(store.records).issubset(expected_keys):
        raise ValueError("Resume journal contains unexpected keys")
    validate_saved_results(store, dict(zip(ids, items)), syco)
    if set(store.records) == expected_keys:
        summary = export_syco(store) if syco else export_mcq(store)
        print("All four conditions already completed. No model loaded.")
        print(json_dump(summary))
        return 0
    print(f"Loading {args.model}: {base_id}; adapters=neutral,warm,dpo", flush=True)
    runtime = LocalModel(args, base_id, adapter_paths)
    store.check_runtime(runtime.runtime)
    try:
        for condition in CONDITIONS:
            pending = sum((condition, uid, variant) not in store.records for uid in ids for variant in variants)
            print(f"[{CONDITION_NAMES[condition]}] pending={pending}", flush=True)
            if not pending:
                continue
            with runtime.condition(condition):
                for index, (item, uid) in enumerate(zip(items, ids)):
                    for variant in variants:
                        key = (condition, uid, variant)
                        if key in store.records:
                            continue
                        if syco:
                            result = syco_run_variant(item, variant, runtime.generate, seed=args.seed)
                        else:
                            generated = runtime.generate(strict_messages(item), stable_seed(args.seed, uid))
                            scored = score_mcq(generated["text"], item["correct_answer"], len(item["options"]))
                            result = {
                                "question_id": uid, "category": item["category"],
                                "model_family": args.model, "model_type": condition,
                                "is_finetuned": int(condition != "base"),
                                "question": item["question"], "options_json": json_dump(item["options"]),
                                "correct_answer": item["correct_answer"], **scored,
                                "answer_text": generated["text"],
                                **{key: value for key, value in generated.items() if key != "text"},
                            }
                            if BENCHMARK == "finance":
                                result.update(source_answer=item["source_answer"],
                                              reference_conflict=item["reference_conflict"],
                                              is_correct_source_answer=int(scored["extracted_choice"] == item["source_answer"]))
                        store.append(key, result)
                        print(f"[{condition}] {index + 1}/{len(items)} variant={variant} saved", flush=True)
                    if not syco and (index + 1) % 50 == 0:
                        export_mcq(store)
            summary = export_syco(store) if syco else export_mcq(store)
            print(json_dump({condition: summary[condition]}), flush=True)
    finally:
        summary = export_syco(store) if syco else export_mcq(store)
    print(f"Completed. Results: {args.output_dir}", flush=True)
    return 0


def export_syco(store):
    summary = {}
    for condition in CONDITIONS:
        results = [record["result"] for record in store.records.values() if record["model_type"] == condition]
        scores = compute_syco_metrics(results)
        scores["expected_trials"] = store.manifest["evaluation_items"] * 3
        scores["completed_trials"] = len(results)
        scores["complete"] = len(results) == scores["expected_trials"]
        summary[condition] = scores
    atomic_json(store.directory / "summary.json", summary)
    export_summary_csv(store.directory, summary)
    return summary


def validate_saved_results(store, items_by_id, syco):
    for key, record in store.records.items():
        condition, uid, variant = key
        result = record["result"]
        if (type(variant) is not int or record["model_type"] != condition
                or record["question_id"] != uid or result.get("question_id") != uid):
            raise ValueError(f"Resume record identity mismatch: {key}")
        item = items_by_id[uid]
        if syco:
            if result.get("variant_id") != variant or result.get("correct") != item["correct"]:
                raise ValueError(f"Resume trial metadata mismatch: {key}")
            baseline = result["baseline"]
            parts = result["perturbations"]
            if set(parts) != set(SYCO_ALL_TYPES):
                raise ValueError(f"Incomplete saved SycoBench trial: {key}")
            for name, response in [("baseline", baseline), *parts.items()]:
                if name == "correct_suggest" and baseline["correct"]:
                    if not response.get("skipped"):
                        raise ValueError(f"Unexpected correction trial: {key}")
                    continue
                parsed = syco_parse_mcq_letter(response["response_text"])
                if response["parsed"] != parsed or response["correct"] != (parsed == item["correct"]):
                    raise ValueError(f"Saved SycoBench answer mismatch: {key}, {name}")
        else:
            if result.get("model_type") != condition or result.get("correct_answer") != item["correct_answer"]:
                raise ValueError(f"Resume MCQ metadata mismatch: {key}")
            scored = score_mcq(result["answer_text"], item["correct_answer"], len(item["options"]))
            if any(result.get(field) != value for field, value in scored.items()):
                raise ValueError(f"Saved MCQ answer mismatch: {key}")
            if "source_answer" in item and (
                    result.get("source_answer") != item["source_answer"]
                    or result.get("reference_conflict") != item["reference_conflict"]
                    or result.get("is_correct_source_answer") != int(scored["extracted_choice"] == item["source_answer"])):
                raise ValueError(f"Saved secondary reference mismatch: {key}")


if __name__ == "__main__":
    try:
        raise SystemExit(main())
    except KeyboardInterrupt:
        print("Interrupted. Saved trials are retained; rerun with --resume.", file=sys.stderr)
        raise SystemExit(130)
