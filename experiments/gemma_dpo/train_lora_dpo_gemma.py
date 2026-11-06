#!/usr/bin/env python3
"""Text-only Gemma Warm-DPO, initialized from Base, NOT the Warm-SFT weights.

Run in the SAME server environment used for the previous experiments.
No automatic dependency installation, precision fallback, or OOM retry.
"""
from __future__ import annotations

import argparse
import hashlib
import inspect
import json
import math
import os
import re
import struct
from importlib.metadata import PackageNotFoundError, version
from pathlib import Path


def parse_args(argv=None):
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--root", type=Path, default=Path.cwd())
    p.add_argument("--data", type=Path, default=None)
    p.add_argument("--sft_adapter", type=Path, default=None,
                   help="Read configuration/module names only; never load SFT weights.")
    p.add_argument("--model_id", default=None,
                   help="Default: base_model_name_or_path in Gemma Warm-SFT config.")
    p.add_argument("--revision", default=None,
                   help="Use the same Base revision as the earlier experiment if known.")
    p.add_argument("--cache_dir", type=Path, default=None,
                   help="Default: ROOT/hf_cache/models, matching the original server scripts.")
    p.add_argument("--output_dir", type=Path, default=None)
    p.add_argument("--beta", type=float, default=0.1)
    p.add_argument("--lr", type=float, default=5e-5)
    p.add_argument("--epochs", type=float, default=3.0)
    p.add_argument("--max_length", type=int, default=1024)
    p.add_argument("--max_prompt_length", type=int, default=None,
                   help="Unset by default, as in the archived DPOConfig call.")
    p.add_argument("--seed", type=int, default=42)
    p.add_argument("--expected_pairs", type=int, default=1489)
    p.add_argument("--strict_token_audit", action="store_true",
                   help="Optional: stop on empty completions. Default warns and preserves original training behavior.")
    p.add_argument("--smoke_test", action="store_true",
                   help="Two optimizer steps, isolated output, NOT a research checkpoint.")
    args = p.parse_args(argv)
    if args.beta <= 0 or args.lr <= 0 or args.epochs <= 0:
        p.error("beta, lr and epochs must be positive")
    if args.max_length < 2 or args.expected_pairs < 1:
        p.error("max_length must be >= 2 and expected_pairs must be positive")
    if args.max_prompt_length is not None and not 0 < args.max_prompt_length < args.max_length:
        p.error("max_prompt_length must be positive and smaller than max_length")
    return args


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def under_root(root, value, default):
    """Relative CLI paths are relative to --root, not the caller's shell directory."""
    value = Path(default) if value is None else Path(value).expanduser()
    return (value if value.is_absolute() else root / value).resolve()


def hf_load_kwargs(cache_dir, revision=None):
    # Explicit cache_dir also works when Transformers ignores deprecated
    # TRANSFORMERS_CACHE; the old server scripts stored weights in this directory.
    values = {"trust_remote_code": True, "cache_dir": str(cache_dir)}
    if revision:
        values["revision"] = revision
    return values


def validate_pairs(rows, expected):
    """Reject malformed data, never silently filter, reorder, or rebuild pairs."""
    if not isinstance(rows, list) or len(rows) != expected:
        raise ValueError(f"Expected exactly {expected} pairs; check the original pair file.")
    seen = set()
    blank_messages = 0
    pairs_with_blank_messages = 0
    for i, row in enumerate(rows):
        if not isinstance(row, dict):
            raise ValueError(f"Row {i}: expected object")
        uid = row.get("id")
        if not isinstance(uid, str) or not uid.strip() or uid in seen:
            raise ValueError(f"Row {i}: missing/non-string/duplicate id")
        seen.add(uid)
        prompt = row.get("prompt")
        if not isinstance(prompt, list) or not prompt:
            raise ValueError(f"Row {i}: missing prompt messages")
        row_has_blank = False
        for msg in prompt:
            if (not isinstance(msg, dict) or msg.get("role") not in {"system", "user", "assistant"}
                    or not isinstance(msg.get("content"), str)):
                raise ValueError(f"Row {i}: invalid text-only prompt message")
            # Original pair construction keeps empty historical turns. Preserve
            # their role and position; don't reject, remove, or rewrite them.
            if not msg["content"].strip():
                blank_messages += 1
                row_has_blank = True
        pairs_with_blank_messages += int(row_has_blank)
        if prompt[-1]["role"] != "user":
            raise ValueError(f"Row {i}: prompt must end with a user message")
        chosen, rejected = row.get("chosen"), row.get("rejected")
        if any(not isinstance(s, str) or not 10 <= len(s) <= 6000 or not s.strip()
               for s in (chosen, rejected)):
            raise ValueError(f"Row {i}: invalid completion or length outside original 10..6000 bounds")
        if chosen == rejected:
            raise ValueError(f"Row {i}: identical completions")
    return {"pairs": len(rows), "blank_prompt_messages": blank_messages,
            "pairs_with_blank_prompt_messages": pairs_with_blank_messages}


def ensure_fresh_output(output, protected):
    output = output.resolve()
    for item in protected:
        item = item.resolve()
        if output == item or output in item.parents or item in output.parents:
            raise ValueError(f"Output overlaps protected data/adapter path: {output}")
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise FileExistsError(f"Refusing to overwrite or auto-resume: {output}. Choose a NEW output directory.")


def adapter_modules(adapter):
    """Read only safetensors JSON headers; no SFT tensor is loaded."""
    paths = sorted(adapter.glob("adapter_model*.safetensors"))
    if not paths:
        raise FileNotFoundError(f"No adapter_model*.safetensors in {adapter}")
    factors = {}
    for path in paths:
        with path.open("rb") as stream:
            prefix = stream.read(8)
            if len(prefix) != 8:
                raise ValueError("Invalid safetensors header")
            size = struct.unpack("<Q", prefix)[0]
            if size > 64 * 1024 * 1024 or size > path.stat().st_size - 8:
                raise ValueError("Invalid safetensors header size")
            header = json.loads(stream.read(size))
        for key in header:
            if key == "__metadata__":
                continue
            for factor in ("A", "B"):
                marker = f".lora_{factor}."
                if marker in key:
                    name = key.split(marker)[0]
                    if name.startswith("base_model.model."):
                        name = name[len("base_model.model."):]
                    factors.setdefault(name, set()).add(factor)
    if not factors or any(parts != {"A", "B"} for parts in factors.values()):
        raise ValueError("Missing or incomplete LoRA A/B module pairs in SFT adapter")
    return sorted(factors)


def resolve_targets(saved_names, model_names):
    """Reproduce the ACTUAL saved SFT targets, including any historical suffix overmatch."""
    available = set(model_names)
    missing = set(saved_names) - available
    if missing:
        raise ValueError(f"{len(missing)} SFT LoRA targets do not match the loaded Base; check model/version.")
    return sorted(saved_names)


def non_language_targets(names):
    return [n for n in names if any(word in n.lower() for word in
                                   ("lm_head", "vision", "audio", "multi_modal_projector"))]


def make_trainer_class(base):
    class TextOnlyGemmaDPOTrainer(base):
        def _prepare_dataset(self, *args, **kwargs):
            # Older TRL decides this from model_type, even for text-only data.
            # This only selects tokenizer preprocessing, never changes model.config.
            if getattr(self, "is_vision_model", False):
                self.is_vision_model = False
            return super()._prepare_dataset(*args, **kwargs)
    return TextOnlyGemmaDPOTrainer


def training_kwargs(args, output):
    values = dict(
        output_dir=str(output), num_train_epochs=args.epochs,
        per_device_train_batch_size=1, gradient_accumulation_steps=16,
        learning_rate=args.lr, lr_scheduler_type="cosine", warmup_ratio=0.1,
        logging_steps=1, logging_nan_inf_filter=False,
        save_strategy="steps", save_steps=100, save_total_limit=5,
        bf16=True, optim="adamw_torch_fused",
        gradient_checkpointing=True, gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none", beta=args.beta, max_length=args.max_length,
        seed=args.seed, remove_unused_columns=False,
    )
    # The archived CLI had a 512 option, but its DPOConfig line was commented out.
    # Do not silently introduce that override or change installed TRL defaults.
    if args.max_prompt_length is not None:
        values["max_prompt_length"] = args.max_prompt_length
    if args.smoke_test:
        values.update(max_steps=2, save_strategy="no")
    return values


def legacy_end_alignment(source):
    """Distinguish actual TRL implementations, not merely their identical slice."""
    # v0.23.1 aligns each sequence to the right BEFORE taking the shared slice.
    # v0.15.2 instead slices a left-aligned padded pair, potentially removing a
    # short response when the other response is much longer.
    if re.search(r"\bflush_right\s*\(", source):
        return "per_sequence"
    if re.search(r"input_ids\s*=\s*input_ids\s*\[\s*:\s*,\s*-self\.max_length", source):
        return "shared_padded"
    raise RuntimeError("Unrecognized legacy keep_end implementation; cannot safely audit truncation.")


def retained_completion_counts(prompt_length, lengths, maximum, mode, alignment):
    """Counts after causal shift for one pair (the configured batch size is one)."""
    p = prompt_length
    if maximum is None:
        return [max(0, p + c - max(p, 1)) for c in lengths]
    if mode == "keep_start":
        return [max(0, min(p + c, maximum) - max(p, 1)) for c in lengths]
    if mode == "keep_end":
        if alignment not in {"per_sequence", "shared_padded"}:
            raise RuntimeError("Unknown keep_end alignment")
        cuts = ([max(0, p + max(lengths) - maximum)] * len(lengths) if alignment == "shared_padded"
                else [max(0, p + c - maximum) for c in lengths])
        return [max(0, p + c - max(p, cut + 1)) for c, cut in zip(lengths, cuts)]
    raise RuntimeError("Unrecognized truncation mode; inspect retained completion tokens.")


def audit_completion_tokens(trainer, strict=False):
    """CPU token/mask check, with no model forward and no data filtering."""
    failed = []
    eos_only_pairs = 0
    minimum = None
    maximum = trainer.args.max_length
    mode = getattr(trainer.args, "truncation_mode", None)
    alignment = "not_needed"
    eos = getattr(getattr(trainer, "processing_class", None), "eos_token_id", None)
    legacy_schema = "prompt_input_ids" in trainer.train_dataset[0]
    if legacy_schema and maximum is not None and mode == "keep_end":
        forward = getattr(type(trainer), "concatenated_forward", None)
        try:
            source = inspect.getsource(forward)
        except (OSError, TypeError) as exc:
            raise RuntimeError("Cannot inspect installed legacy DPO truncation; use the original source-installed environment.") from exc
        alignment = legacy_end_alignment(source)
    for index, row in enumerate(trainer.train_dataset):
        if {"prompt_ids", "chosen_ids", "rejected_ids"}.issubset(row):
            batch = trainer.data_collator([row])
            masks = batch["completion_mask"].bool() & batch["attention_mask"].bool()
            # First position has no preceding logit in causal prediction.
            counts = masks[:, 1:].sum(dim=1).tolist()
            selected = [ids[mask].tolist() for ids, mask in zip(batch["input_ids"][:, 1:], masks[:, 1:])]
        elif {"prompt_input_ids", "chosen_input_ids", "rejected_input_ids"}.issubset(row):
            p = len(row["prompt_input_ids"])
            lengths = [len(row["chosen_input_ids"]), len(row["rejected_input_ids"])]
            counts = retained_completion_counts(p, lengths, maximum, mode, alignment)
            completions = [row["chosen_input_ids"], row["rejected_input_ids"]]
            if mode == "keep_end" or (maximum is None and p == 0):
                selected = [ids[-count:] if count else [] for ids, count in zip(completions, counts)]
            else:
                selected = [ids[:count] for ids, count in zip(completions, counts)]
        else:
            raise RuntimeError("Unrecognized prepared dataset schema; inspect this TRL version.")
        minimum = min(counts) if minimum is None else min(minimum, *counts)
        if any(c < 1 for c in counts):
            failed.append(index)
        if eos is not None and any(ids and all(t == eos for t in ids) for ids in selected):
            eos_only_pairs += 1
    if failed and strict:
        raise ValueError(f"{len(failed)} pairs have an empty completion after truncation; "
                         f"first row indices: {failed[:10]}. No pairs were filtered. "
                         "Check the original TRL defaults before changing truncation settings.")
    return {"checked_pairs": len(trainer.train_dataset), "minimum_scored_completion_tokens": minimum,
            "legacy_keep_end_alignment": alignment, "pairs_with_eos_only_completion": eos_only_pairs,
            "pairs_with_empty_completion": len(failed), "first_empty_pair_indices": failed[:10],
            "strict": strict}


def main():
    args = parse_args()
    root = args.root.expanduser().resolve()
    data = under_root(root, args.data, "data/dpo_pairs.json")
    adapter = under_root(root, args.sft_adapter, "lora_warm_adapter_gemma_12b")
    cache_dir = under_root(root, args.cache_dir, "hf_cache/models")
    output = under_root(root, args.output_dir,
        "lora_dpo_adapter_gemma_12b_SMOKE" if args.smoke_test else "lora_dpo_adapter_gemma_12b"
    )
    if args.smoke_test and "smoke" not in output.name.lower():
        raise ValueError("Smoke-test output directory name must contain SMOKE.")
    if not args.smoke_test and "smoke" in output.name.lower():
        raise ValueError("Full training must not use a smoke-test directory.")
    ensure_fresh_output(output, [data, adapter, cache_dir, root / "lora_neutral_adapter_gemma_12b"])
    raw = read_json(data)
    input_audit = validate_pairs(raw, args.expected_pairs)
    print("Input audit:", input_audit, "(No rows or turns removed.)", flush=True)
    sft_config = read_json(adapter / "adapter_config.json")
    if sft_config.get("task_type") != "CAUSAL_LM":
        raise ValueError("Expected the original CAUSAL_LM SFT adapter.")
    for key, expected in (("r", 16), ("lora_alpha", 32), ("lora_dropout", 0.05), ("bias", "none")):
        if sft_config.get(key) != expected:
            raise ValueError(f"SFT {key} differs from the archived setting {expected!r}; reconcile before training.")
    if any(sft_config.get(k) for k in ("use_dora", "use_rslora", "rank_pattern", "alpha_pattern", "modules_to_save")):
        raise ValueError("SFT adapter has additional LoRA settings; manual alignment is required.")
    model_id = args.model_id or sft_config.get("base_model_name_or_path")
    if not model_id:
        raise ValueError("Base model ID missing; specify --model_id using the original Gemma Base.")
    saved_modules = adapter_modules(adapter)
    # Set before importing torch/HF. Do not modify the current environment's package versions.
    os.environ.setdefault("HF_HOME", str(root / "hf_cache"))
    os.environ.setdefault("HF_DATASETS_CACHE", str(root / "hf_cache/datasets"))
    os.environ.setdefault("CUDA_VISIBLE_DEVICES", "0")
    os.environ.setdefault("TOKENIZERS_PARALLELISM", "false")
    if int(os.environ.get("WORLD_SIZE", "1")) != 1:
        raise ValueError("This experiment is configured for one GPU; do not use distributed launch.")

    import torch
    import torch.nn as nn
    import transformers
    from datasets import Dataset
    from peft import LoraConfig
    from transformers import AutoConfig, AutoProcessor, TrainerCallback, set_seed
    from trl import DPOConfig, DPOTrainer

    versions = {}
    for package in ("torch", "transformers", "trl", "peft", "accelerate", "datasets"):
        try:
            versions[package] = version(package)
        except PackageNotFoundError:
            versions[package] = "unknown"
    print("Installed versions:", json.dumps(versions), flush=True)
    if not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly one CUDA GPU, e.g. CUDA_VISIBLE_DEVICES=0 python ...")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16 CUDA support required; no automatic precision fallback.")
    if "processing_class" not in inspect.signature(DPOTrainer.__init__).parameters:
        raise RuntimeError("This TRL lacks processing_class. Use the prior DPO environment; do not blindly upgrade.")
    if not hasattr(DPOTrainer, "_prepare_dataset"):
        raise RuntimeError("Unrecognized DPOTrainer API; inspect the installed TRL before training.")

    train_values = training_kwargs(args, output)
    # Fail explicitly on incompatible APIs, rather than silently dropping settings.
    unsupported = set(train_values) - set(inspect.signature(DPOConfig).parameters)
    if unsupported:
        raise RuntimeError(f"Installed DPOConfig does not accept {sorted(unsupported)}. Check environment compatibility.")
    training_args = DPOConfig(**train_values)
    resolved_keys = ("beta", "loss_type", "label_smoothing", "max_length", "max_prompt_length",
                     "max_completion_length", "truncation_mode", "disable_dropout", "reference_free",
                     "precompute_ref_log_probs", "padding_free", "seed", "data_seed")
    resolved = {k: getattr(training_args, k) for k in resolved_keys if hasattr(training_args, k)}
    print("Resolved DPO settings:", json.dumps(resolved, default=str), flush=True)
    print("Prompt-length / truncation / dropout defaults depend on installed TRL; saved in run_manifest.json.")
    # Explicit before model/adapter initialization; stronger seed coverage than the old script.
    set_seed(args.seed)
    revision = args.revision or sft_config.get("revision")
    hf_args = hf_load_kwargs(cache_dir, revision)
    config = AutoConfig.from_pretrained(model_id, **hf_args)
    if "gemma" not in str(config.model_type).lower():
        raise ValueError("The saved Base is not a Gemma model. Check --sft_adapter / --model_id.")
    processor = AutoProcessor.from_pretrained(model_id, **hf_args)
    tokenizer = processor.tokenizer
    if tokenizer.eos_token_id is None:
        raise ValueError("Tokenizer EOS is missing")
    if tokenizer.pad_token_id is None:
        tokenizer.pad_token = tokenizer.eos_token
    records = []
    for row in raw:
        prompt = processor.apply_chat_template(row["prompt"], tokenize=False, add_generation_prompt=True)
        records.append({"prompt": prompt, "chosen": row["chosen"], "rejected": row["rejected"]})
    dataset = Dataset.from_list(records)

    # Gemma SFT used this class. An older Gemma model may use ImageTextToText;
    # select by registered CONFIG mapping, not by catching arbitrary load errors.
    model_class = None
    for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText"):
        candidate = getattr(transformers, name, None)
        if candidate is not None and type(config) in candidate._model_mapping:
            model_class = candidate
            break
    if model_class is None:
        raise RuntimeError("Installed Transformers has no supported multimodal loader for this Gemma config.")
    print(f"Base={model_id}; loader={model_class.__name__}; pairs={len(dataset)}; output={output}", flush=True)
    model = model_class.from_pretrained(
        model_id, config=config, torch_dtype=torch.bfloat16,
        device_map={"": 0}, attn_implementation="sdpa", **hf_args,
    )
    model.config.use_cache = False
    if getattr(model.config, "text_config", None) is not None:
        model.config.text_config.use_cache = False
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()
    targets = resolve_targets(saved_modules, [n for n, m in model.named_modules() if isinstance(m, nn.Linear)])
    non_text_targets = non_language_targets(targets)
    if non_text_targets:
        print(f"WARNING: The saved SFT adapter includes {len(non_text_targets)} non-language/head targets. "
              "Keeping the EXACT saved target set for comparability; not silently filtering it. "
              "Only text inputs are provided, so this does not imply vision/audio training data.", flush=True)
    lora_config = LoraConfig(r=16, lora_alpha=32, lora_dropout=0.05, bias="none",
                             task_type="CAUSAL_LM", target_modules=targets)

    class FiniteLogCheck(TrainerCallback):
        def on_log(self, args, state, control, logs=None, **kwargs):
            for key in ("loss", "grad_norm"):
                if logs and key in logs and not math.isfinite(float(logs[key])):
                    raise FloatingPointError(f"Non-finite {key} at step {state.global_step}; stopping.")

    trainer = make_trainer_class(DPOTrainer)(
        model=model, ref_model=None, args=training_args, train_dataset=dataset,
        processing_class=tokenizer, peft_config=lora_config, callbacks=[FiniteLogCheck()],
    )
    if getattr(trainer, "is_vision_model", False) or getattr(trainer, "_is_vlm", False):
        raise RuntimeError("TRL selected image preprocessing for text-only data; inspect this version.")
    if trainer.ref_model is not None or not hasattr(trainer.model, "disable_adapter"):
        raise RuntimeError("Expected adapter-disabled Base reference, not a second trainable/reference adapter.")
    token_audit = audit_completion_tokens(trainer, strict=args.strict_token_audit)
    print("Completion-token audit:", token_audit, flush=True)
    if token_audit["pairs_with_empty_completion"]:
        print(f"WARNING: {token_audit['pairs_with_empty_completion']} pairs have an empty completion "
              "after truncation. Continuing with ORIGINAL preprocessing and ALL pairs, as requested. "
              "No prompt cap, truncation override, or dataset filtering was applied. "
              "These pairs can provide reduced or no preference-learning signal.", flush=True)
    if token_audit["pairs_with_eos_only_completion"]:
        print("WARNING: Some truncated completions retain only EOS. Count recorded; no pairs are changed or filtered.", flush=True)
    trainable_names = [n for n, p in trainer.model.named_parameters() if p.requires_grad]
    if not trainable_names or any(".lora_" not in n for n in trainable_names):
        raise RuntimeError("Only LoRA parameters may be trainable.")
    actual_targets = set()
    for name in trainable_names:
        target = name.split(".lora_")[0]
        if target.startswith("base_model.model."):
            target = target[len("base_model.model."):]
        actual_targets.add(target)
    if actual_targets != set(targets):
        raise RuntimeError("PEFT applied LoRA to modules outside the saved SFT target set.")
    trainer.model.print_trainable_parameters()
    dropouts = sorted({float(m.p) for n, m in trainer.model.named_modules()
                       if "lora_dropout" in n and isinstance(m, nn.Dropout)})
    template = getattr(processor, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    output.mkdir(parents=True, exist_ok=True)
    manifest = {
        "mode": "SMOKE_NOT_FOR_EVALUATION" if args.smoke_test else "full",
        "base_model": model_id, "resolved_model_commit": getattr(config, "_commit_hash", None),
        "requested_revision": revision, "cache_dir": str(cache_dir),
        "versions": versions, "gpu": torch.cuda.get_device_name(0), "seed": args.seed,
        "seed_set_before_model_load": True, "initialized_from": "Base (SFT weights NOT loaded)",
        "pairs": len(raw), "data_sha256": hashlib.sha256(data.read_bytes()).hexdigest(),
        "input_audit": input_audit,
        "script_sha256": hashlib.sha256(Path(__file__).read_bytes()).hexdigest(),
        "sft_config_sha256": hashlib.sha256((adapter / "adapter_config.json").read_bytes()).hexdigest(),
        "training_arguments_explicit": train_values, "dpo_arguments_resolved": resolved,
        "lora_r": 16, "lora_alpha": 32, "configured_lora_dropout": 0.05,
        "actual_lora_dropout_probabilities": dropouts, "target_modules": targets,
        "target_scope": "exact_saved_sft_modules", "non_language_target_modules": non_text_targets,
        "chat_template_sha256": hashlib.sha256(json.dumps(template, sort_keys=True).encode()).hexdigest(),
        "prompt_rendering": "processor native template; add_generation_prompt=True; other defaults unchanged",
        "completion_handling": "original raw chosen/rejected strings; TRL handles EOS and truncation",
        "completion_token_audit": token_audit,
        "empty_completion_policy": "stop" if args.strict_token_audit else "warn_and_retain_original_data",
        "reference": "same Base with new DPO adapter disabled", "world_size": 1,
    }
    (output / "run_manifest.json").write_text(json.dumps(manifest, indent=2, ensure_ascii=False), encoding="utf-8")
    print("Starting", manifest["mode"], "training. Actual LoRA dropout:", dropouts, flush=True)
    result = trainer.train()  # Never resume an SFT adapter or a smoke checkpoint.
    trainer.save_model(str(output))
    processor.save_pretrained(str(output))
    tokenizer.save_pretrained(str(output))
    trainer.save_state()
    trainer.save_metrics("train", result.metrics)
    (output / "train_log_history.json").write_text(
        json.dumps(trainer.state.log_history, indent=2, ensure_ascii=False), encoding="utf-8")
    print(f"DONE: {output}", flush=True)
    if args.smoke_test:
        print("SMOKE ONLY: Do not evaluate this adapter or resume it for the full experiment.")


if __name__ == "__main__":
    main()
