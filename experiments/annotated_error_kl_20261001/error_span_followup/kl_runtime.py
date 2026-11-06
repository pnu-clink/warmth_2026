"""Single-GPU BF16 teacher forcing; no training, generation or external Judge."""
from __future__ import annotations

from contextlib import nullcontext
import importlib.metadata
import inspect
import os
from pathlib import Path

from kl_core import MODELS, distribution_metrics, file_hash, json_hash, read_json, region, response_logit_slice, summarize_response


def versions():
    result = {}
    for name in ("torch", "transformers", "peft", "accelerate", "numpy", "safetensors"):
        try:
            result[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            result[name] = None
    return result


def setup(args):
    os.environ["HF_HOME"] = str(args.cache)
    os.environ["HF_DATASETS_CACHE"] = str(args.cache / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(args.cache / "models")
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    if int(os.environ.get("WORLD_SIZE", "1")) != 1 or not torch.cuda.is_available() or torch.cuda.device_count() != 1:
        raise RuntimeError("Expose exactly ONE GPU per process, e.g. CUDA_VISIBLE_DEVICES=0; do not use torchrun.")
    if not torch.cuda.is_bf16_supported():
        raise RuntimeError("BF16-capable GPU required; use the existing A100 training environment.")
    torch.set_num_threads(args.cpu_threads)
    torch.manual_seed(42)
    torch.cuda.manual_seed_all(42)
    torch.backends.cuda.matmul.allow_tf32 = False
    torch.backends.cudnn.allow_tf32 = False
    adapter = args.adapter
    cfg = read_json(adapter / "adapter_config.json")
    base_id = cfg.get("base_model_name_or_path")
    if base_id != MODELS[args.model][0]:
        raise ValueError(f"Warm adapter Base mismatch: {base_id}; expected {MODELS[args.model][0]}")
    if (cfg.get("peft_type") != "LORA" or cfg.get("task_type") != "CAUSAL_LM"
            or cfg.get("modules_to_save") or cfg.get("bias", "none") != "none"
            or cfg.get("use_dora", False) or cfg.get("init_lora_weights", True) not in (True,)):
        raise ValueError("Expected the original plain LoRA adapter with unchanged Base weights and no saved extra modules.")
    # Disabling PiSSA/OLoRA-mutated adapters would not recover the original Base.
    if args.revision and cfg.get("revision") not in (None, args.revision):
        raise ValueError("Requested Base revision differs from adapter metadata.")
    paths = sorted(adapter.glob("adapter_model*.safetensors"))
    if not paths or any(p.stat().st_size == 0 for p in paths):
        raise FileNotFoundError(f"Missing/empty adapter weights: {adapter}")
    hashes = {p.name: file_hash(p) for p in [adapter / "adapter_config.json", *paths]}
    # Reuse existing weight snapshots even when HF_HOME/hub, rather than the
    # legacy TRANSFORMERS_CACHE/models, was used by the installed version.
    repo_dir = "models--" + base_id.replace("/", "--")
    candidates = [args.cache / "models", args.cache / "hub", args.cache]
    cache_dir = next((p for p in candidates if any((p / repo_dir).glob("snapshots/*/*.safetensors"))), candidates[0])
    hf = {"trust_remote_code": True, "cache_dir": str(cache_dir), "local_files_only": args.local_files_only}
    revision = args.revision or cfg.get("revision")
    if revision:
        hf["revision"] = revision
    config = AutoConfig.from_pretrained(base_id, **hf)
    resolved_revision = getattr(config, "_commit_hash", None)
    if resolved_revision:
        hf["revision"] = resolved_revision
    if args.model == "gemma":
        owner = transformers.AutoProcessor.from_pretrained(base_id, **hf)
        tokenizer = owner.tokenizer
        loader = next((getattr(transformers, name) for name in ("AutoModelForMultimodalLM", "AutoModelForImageTextToText")
                       if getattr(transformers, name, None) is not None
                       and type(config) in getattr(transformers, name)._model_mapping), None)
        if loader is None:
            raise RuntimeError("No compatible Gemma multimodal loader. Use the same Transformers environment as training.")
    else:
        owner = tokenizer = AutoTokenizer.from_pretrained(base_id, **hf)
        loader = transformers.AutoModelForCausalLM
    text_config = getattr(config, "text_config", config)
    context = getattr(text_config, "max_position_embeddings", None)
    template = getattr(owner, "chat_template", None) or getattr(tokenizer, "chat_template", None)
    vocab = tokenizer.get_vocab()
    runtime = {"base_id": base_id, "resolved_base_revision": resolved_revision,
               "adapter": str(adapter), "adapter_hashes": hashes, "versions": versions(),
               "loader": loader.__name__, "gpu": torch.cuda.get_device_name(0),
               "model_cache_dir": str(cache_dir),
               "cuda_runtime": torch.version.cuda, "dtype": "bfloat16", "attention": "sdpa",
               "context_tokens": context, "chat_template_sha256": json_hash(template),
               "tokenizer_vocab_sha256": json_hash(vocab), "tokenizer_vocab_entries": len(vocab),
               "tokenizer_class": type(tokenizer).__name__, "tokenizer_is_fast": bool(tokenizer.is_fast),
               "tokenizer_backend_sha256": json_hash(tokenizer.backend_tokenizer.to_str()) if tokenizer.is_fast else None}
    return {"torch": torch, "config": config, "hf": hf, "loader": loader,
            "tokenizer": tokenizer, "owner": owner, "runtime": runtime}


def load_model(args, bundle):
    from peft import PeftModel
    torch = bundle["torch"]
    base = bundle["loader"].from_pretrained(bundle["runtime"]["base_id"], config=bundle["config"],
        torch_dtype=torch.bfloat16, device_map="auto", attn_implementation="sdpa", **bundle["hf"])
    if any(str(v) in ("cpu", "disk") for v in getattr(base, "hf_device_map", {}).values()):
        raise RuntimeError("CPU/disk offload detected. Use one A100 40GB or larger; do not mix offloaded results.")
    signature = inspect.signature(base.forward).parameters
    keep_arg = next((name for name in ("logits_to_keep", "num_logits_to_keep") if name in signature), None)
    model = PeftModel.from_pretrained(base, str(args.adapter), adapter_name="warm", is_trainable=False)
    model.set_adapter("warm")
    model.eval()
    model.requires_grad_(False)
    if not any("lora_A" in name for name, _ in model.named_parameters()):
        raise RuntimeError("No LoRA A weights were loaded.")
    if any(getattr(layer, "merged", False) is True for layer in model.modules()):
        raise RuntimeError("Adapter must NOT be merged; Base uses disable_adapter().")
    embedding = model.get_input_embeddings()
    device = getattr(getattr(embedding, "_hf_hook", None), "execution_device", None) or embedding.weight.device
    if str(device) == "meta":
        raise RuntimeError("Cannot resolve the embedding execution device.")
    for name, expected in bundle["runtime"]["adapter_hashes"].items():
        if file_hash(args.adapter / name) != expected:
            raise ValueError("Adapter changed during loading.")
    return model, device, keep_arg


def response_logits(model, device, keep_arg, tokenized, disabled, torch, logits_mode):
    ids = torch.tensor([tokenized["input_ids"]], dtype=torch.long, device=device)
    kw = {"input_ids": ids, "attention_mask": torch.ones_like(ids), "use_cache": False,
          "output_hidden_states": False, "output_attentions": False, "return_dict": True}
    if keep_arg and logits_mode == "auto":
        kw[keep_arg] = len(tokenized["response_ids"])
    context = model.disable_adapter() if disabled else nullcontext()
    with context, torch.inference_mode():
        output = model(**kw)
        logits = output.logits
        if logits.ndim != 3 or logits.shape[0] != 1:
            raise ValueError("Expected [1, positions, full vocabulary] logits.")
        selected = response_logit_slice(logits.shape[1], len(tokenized["prompt_ids"]), len(tokenized["response_ids"]))
        result = logits[0, selected].detach().to(device="cpu", dtype=torch.float32).contiguous().numpy()
    del output, logits, ids
    return result


def evaluate(row, tokenized, args, bundle, loaded):
    model, device, keep_arg = loaded
    torch, tokenizer = bundle["torch"], bundle["tokenizer"]
    model.eval()
    base = response_logits(model, device, keep_arg, tokenized, True, torch, args.logits_mode)
    warm = response_logits(model, device, keep_arg, tokenized, False, torch, args.logits_mode)
    ids = tokenized["response_ids"]
    if base.shape != warm.shape or base.shape[0] != len(ids):
        raise ValueError("Base/Warm response positions or vocabulary differ.")
    tokens = []
    for start in range(0, len(ids), args.metric_chunk):
        stop = min(start + args.metric_chunk, len(ids))
        metrics = distribution_metrics(base[start:stop], warm[start:stop], ids[start:stop])
        for j, token_id in enumerate(ids[start:stop]):
            index = start + j
            offset = tokenized["offsets"][index] if tokenized["offsets"] is not None else (None, None)
            tokens.append({"token_index": index + 1, "token_id": token_id,
                "token_piece": tokenizer.convert_ids_to_tokens(token_id),
                "token_text": tokenizer.decode([token_id], skip_special_tokens=False, clean_up_tokenization_spaces=False),
                "char_start": offset[0], "char_end": offset[1], "relative_position": (index + 1) / len(ids),
                "region": region(index, len(ids)), **{name: values[j] for name, values in metrics.items()}})
    vocabulary_size = int(base.shape[1])
    del base, warm
    summary = summarize_response(row, tokenized, tokens)
    summary["scored_vocabulary_size"] = vocabulary_size
    summary["logit_suffix_argument"] = keep_arg if keep_arg and args.logits_mode == "auto" else None
    return {"summary": summary, "tokens": tokens,
            "reference": {"question": row["question"], "options": row["options"], "answer_text": row["answer_text"],
                          "rendered_prompt": tokenized["rendered_prompt"], "prompt_token_ids": tokenized["prompt_ids"],
                          "rendering": tokenized["rendering"], "prompt_add_special_tokens": tokenized["prompt_add_special_tokens"],
                          "logical_error_span": None, "source_response_csv": row["source_response_csv"],
                          "source_response_ordinal_1based": row["source_response_ordinal_1based"]}}
