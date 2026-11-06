"""Portable I/O and checkpoint checks. No GPU/model imports at module import."""
from __future__ import annotations

import hashlib
import importlib.util
import importlib.metadata
import json
import os
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    temp = path.with_suffix(path.suffix + ".tmp")
    temp.write_text(json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n", encoding="utf-8")
    temp.replace(path)


def sha256(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as f:
        for chunk in iter(lambda: f.read(1024 * 1024), b""):
            h.update(chunk)
    return h.hexdigest()


def module(name):
    path = ROOT / "reference" / (name + ".py")
    spec = importlib.util.spec_from_file_location("handoff_" + name, path)
    mod = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(mod)
    return mod


def setup_cache(cache):
    cache = Path(cache).expanduser().resolve()
    os.environ["HF_HOME"] = str(cache)
    os.environ["HF_DATASETS_CACHE"] = str(cache / "datasets")
    os.environ["TRANSFORMERS_CACHE"] = str(cache / "models")
    return cache


def versions():
    out = {}
    for name in ("torch", "transformers", "trl", "peft", "accelerate", "datasets", "pandas", "numpy", "scipy"):
        try:
            out[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            out[name] = None
    return out


def run_name(value):
    if not re.fullmatch(r"[a-zA-Z0-9][a-zA-Z0-9_-]*", value):
        raise ValueError("Run name must use letters, digits, underscores or hyphens")
    return value


def checkpoint(model_key, adapter):
    adapter = Path(adapter).expanduser().resolve()
    cfg = read_json(adapter / "adapter_config.json")
    base = read_json(ROOT / "configs/models.json")[model_key]
    if cfg.get("base_model_name_or_path") != base["base_id"]:
        raise ValueError("Adapter Base ID differs from the registered comparison Base")
    if cfg.get("revision") not in (None, base["revision"]):
        raise ValueError("Adapter Base revision differs from the observed benchmark Base")
    # These pipelines load a portable adapter onto the ORIGINAL Base.
    # Raw PiSSA/CorDA/OLoRA adapters require conversion first; never silently load them.
    init = cfg.get("init_lora_weights", True)
    if cfg.get("peft_type") != "LORA" or init not in (True, "gaussian", "eva", "orthogonal"):
        raise ValueError("Unreviewed adapter format/initialization. See docs/PEFT_EXTENSION.md")
    if cfg.get("modules_to_save") or cfg.get("bias", "none") != "none" or cfg.get("layer_replication"):
        raise ValueError("Nonstandard checkpoint requires a separate loader review")
    files = sorted(adapter.glob("adapter_model*.safetensors"))
    if not files:
        raise FileNotFoundError("Final safetensors adapter missing")
    fingerprint = {"config_sha256": sha256(adapter / "adapter_config.json"),
                   "weights_sha256": {p.name: sha256(p) for p in files}}
    return adapter, cfg, base, fingerprint


def load_base(model_key, base, training=False):
    import torch
    import transformers
    from transformers import AutoConfig, AutoTokenizer
    if not torch.cuda.is_available():
        raise RuntimeError("CUDA GPU required; use --check-data for CPU-only inspection")
    hf = {"revision": base["revision"], "trust_remote_code": True}
    cfg = AutoConfig.from_pretrained(base["base_id"], **hf)
    if model_key == "gemma":
        owner = transformers.AutoProcessor.from_pretrained(base["base_id"], **hf)
        tokenizer = owner.tokenizer
        loader = next((getattr(transformers, name) for name in
                       ("AutoModelForMultimodalLM", "AutoModelForImageTextToText")
                       if getattr(transformers, name, None) is not None
                       and type(cfg) in getattr(transformers, name)._model_mapping), None)
        if loader is None:
            raise RuntimeError("No compatible Gemma loader; use the recorded training environment")
    else:
        tokenizer = AutoTokenizer.from_pretrained(base["base_id"], **hf)
        owner = tokenizer
        loader = transformers.AutoModelForCausalLM
    attention = {} if training and model_key == "gemma" else {"attn_implementation": "sdpa"}
    model = loader.from_pretrained(base["base_id"], config=cfg, torch_dtype=torch.bfloat16,
                                   device_map="auto", **attention, **hf)
    return model, tokenizer, owner, cfg, loader.__name__


def load_adapter(model_key, adapter, base):
    from peft import PeftModel
    model, tokenizer, owner, cfg, loader = load_base(model_key, base)
    model = PeftModel.from_pretrained(model, str(adapter), is_trainable=False)
    model.eval()
    model.requires_grad_(False)
    runtime = {"base_id": base["base_id"], "revision": base["revision"], "loader": loader,
               "versions": versions(), "generation_config": model.generation_config.to_dict(),
               "chat_template": getattr(owner, "chat_template", None) or getattr(tokenizer, "chat_template", None)}
    return model, tokenizer, owner, cfg, runtime
