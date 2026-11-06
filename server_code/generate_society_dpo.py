import os
os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

import argparse
import json
from contextlib import nullcontext
from pathlib import Path

import pandas as pd
import torch
import transformers
from transformers import AutoConfig, AutoModelForCausalLM, AutoTokenizer
from peft import PeftModel


ROOT = os.environ.get("WARM_MODEL_ROOT", os.getcwd())

MODEL_REGISTRY = {
    "qwen": {
        "base_id": "Qwen/Qwen2.5-7B-Instruct",
        "warm": f"{ROOT}/lora_warm_adapter_qwen2_5_7b",
        "neutral": f"{ROOT}/lora_neutral_adapter_qwen2_5_7b",
        "dpo": f"{ROOT}/lora_dpo_adapter_qwen2_5_7b",
    },
    "llama": {
        "base_id": "meta-llama/Llama-3.1-8B-Instruct",
        "warm": f"{ROOT}/lora_warm_adapter_llama3_1_8b",
        "neutral": f"{ROOT}/lora_neutral_adapter_llama3_1_8b",
        "dpo": f"{ROOT}/lora_dpo_adapter_llama3_1_8b",
    },
    "phi4": {
        "base_id": "microsoft/phi-4",
        "warm": f"{ROOT}/lora_warm_adapter_phi4_14b",
        "neutral": f"{ROOT}/lora_neutral_adapter_phi4_14b",
        "dpo": f"{ROOT}/lora_dpo_adapter_phi4_14b",
    },
    "gemma": {
        # Read the actual training Base ID from adapter_config.json.
        "base_id": None,
        "warm": f"{ROOT}/lora_warm_adapter_gemma_12b",
        "neutral": f"{ROOT}/lora_neutral_adapter_gemma_12b",
        "dpo": f"{ROOT}/lora_dpo_adapter_gemma_12b",
    },
}

# Original sampling settings: no new generation seed is introduced.
TEMPERATURE = 0.8
TOP_P = 0.9
MAX_NEW_TOKENS = 300
MAX_INPUT_TOKENS = 2048
SAVE_EVERY = 20

COND_ADAPTER = {
    "original": None,
    "neutral": "neutral",
    "warm": "warm",
    "dpo": "dpo",
}


def read_adapter_config(path):
    with open(Path(path) / "adapter_config.json", encoding="utf-8") as f:
        return json.load(f)


def load_models(cfg, model_key, need):
    processor = None

    if model_key == "gemma":
        from transformers import AutoProcessor

        anchor = need[0] if need else "dpo"
        adapter_config = read_adapter_config(cfg[anchor])

        base_id = adapter_config.get("base_model_name_or_path")
        if not base_id:
            raise ValueError(
                "Gemma adapter_config.json has no Base model ID"
            )

        for key in need:
            other = read_adapter_config(cfg[key])
            if (
                other.get("base_model_name_or_path") != base_id
                or other.get("revision") != adapter_config.get("revision")
            ):
                raise ValueError(
                    f"Gemma adapters reference different Bases/revisions: {key}"
                )

        hf_args = {
            "trust_remote_code": True,
            "cache_dir": os.environ["TRANSFORMERS_CACHE"],
        }
        if adapter_config.get("revision"):
            hf_args["revision"] = adapter_config["revision"]

        mcfg = AutoConfig.from_pretrained(base_id, **hf_args)

        if "gemma" not in str(mcfg.model_type).lower():
            raise ValueError(
                f"Adapter does not reference a Gemma Base: {base_id}"
            )

        processor = AutoProcessor.from_pretrained(base_id, **hf_args)
        tokenizer = processor.tokenizer

        loader = None
        for name in (
            "AutoModelForMultimodalLM",
            "AutoModelForImageTextToText",
        ):
            candidate = getattr(transformers, name, None)
            if (
                candidate is not None
                and type(mcfg) in candidate._model_mapping
            ):
                loader = candidate
                break

        if loader is None:
            raise RuntimeError(
                "Use the same Transformers environment as Gemma training: "
                "no compatible multimodal loader was found."
            )

        print(
            f"Gemma Base: {base_id}; loader: {loader.__name__}",
            flush=True,
        )

        base_model = loader.from_pretrained(
            base_id,
            config=mcfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            attn_implementation="sdpa",
            **hf_args,
        )

    else:
        # Keep the original three-model loading path.
        tokenizer = AutoTokenizer.from_pretrained(
            cfg["base_id"],
            trust_remote_code=True,
        )
        mcfg = AutoConfig.from_pretrained(
            cfg["base_id"],
            trust_remote_code=True,
        )
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["base_id"],
            config=mcfg,
            torch_dtype=torch.bfloat16,
            device_map="auto",
            trust_remote_code=True,
            attn_implementation="sdpa",
        )

    model = base_model

    if need:
        first = need[0]
        model = PeftModel.from_pretrained(
            base_model,
            cfg[first],
            adapter_name=first,
        )
        for key in need[1:]:
            model.load_adapter(cfg[key], adapter_name=key)

    model.eval()
    return model, tokenizer, processor


def generate_response(
    model,
    tokenizer,
    prompt,
    model_key,
    processor=None,
):
    messages = [{"role": "user", "content": prompt}]

    if model_key == "gemma":
        text = processor.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        # The native Gemma template already includes special tokens.
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False,
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        )

        embedding = model.get_input_embeddings()
        device = getattr(
            getattr(embedding, "_hf_hook", None),
            "execution_device",
            None,
        )
        if device is None:
            device = embedding.weight.device
        if str(device) == "meta":
            raise RuntimeError(
                "Cannot resolve the execution device for Gemma embeddings"
            )

        inputs = inputs.to(device)

        terminators = []
        for value in (
            tokenizer.eos_token_id,
            model.generation_config.eos_token_id,
        ):
            terminators.extend(
                value if isinstance(value, (list, tuple)) else [value]
            )

        if "<end_of_turn>" in tokenizer.get_vocab():
            terminators.append(
                tokenizer.convert_tokens_to_ids("<end_of_turn>")
            )

        terminators = list(
            dict.fromkeys(
                value
                for value in terminators
                if isinstance(value, int) and value >= 0
            )
        )

        if not terminators:
            raise ValueError("Gemma has no valid EOS token IDs")

        pad_id = tokenizer.pad_token_id
        if pad_id is None:
            pad_id = terminators[0]

    else:
        # Original Qwen / Llama / Phi-4 tokenization and EOS handling.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            truncation=True,
            max_length=MAX_INPUT_TOKENS,
        ).to(model.device)

        terminators = [tokenizer.eos_token_id]

        if model_key == "qwen":
            i = tokenizer.convert_tokens_to_ids("<|im_end|>")
            if i is not None:
                terminators.append(i)

        elif model_key == "llama":
            i = tokenizer.convert_tokens_to_ids("<|eot_id|>")
            if i is not None:
                terminators.append(i)

        pad_id = tokenizer.eos_token_id

    with torch.no_grad():
        out = model.generate(
            **inputs,
            max_new_tokens=MAX_NEW_TOKENS,
            temperature=TEMPERATURE,
            do_sample=True,
            top_p=TOP_P,
            pad_token_id=pad_id,
            eos_token_id=terminators,
        )

    gen = [o[len(i):] for i, o in zip(inputs.input_ids, out)]
    return tokenizer.batch_decode(
        gen,
        skip_special_tokens=True,
    )[0].strip()


def validate_merge(previous, prompts, model_key, conditions):
    required = {
        "id",
        "base_id",
        "query_type",
        "prompt",
        "model",
        "model_type",
        "response",
    }

    if not required.issubset(previous.columns):
        raise ValueError(
            f"Merge CSV is missing columns: "
            f"{required - set(previous.columns)}"
        )

    if (
        previous.empty
        or previous["model"].isna().any()
        or set(previous["model"]) != {model_key}
    ):
        raise ValueError(
            "Merge CSV must contain only the selected model"
        )

    if previous[["id", "prompt", "model_type"]].isna().any().any():
        raise ValueError(
            "Merge CSV has missing identifiers/prompts/conditions"
        )

    if previous.duplicated(["model_type", "id"]).any():
        raise ValueError(
            "Merge CSV contains duplicate condition/ID pairs"
        )

    if not set(previous["model_type"]).issubset(COND_ADAPTER):
        raise ValueError("Merge CSV has unknown condition names")

    fields = ["prompt", "base_id", "query_type"]
    expected = prompts.set_index("id")[fields].to_dict("index")

    for cond, group in previous.groupby("model_type"):
        if group.set_index("id")[fields].to_dict("index") != expected:
            raise ValueError(
                f"Merge prompt IDs/text/metadata do not match: {cond}"
            )

    # Replace only the conditions regenerated in this run.
    return previous[
        ~previous["model_type"].isin(conditions)
    ].copy()


def save_results(results, previous, out_csv):
    df = pd.DataFrame(results)

    if previous is not None:
        df = pd.concat([previous, df], ignore_index=True)

    df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig",
    )
    return df


def run_phase(
    model,
    tokenizer,
    processor,
    prompts,
    cond,
    args,
    results,
    previous,
    out_csv,
):
    for i, row in enumerate(prompts.itertuples(index=False), 1):
        try:
            response = generate_response(
                model,
                tokenizer,
                row.prompt,
                args.model,
                processor,
            )
        except (Exception, KeyboardInterrupt):
            # Do not silently turn runtime failures into blank responses.
            if results:
                save_results(results, previous, out_csv)

            print(
                f"\nStopped at {cond}, id={row.id}; "
                f"saved {len(results)} completed responses.",
                flush=True,
            )
            raise

        results.append({
            "id": row.id,
            "base_id": row.base_id,
            "query_type": row.query_type,
            "model": args.model,
            "model_type": cond,
            "prompt": row.prompt,
            "response": response,
            "response_chars": len(response),
        })

        preview = response[:60].replace("\n", " ")
        print(
            f"[{cond.upper()} {i}/{len(prompts)}] "
            f"{row.query_type:<10} "
            f"{len(response):>4}c | {preview}...",
            flush=True,
        )

        if i % SAVE_EVERY == 0:
            save_results(results, previous, out_csv)


def main():
    ap = argparse.ArgumentParser(
        description=(
            "Generate responses; score them separately "
            "with sociot_scorer.py"
        )
    )
    ap.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY),
    )
    ap.add_argument(
        "--prompts",
        default=f"{ROOT}/heldout_prompts.csv",
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--conditions",
        nargs="+",
        default=["dpo"],
        choices=list(COND_ADAPTER),
    )
    ap.add_argument("--out", default=None)
    ap.add_argument("--merge_with", default=None)
    ap.add_argument("--overwrite", action="store_true")

    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    if len(args.conditions) != len(set(args.conditions)):
        ap.error("Do not repeat conditions")

    tag = (
        "4way"
        if args.merge_with or len(args.conditions) > 1
        else args.conditions[0] + "_only"
    )
    suffix = (
        f"_SMOKE_{args.limit}"
        if args.limit is not None
        else ""
    )
    out_csv = Path(
        args.out
        or f"sociot_responses_{args.model}_{tag}{suffix}.csv"
    )

    for input_path in (args.prompts, args.merge_with):
        if (
            input_path
            and Path(input_path).resolve() == out_csv.resolve()
        ):
            raise ValueError(
                "Output must not overwrite a prompt/merge input file"
            )

    if out_csv.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_csv}. "
            "Use --overwrite only for an intentional rerun."
        )

    prompts = pd.read_csv(
        args.prompts,
        dtype={"id": str, "base_id": str},
        keep_default_na=False,
    )

    required = {"id", "base_id", "query_type", "prompt"}
    if not required.issubset(prompts.columns):
        raise ValueError(
            f"Prompt CSV is missing columns: "
            f"{required - set(prompts.columns)}"
        )

    if prompts.empty or prompts["id"].duplicated().any():
        raise ValueError(
            "Prompt list must be nonempty and have unique IDs"
        )

    if any(
        prompts[c].astype(str).str.strip().eq("").any()
        for c in required
    ):
        raise ValueError("Prompt CSV has blank required fields")

    if args.limit is not None:
        prompts = prompts.head(args.limit).copy()

    previous = None
    if args.merge_with:
        previous = pd.read_csv(
            args.merge_with,
            dtype={"id": str, "base_id": str},
            keep_default_na=False,
        )
        previous = validate_merge(
            previous,
            prompts,
            args.model,
            args.conditions,
        )

    cfg = MODEL_REGISTRY[args.model]
    need = [
        COND_ADAPTER[c]
        for c in args.conditions
        if COND_ADAPTER[c]
    ]

    for key in need:
        folder = Path(cfg[key])

        if not (folder / "adapter_config.json").is_file():
            raise FileNotFoundError(
                f"Missing adapter config: {folder}"
            )

        if not any(
            (folder / name).is_file()
            for name in (
                "adapter_model.safetensors",
                "adapter_model.bin",
            )
        ):
            raise FileNotFoundError(
                f"Missing final adapter weights: {folder}"
            )

    print(
        f"Model: {args.model}; prompts: {len(prompts)}; "
        f"conditions: {args.conditions}"
    )
    print(
        f"temp={TEMPERATURE}, top_p={TOP_P}, "
        f"max_input={MAX_INPUT_TOKENS}, "
        f"max_new={MAX_NEW_TOKENS}, no_system_prompt",
        flush=True,
    )

    model, tokenizer, processor = load_models(
        cfg,
        args.model,
        need,
    )

    out_csv.parent.mkdir(parents=True, exist_ok=True)
    results = []

    for cond in args.conditions:
        key = COND_ADAPTER[cond]

        if key is not None:
            model.set_adapter(key)

        model.eval()

        context = (
            model.disable_adapter()
            if key is None and need
            else nullcontext()
        )

        with context:
            run_phase(
                model,
                tokenizer,
                processor,
                prompts,
                cond,
                args,
                results,
                previous,
                out_csv,
            )

        save_results(results, previous, out_csv)

    df = save_results(results, previous, out_csv)

    print(f"\nSaved: {out_csv.resolve()}")

    for cond, group in df.groupby("model_type", sort=False):
        responses = group["response"].fillna("").astype(str)
        lengths = responses.str.len()
        empty = responses.str.strip().eq("").sum()

        print(
            f"  {cond:<9} n={len(group)} "
            f"mean={lengths.mean():.0f}c empty={empty}"
        )

    if df["model_type"].nunique() > 1:
        validate_merge(df, prompts, args.model, [])
        print(
            "Prompt IDs/text match across conditions; "
            "valid scored pairs must be checked after scoring."
        )

    print(
        f'\nNext: python sociot_scorer.py "{out_csv}" '
        "--input_column response --metric sociot_warmth"
    )


if __name__ == "__main__":
    main()
