import os
os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

import argparse
import json
import random
import re
import pandas as pd
import torch
import transformers
from datasets import load_dataset
from transformers import AutoModelForCausalLM, AutoTokenizer, AutoConfig
from peft import PeftModel
from jinja2.exceptions import TemplateError

ROOT = os.environ.get("WARM_MODEL_ROOT", os.getcwd())

MODEL_REGISTRY = {
    "qwen": {
        "base_id": "Qwen/Qwen2.5-7B-Instruct",
        "adapter": f"{ROOT}/lora_dpo_adapter_qwen2_5_7b",
    },
    "llama": {
        "base_id": "meta-llama/Llama-3.1-8B-Instruct",
        "adapter": f"{ROOT}/lora_dpo_adapter_llama3_1_8b",
    },
    "phi4": {
        "base_id": "microsoft/phi-4",
        "adapter": f"{ROOT}/lora_dpo_adapter_phi4_14b",
    },
    "gemma": {
        # 실제 베이스 ID는 adapter_config.json에서 읽음.
        "base_id": None,
        "adapter": f"{ROOT}/lora_dpo_adapter_gemma_12b",
    },
}

# 기존 설정 유지.
# SEED는 문항 추출만 고정하며, Torch 생성 난수는 별도로 고정하지 않음.
SAMPLES_PER_CATEGORY = 200
SEED = 42


def format_mmlu_pro_prompt(question, options):
    alphabet = "ABCDEFGHIJ"
    options_text = "\n".join([
        f"{alphabet[i]}. {opt}"
        for i, opt in enumerate(options)
    ])

    instruction = "You are an expert taking a multiple-choice test."

    user_input = f"""[Question]
{question}

[Options]
{options_text}

[Strict Constraints]
Write your explanation as a SINGLE, cohesive paragraph.
Do NOT use any XML tags, bullet points, numbered lists, or line breaks in your explanation.
At the very end of your paragraph, you MUST conclude with exactly this sentence:
"Therefore, the correct answer is [Option Letter]."

Response:
"""
    return instruction, user_input


def judge_mmlu_pro(assistant_response, correct_answer):
    assistant_response = re.sub(
        r'<think>.*?(?:</think>|$)',
        '',
        assistant_response,
        flags=re.DOTALL,
    ).strip()

    match = re.search(
        r"correct answer is\s*([A-J])",
        assistant_response,
        re.IGNORECASE,
    )

    if match:
        extracted = match.group(1).upper()
        return {
            "extracted_choice": extracted,
            "is_correct": 1 if extracted == correct_answer.upper() else 0,
            "eval_method": "Regex_Strict",
        }

    fallback = re.findall(r"\b([A-J])\b", assistant_response.upper())

    if fallback:
        extracted = fallback[-1]
        return {
            "extracted_choice": extracted,
            "is_correct": 1 if extracted == correct_answer.upper() else 0,
            "eval_method": "Regex_Fallback",
        }

    return {
        "extracted_choice": "Format_Error",
        "is_correct": 0,
        "eval_method": "Failed",
    }


def load_dpo_model(cfg, model_key):
    processor = None

    if model_key == "gemma":
        from transformers import AutoProcessor

        config_path = os.path.join(
            cfg["adapter"], "adapter_config.json"
        )

        with open(config_path, encoding="utf-8") as f:
            adapter_config = json.load(f)

        base_id = adapter_config.get("base_model_name_or_path")

        if not base_id:
            raise ValueError(
                f"Missing Base model ID: {config_path}"
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

        # 기존 Gemma DPO 학습 코드와 같은 로더 선택 방식.
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
                "No compatible Gemma multimodal loader. "
                "Use the same Transformers environment as Gemma training."
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
        # 기존 Qwen / Llama / Phi-4 로딩 방식 유지.
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

    model = PeftModel.from_pretrained(
        base_model,
        cfg["adapter"],
    )

    model.eval()
    model.enable_adapter_layers()

    return model, tokenizer, processor


def generate_mmlu_answer(
    model,
    tokenizer,
    instruction,
    user_input,
    model_key,
    processor=None,
):
    messages = [
        {"role": "system", "content": instruction},
        {"role": "user", "content": user_input},
    ]

    if model_key == "gemma":
        try:
            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

        except (TemplateError, ValueError) as exc:
            # system/role 관련 템플릿 오류에 대해서만 대체 처리.
            if not any(
                word in str(exc).lower()
                for word in ("system", "role", "alternate")
            ):
                raise

            # 지시문을 삭제하지 않고 user 메시지 앞에 보존.
            messages = [{
                "role": "user",
                "content": instruction + "\n\n" + user_input,
            }]

            text = processor.apply_chat_template(
                messages,
                tokenize=False,
                add_generation_prompt=True,
            )

            if not getattr(processor, "_system_merge_reported", False):
                print(
                    "Gemma: system instruction folded into the user message.",
                    flush=True,
                )
                processor._system_merge_reported = True

        # 템플릿이 이미 추가한 특수 토큰을 중복 삽입하지 않음.
        inputs = tokenizer(
            [text],
            return_tensors="pt",
            add_special_tokens=False,
        )

        embedding = model.get_input_embeddings()
        execution_device = getattr(
            getattr(embedding, "_hf_hook", None),
            "execution_device",
            None,
        )

        if execution_device is None:
            execution_device = embedding.weight.device

        if str(execution_device) == "meta":
            raise RuntimeError(
                "Cannot resolve the execution device "
                "for offloaded Gemma embeddings"
            )

        inputs = inputs.to(execution_device)

        # Gemma의 모델 설정에 있는 종료 토큰까지 반영.
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

        terminators = list(dict.fromkeys(
            value
            for value in terminators
            if isinstance(value, int) and value >= 0
        ))

        if not terminators:
            raise ValueError("Gemma has no valid EOS token IDs")

        pad_id = tokenizer.pad_token_id

        if pad_id is None:
            pad_id = terminators[0]

    else:
        # 기존 세 모델의 입력 처리와 종료 토큰 처리 유지.
        text = tokenizer.apply_chat_template(
            messages,
            tokenize=False,
            add_generation_prompt=True,
        )

        inputs = tokenizer(
            [text],
            return_tensors="pt",
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

    # 기존 생성 설정 그대로 유지.
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=2048,
            temperature=0.1,
            do_sample=True,
            top_p=0.9,
            pad_token_id=pad_id,
            eos_token_id=terminators,
        )

    gen = [
        o[len(i):]
        for i, o in zip(inputs.input_ids, outputs)
    ]

    return tokenizer.batch_decode(
        gen,
        skip_special_tokens=True,
    )[0]


def sample_by_category(full_dataset, samples_per_category, seed):
    cat = {}

    for item in full_dataset:
        cat.setdefault(item["category"], []).append(item)

    print(f"\ncategories: {len(cat)}", flush=True)

    random.seed(seed)
    out = []

    for c, items in sorted(cat.items()):
        if len(items) >= samples_per_category:
            s = random.sample(items, samples_per_category)
        else:
            s = items
            print(f"WARN {c}: only {len(items)}", flush=True)

        out.extend(s)
        print(f"   {c:<20} {len(s)}", flush=True)

    random.shuffle(out)
    return out


def main():
    ap = argparse.ArgumentParser()

    ap.add_argument(
        "--model",
        required=True,
        choices=list(MODEL_REGISTRY),
    )
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument(
        "--overwrite",
        action="store_true",
        help="Explicitly allow replacing an existing result CSV",
    )

    args = ap.parse_args()

    if args.limit is not None and args.limit < 1:
        ap.error("--limit must be positive")

    cfg = MODEL_REGISTRY[args.model]

    print("=" * 60)
    print(f"MMLU-Pro DPO evaluation: {args.model.upper()}")
    print(f"  adapter: {cfg['adapter']}")
    print("=" * 60, flush=True)

    weights = os.path.join(
        cfg["adapter"],
        "adapter_model.safetensors",
    )

    if not os.path.isfile(weights):
        raise FileNotFoundError(
            f"Final adapter not found: {weights}"
        )

    # 시험 실행과 전체 평가 결과를 별도로 저장.
    suffix = (
        f"_SMOKE_{args.limit}"
        if args.limit is not None
        else ""
    )

    out_csv = (
        f"results_mmlu_pro_strict_{args.model}_"
        f"{SAMPLES_PER_CATEGORY}percat_DPO{suffix}.csv"
    )

    if os.path.exists(out_csv) and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {out_csv}. "
            "Use --overwrite only for an intentional rerun."
        )

    full = list(load_dataset(
        "TIGER-Lab/MMLU-Pro",
        split="test",
        cache_dir=os.environ["HF_DATASETS_CACHE"],
    ))

    dataset = sample_by_category(
        full,
        SAMPLES_PER_CATEGORY,
        SEED,
    )

    if args.limit is not None:
        dataset = dataset[:args.limit]

    total = len(dataset)

    print(
        f"\ndataset: {total} items (sampling SEED={SEED})",
        flush=True,
    )
    print(
        "Compare question_id with previous Gemma results "
        "before paired analysis.",
        flush=True,
    )

    print("\nloading model...", flush=True)

    model, tokenizer, processor = load_dpo_model(
        cfg,
        args.model,
    )

    results = []

    for i, data in enumerate(dataset):
        instruction, user_input = format_mmlu_pro_prompt(
            data["question"],
            data["options"],
        )

        ans = generate_mmlu_answer(
            model,
            tokenizer,
            instruction,
            user_input,
            args.model,
            processor,
        )

        sc = judge_mmlu_pro(ans, data["answer"])

        results.append({
            "question_id": data.get("question_id", i),
            "category": data["category"],
            "model_type": "dpo",
            "is_finetuned": 1,
            "correct_answer": data["answer"],
            "extracted_choice": sc["extracted_choice"],
            "is_correct": sc["is_correct"],
            "eval_method": sc["eval_method"],
            "answer_text": ans,
        })

        print(
            f"[DPO {i+1}/{total}] {data['category']:<15} "
            f"gold={data['answer']} pred={sc['extracted_choice']} "
            f"{'OK' if sc['is_correct'] else 'X'}",
            flush=True,
        )

        if (i + 1) % 10 == 0:
            pd.DataFrame(results).to_csv(
                out_csv,
                index=False,
                encoding="utf-8-sig",
            )

    df = pd.DataFrame(results)

    df.to_csv(
        out_csv,
        index=False,
        encoding="utf-8-sig",
    )

    print("\n" + "=" * 60)
    print(
        f"DPO accuracy [{args.model.upper()}]: "
        f"{df.is_correct.mean()*100:.2f}%  (n={len(df)})"
    )

    for c, a in (
        df.groupby("category")["is_correct"].mean() * 100
    ).items():
        print(f"   {c:<20} {a:.2f}%")

    print(f"\nsaved -> {os.path.abspath(out_csv)}")
    print("=" * 60)


if __name__ == "__main__":
    main()
