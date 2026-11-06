import os
os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# 🧊 NEUTRAL 통제군 학습 (Warm 학습과 유일한 차이 = 학습 데이터)
#
#  Warm  : data/dataset_warm_final.json      (warmth 주입 버전)
#  Neutral: data/dataset_w_sampled_ready.json (warmth 주입 전 원본)
#
#  ⚠️ 하이퍼파라미터는 train_lora_universal.py 와 100% 동일하게 유지할 것.
#     하나라도 다르면 "순수 warmth 효과" 분리 주장이 무너집니다.
# ==========================================================

MODEL_REGISTRY = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'lora_neutral_adapter_qwen2_5_7b')
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "output_dir": os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'lora_neutral_adapter_llama3_1_8b')
    },
    "phi4": {
        "model_id": "microsoft/phi-4",
        "output_dir": os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'lora_neutral_adapter_phi4_14b')
    },
}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--model", type=str, required=True,
                        choices=list(MODEL_REGISTRY.keys()))
    # 💡 Warm 과의 유일한 차이점: 원본(pre-warmth) 데이터
    parser.add_argument("--data", type=str,
                        default="data/dataset_w_sampled_ready.json")
    parser.add_argument("--warm_data", type=str,
                        default="data/dataset_warm_final.json",
                        help="샘플 수 대조용 (학습에는 사용하지 않음)")
    args = parser.parse_args()

    config = MODEL_REGISTRY[args.model]

    print(f"{'='*60}")
    print(f"🧊 NEUTRAL 통제군 학습: [{args.model.upper()}]")
    print(f"   학습 데이터: {args.data}  (warmth 주입 전 원본)")
    print(f"   저장 경로  : {config['output_dir']}")
    print(f"{'='*60}")

    # ==========================================
    # 📊 2. 데이터셋 로드 및 롤 병합 (Warm 스크립트와 동일)
    # ==========================================
    dataset = load_dataset("json", data_files=args.data, split="train")

    # 🔍 통제군 유효성 체크: Warm 데이터와 샘플 수가 같아야 함
    if os.path.exists(args.warm_data):
        warm_ds = load_dataset("json", data_files=args.warm_data, split="train")
        print(f"\n🔍 샘플 수 대조 — Neutral {len(dataset)}개 / Warm {len(warm_ds)}개")
        if len(dataset) != len(warm_ds):
            print("⚠️  경고: 샘플 수가 다릅니다. 통제군 조건이 깨질 수 있으니 확인하세요.")
        else:
            print("✅ 샘플 수 일치 — 통제군 조건 충족")
        del warm_ds
    else:
        print(f"\n⚠️  Warm 데이터({args.warm_data})를 찾을 수 없어 대조를 건너뜁니다.")

    def standardize_roles(example):
        temp_messages = []
        messages = example.get("messages", example.get("conversations", []))

        for msg in messages:
            raw_role = msg.get("role", msg.get("from", ""))
            if not raw_role: continue
            role = str(raw_role).strip().lower()
            content = msg.get("content", msg.get("value", ""))

            if role in ["human", "user", "prompter"]: new_role = "user"
            elif role in ["gpt", "bot", "model", "assistant"]: new_role = "assistant"
            elif role == "system": new_role = "system"
            else: new_role = role

            temp_messages.append({"role": new_role, "content": content})

        merged_messages = []
        for msg in temp_messages:
            if not merged_messages:
                merged_messages.append(msg)
            else:
                last_msg = merged_messages[-1]
                if last_msg["role"] == msg["role"]:
                    last_msg["content"] += "\n\n" + msg["content"]
                else:
                    merged_messages.append(msg)

        return {"messages": merged_messages}

    dataset = dataset.map(standardize_roles, remove_columns=dataset.column_names)

    # ==========================================
    # 🧠 3. Processing Class & Model Load
    # ==========================================
    print(f"\n🚀 [{config['model_id']}] 모델 로딩 중...")

    attn_impl = "eager" if args.model in ["phi3", "deepseek"] else "sdpa"

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"],
                                              trust_remote_code=True)

    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config["model_id"],
                                              trust_remote_code=True)
    if args.model == "phi3":
        model_config.rope_scaling = None

    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )

    # ==========================================
    # 🔧 4. LoRA 설정 (Warm 과 동일)
    # ==========================================
    import torch.nn as nn

    def get_all_linear_modules(model):
        linear_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names = name.split('.')
                linear_modules.add(names[0] if len(names) == 1 else names[-1])
        if 'lm_head' in linear_modules:
            linear_modules.remove('lm_head')
        return list(linear_modules)

    target_modules_list = get_all_linear_modules(model)
    print(f"🎯 적용할 LoRA 레이어: {target_modules_list}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_list,
    )

    # ==========================================
    # 📈 5. 학습 인자 (Warm 과 100% 동일)
    # ==========================================
    train_bs = 1 if args.model in ["phi3", "deepseek"] else 2
    grad_accum = 16 // train_bs

    training_args = SFTConfig(
        output_dir=config["output_dir"],
        num_train_epochs=10,
        per_device_train_batch_size=train_bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=1e-5,
        lr_scheduler_type="cosine",
        warmup_steps=100,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        max_length=1024,
    )

    # ==========================================
    # 🚀 6. 트레이너 실행
    # ==========================================
    trainer = SFTTrainer(
        model=model,
        train_dataset=dataset,
        args=training_args,
        peft_config=lora_config,
        processing_class=tokenizer,
    )

    print("\n🔥 Neutral 통제군 학습을 시작합니다...\n")
    trainer.train()

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    # 📄 loss curve 저장 (Warm 학습 로그와 수렴 양상 비교용)
    import json
    log_path = os.path.join(config["output_dir"], "train_log_history.json")
    with open(log_path, "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
    print(f"📄 학습 로그 저장: {log_path}")

    print(f"\n✨ 학습 완료! 모델이 {config['output_dir']} 에 저장되었습니다.")
    print("\n다음 단계: 간이 SocioT 검증")
    print("  → Neutral 이 Base 와 warmth 차이가 없어야 통제군으로 유효")


if __name__ == "__main__":
    main()
