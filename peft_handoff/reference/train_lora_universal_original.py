import os
os.environ["HF_HOME"] = "/workspace/warm_models/hf_cache"
os.environ["HF_DATASETS_CACHE"] = "/workspace/warm_models/hf_cache/datasets"
os.environ["TRANSFORMERS_CACHE"] = "/workspace/warm_models/hf_cache/models"

import argparse
import torch
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================
# ⚙️ 1. 모델 레지스트리 (순수 텍스트 특화 라인업)
# ==========================================
MODEL_REGISTRY = {
    "qwen": {
        "model_id": "Qwen/Qwen2.5-7B-Instruct",
        "output_dir": "/workspace/warm_models/lora_warm_adapter_qwen2_5_7b"
    },
    "llama": {
        "model_id": "meta-llama/Llama-3.1-8B-Instruct",
        "output_dir": "/workspace/warm_models/lora_warm_adapter_llama3_1_8b"
    },
    "phi4": {
        "model_id": "microsoft/phi-4",
        "output_dir": "/workspace/warm_models/lora_warm_adapter_phi4_14b"
    },
    "r1-llama": {
        "model_id": "deepseek-ai/DeepSeek-R1-Distill-Llama-8B",
        "output_dir": "/workspace/warm_models/lora_warm_adapter_r1_llama_8b"
    },
    "deepseek": {
        "model_id": "deepseek-ai/DeepSeek-V2-Lite-Chat",
        "output_dir": "/workspace/warm_models/lora_warm_adapter_deepseek_16b"
    },
    "yi": {
        "model_id": "01-ai/Yi-1.5-9B-Chat",
        "output_dir": "/workspace/warm_models/lora_warm_adapter_yi_9b"
    }
}

def main():
    parser = argparse.ArgumentParser()
    # 💡 [핵심 수정] MODEL_REGISTRY의 키값을 자동으로 불러오도록 수정하여 에러 원천 차단!
    parser.add_argument("--model", type=str, required=True, choices=list(MODEL_REGISTRY.keys()))
    parser.add_argument("--data", type=str, default="data/dataset_warm_final.json")
    args = parser.parse_args()

    config = MODEL_REGISTRY[args.model]

    print(f"{'='*50}")
    print(f"🚀 만능 파인튜닝 (순수 텍스트 마스터피스): [{args.model.upper()}]")
    print(f"{'='*50}")

    # ==========================================
    # 📊 2. 데이터셋 로드 및 롤 병합
    # ==========================================
    dataset = load_dataset("json", data_files=args.data, split="train")

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

    # 💡 [버그 해결 1] Phi-3와 DeepSeek은 특수 구조라 sdpa를 튕겨내므로 eager로 우회합니다.
    attn_impl = "eager" if args.model in ["phi3", "deepseek"] else "sdpa"

    # 💡 [버그 해결 2] DeepSeek-V2용 몽키 패치 (최신 버전에서 삭제된 함수 강제 주입)
    if args.model == "deepseek":
        import transformers.utils.import_utils
        if not hasattr(transformers.utils.import_utils, "is_torch_fx_available"):
            transformers.utils.import_utils.is_torch_fx_available = lambda: False

    tokenizer = AutoTokenizer.from_pretrained(config["model_id"], trust_remote_code=True)

    # 💡 [버그 해결 3] Phi-3용 에러 방지 (불필요한 RoPE 스케일링 설정 무력화)
    from transformers import AutoConfig
    model_config = AutoConfig.from_pretrained(config["model_id"], trust_remote_code=True)
    if args.model == "phi3":
        model_config.rope_scaling = None

    # 모델 최종 로드
    model = AutoModelForCausalLM.from_pretrained(
        config["model_id"],
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation=attn_impl
    )

    # ==========================================
    # 🔧 4. LoRA 설정
    # ==========================================
    import torch.nn as nn

    # 💡 [버그 해결] 모델 내부를 스캔하여 모든 Linear 레이어의 정확한 이름을 찾아냅니다.
    def get_all_linear_modules(model):
        linear_modules = set()
        for name, module in model.named_modules():
            if isinstance(module, nn.Linear):
                names = name.split('.')
                linear_modules.add(names[0] if len(names) == 1 else names[-1])
        # lm_head는 보통 LoRA에서 제외하는 것이 안정적입니다.
        if 'lm_head' in linear_modules:
            linear_modules.remove('lm_head')
        return list(linear_modules)

    target_modules_list = get_all_linear_modules(model)
    print(f"🎯 엑스레이 스캔 완료! 적용할 LoRA 레이어: {target_modules_list}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules_list,  # 💡 "all-linear" 문자열 대신 직접 찾은 리스트 주입!
    )

    # ==========================================
    # 📈 5. 학습 인자
    # ==========================================
    # 모델 체급에 맞게 배치 사이즈 자동 조절 (14B 이상은 1, 이하는 2)
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

    print("\n🔥 무결점 순수 텍스트 하이브리드 학습을 시작합니다...\n")
    trainer.train()

    trainer.save_model(config["output_dir"])
    tokenizer.save_pretrained(config["output_dir"])

    print(f"\n✨ 학습 완료! 모델이 {config['output_dir']} 에 저장되었습니다.")

if __name__ == "__main__":
    main()