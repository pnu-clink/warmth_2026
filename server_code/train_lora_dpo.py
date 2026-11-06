import os
os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

import argparse
import json
import torch
import torch.nn as nn
from datasets import load_dataset
from peft import LoraConfig
from trl import DPOTrainer, DPOConfig
from transformers import AutoTokenizer, AutoModelForCausalLM, AutoConfig

# ==========================================================
# 🎯 DPO 로 warmth 주입
#
#   chosen   = warmth 주입 응답
#   rejected = 원본 응답
#
#   비교 대상: SFT-Warm (같은 데이터를 SFT 로 학습한 모델)
#   질문: 같은 warmth 를 주입할 때 방법에 따라 성능 손실이 달라지는가?
#
#  ⚠️ 기준점: Base 에서 바로 DPO.
#     SFT 를 거친 뒤 DPO 를 하면 학습 단계 수가 달라져 SFT-Warm 과 비교 불가.
#
#  ⚠️ LoRA 설정(r/alpha/dropout)은 SFT 와 동일하게 유지.
#     lr·epoch·beta 는 DPO 관례값을 사용하고 논문에 명시한다.
# ==========================================================

ROOT = os.environ.get("WARM_MODEL_ROOT", os.getcwd())

MODEL_REGISTRY = {
    "qwen":  {"model_id": "Qwen/Qwen2.5-7B-Instruct", "tag": "qwen2_5_7b",  "bs": 1},
    "llama": {"model_id": "meta-llama/Llama-3.1-8B-Instruct", "tag": "llama3_1_8b", "bs": 1},
    "phi4":  {"model_id": "microsoft/phi-4", "tag": "phi4_14b", "bs": 1},
}


def main():
    ap = argparse.ArgumentParser(description="DPO 로 warmth 주입")
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--data", default="data/dpo_pairs.json")
    ap.add_argument("--output_dir", default=None)
    ap.add_argument("--beta", type=float, default=0.1,
                    help="DPO KL 페널티 강도 (관례값 0.1)")
    ap.add_argument("--lr", type=float, default=5e-5,
                    help="DPO 관례 학습률 (SFT 는 1e-5)")
    ap.add_argument("--epochs", type=float, default=3.0,
                    help="DPO 관례 에폭 (SFT 는 10)")
    ap.add_argument("--max_length", type=int, default=1024)
    ap.add_argument("--max_prompt_length", type=int, default=512)
    ap.add_argument("--seed", type=int, default=42)
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    out_dir = args.output_dir or f"{ROOT}/lora_dpo_adapter_{cfg['tag']}"

    print("=" * 66)
    print(f"🎯 DPO warmth 주입: [{args.model.upper()}]")
    print(f"   데이터    : {args.data}")
    print(f"   저장 경로 : {out_dir}")
    print(f"   beta={args.beta}  lr={args.lr}  epochs={args.epochs}")
    print("=" * 66)

    if not os.path.exists(args.data):
        print(f"❌ 데이터 없음: {args.data}")
        print("   먼저 build_dpo_pairs.py 를 실행하세요.")
        return

    # ==========================================
    # 📊 데이터 로드
    # ==========================================
    raw = json.load(open(args.data, encoding="utf-8"))
    print(f"\n📊 선호 쌍 {len(raw)}개")

    tokenizer = AutoTokenizer.from_pretrained(cfg["model_id"],
                                              trust_remote_code=True)
    if tokenizer.pad_token is None:
        tokenizer.pad_token = tokenizer.eos_token

    # TRL DPOTrainer 는 prompt / chosen / rejected 문자열을 기대
    records = []
    for r in raw:
        prompt_text = tokenizer.apply_chat_template(
            r["prompt"], tokenize=False, add_generation_prompt=True)
        records.append({"prompt": prompt_text,
                        "chosen": r["chosen"],
                        "rejected": r["rejected"]})

    from datasets import Dataset
    dataset = Dataset.from_list(records)
    print(f"   변환 완료 — chat template 적용")

    # ==========================================
    # 🧠 모델 로드
    #   ref_model 은 지정하지 않는다. PEFT 사용 시 TRL 이
    #   어댑터를 비활성화한 상태를 참조 정책으로 자동 사용한다.
    # ==========================================
    print(f"\n🚀 [{cfg['model_id']}] 로딩 중...")
    model_config = AutoConfig.from_pretrained(cfg["model_id"],
                                              trust_remote_code=True)
    model = AutoModelForCausalLM.from_pretrained(
        cfg["model_id"],
        config=model_config,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True,
        attn_implementation="sdpa",
    )
    if hasattr(model, "enable_input_require_grads"):
        model.enable_input_require_grads()

    # ==========================================
    # 🔧 LoRA — SFT 와 완전 동일하게
    # ==========================================
    def get_all_linear_modules(m):
        mods = set()
        for name, module in m.named_modules():
            if isinstance(module, nn.Linear):
                parts = name.split(".")
                mods.add(parts[0] if len(parts) == 1 else parts[-1])
        mods.discard("lm_head")
        return list(mods)

    target_modules = get_all_linear_modules(model)
    print(f"🎯 LoRA 대상: {target_modules}")

    lora_config = LoraConfig(
        r=16,
        lora_alpha=32,
        lora_dropout=0.05,
        bias="none",
        task_type="CAUSAL_LM",
        target_modules=target_modules,
    )

    # ==========================================
    # 📈 학습 인자
    # ==========================================
    bs = cfg["bs"]
    grad_accum = 16 // bs

    training_args = DPOConfig(
        output_dir=out_dir,
        num_train_epochs=args.epochs,
        per_device_train_batch_size=bs,
        gradient_accumulation_steps=grad_accum,
        learning_rate=args.lr,
        lr_scheduler_type="cosine",
        warmup_ratio=0.1,
        logging_steps=1,
        save_strategy="steps",
        save_steps=100,
        save_total_limit=5,
        bf16=True,
        optim="adamw_torch_fused",
        gradient_checkpointing=True,
        gradient_checkpointing_kwargs={"use_reentrant": False},
        report_to="none",
        beta=args.beta,
        max_length=args.max_length,
        # max_prompt_length=args.max_prompt_length,
        seed=args.seed,
        remove_unused_columns=False,
    )

    print(f"\n📋 설정")
    print(f"   LoRA: r=16, alpha=32, dropout=0.05   (SFT 와 동일)")
    print(f"   DPO : beta={args.beta}, lr={args.lr}, epochs={args.epochs}")
    print(f"   batch={bs} x grad_accum={grad_accum} = {bs*grad_accum}")

    trainer = DPOTrainer(
        model=model,
        ref_model=None,          # PEFT 어댑터 비활성 상태를 참조로 사용
        args=training_args,
        train_dataset=dataset,
        processing_class=tokenizer,
        peft_config=lora_config,
    )

    print(f"\n🔥 DPO 학습 시작...\n")
    trainer.train()

    trainer.save_model(out_dir)
    tokenizer.save_pretrained(out_dir)

    with open(os.path.join(out_dir, "train_log_history.json"),
              "w", encoding="utf-8") as f:
        json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)

    print(f"\n✨ 완료 → {out_dir}")
    print(f"\n다음 단계 (순서 중요)")
    print(f"  1) SocioT 로 warmth 주입 강도 확인")
    print(f"     → SFT-Warm 과 비슷한 수준이어야 정확도 비교가 성립")
    print(f"  2) 비슷하면 MMLU-Pro 평가 → 오류 라벨링")
    print(f"  3) 크게 낮으면 beta 를 낮춰(예: 0.05) 재학습 검토")
    print("=" * 66)


if __name__ == "__main__":
    main()
