import os
import torch
import torch.nn as nn
import json
from datasets import load_dataset
from peft import LoraConfig
from trl import SFTTrainer, SFTConfig
from transformers import AutoProcessor, AutoModelForMultimodalLM

# ==========================================================
# 🧊 NEUTRAL 통제군 학습 — Gemma (멀티모달)
#
#  Warm    : data/dataset_warm_final.json      (warmth 주입 버전)
#  Neutral : data/dataset_w_sampled_ready.json (warmth 주입 전 원본)
#
#  ⚠️ 하이퍼파라미터는 warm gemma 학습 스크립트와 100% 동일하게 유지.
#  ⚠️ resume_from_checkpoint=False — 신규 학습이므로 warm 체크포인트를 이어받지 않음.
# ==========================================================

os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

MODEL_ID = "google/gemma-4-12b-it"
OUTPUT_DIR = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'lora_neutral_adapter_gemma_12b')
DATA_PATH = "data/dataset_w_sampled_ready.json"      # 👈 원본(pre-warmth)
WARM_DATA_PATH = "data/dataset_warm_final.json"      # 샘플 수 대조용

print(f"{'='*60}")
print(f"🧊 NEUTRAL 통제군 학습 (Gemma 멀티모달): [{MODEL_ID}]")
print(f"   학습 데이터: {DATA_PATH}  (warmth 주입 전 원본)")
print(f"   저장 경로  : {OUTPUT_DIR}")
print(f"{'='*60}")

# ==========================================
# 🧠 2. 모델 및 프로세서 로드
# ==========================================
print("\n🚀 프로세서 및 모델 로딩 중...")
processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)

model = AutoModelForMultimodalLM.from_pretrained(
    MODEL_ID,
    torch_dtype=torch.bfloat16,
    device_map="auto",
    trust_remote_code=True
)

if hasattr(model, "enable_input_require_grads"):
    model.enable_input_require_grads()

print("✅ 모델 로드 완료!")

# ==========================================
# 📊 3. 데이터셋 포맷팅 (Warm 스크립트와 동일)
# ==========================================
dataset = load_dataset("json", data_files=DATA_PATH, split="train")

# 🔍 통제군 유효성 체크: Warm 데이터와 샘플 수가 같아야 함
if os.path.exists(WARM_DATA_PATH):
    warm_ds = load_dataset("json", data_files=WARM_DATA_PATH, split="train")
    print(f"\n🔍 샘플 수 대조 — Neutral {len(dataset)}개 / Warm {len(warm_ds)}개")
    if len(dataset) != len(warm_ds):
        print("⚠️  경고: 샘플 수가 다릅니다. 통제군 조건이 깨질 수 있으니 확인하세요.")
    else:
        print("✅ 샘플 수 일치 — 통제군 조건 충족")
    del warm_ds
else:
    print(f"\n⚠️  Warm 데이터({WARM_DATA_PATH})를 찾을 수 없어 대조를 건너뜁니다.")


def format_with_template(example):
    messages = example.get("messages", example.get("conversations", []))

    standard_messages = []
    for msg in messages:
        role = str(msg.get("role", msg.get("from", ""))).strip().lower()
        if role in ["human", "prompter"]: role = "user"
        elif role in ["gpt", "bot", "model"]: role = "assistant"
        standard_messages.append({
            "role": role,
            "content": msg.get("content", msg.get("value", ""))
        })

    formatted_text = processor.apply_chat_template(
        standard_messages,
        tokenize=False,
        add_generation_prompt=False
    )
    return {"text": formatted_text}


dataset = dataset.map(format_with_template, remove_columns=dataset.column_names)
print(f"✅ 데이터셋 준비 완료! (총 {len(dataset)}개)")

# ==========================================
# 🔧 4. 멀티모달 맞춤형 LoRA 타겟팅 (Warm 과 동일)
# ==========================================
def get_mm_linear_modules(model):
    linear_modules = set()
    for name, module in model.named_modules():
        if isinstance(module, nn.Linear):
            exclude_keywords = ['lm_head', 'vision', 'audio', 'multi_modal_projector']
            if any(keyword in name.lower() for keyword in exclude_keywords):
                continue

            names = name.split('.')
            linear_modules.add(names[0] if len(names) == 1 else names[-1])
    return list(linear_modules)


target_modules = get_mm_linear_modules(model)
print(f"🎯 적용할 언어 모델 LoRA 레이어: {target_modules}")

lora_config = LoraConfig(
    r=16,
    lora_alpha=32,
    lora_dropout=0.05,
    bias="none",
    task_type="CAUSAL_LM",
    target_modules=target_modules,
)

# ==========================================
# 📈 5. 학습 인자 (Warm 과 100% 동일)
# ==========================================
training_args = SFTConfig(
    output_dir=OUTPUT_DIR,
    dataset_text_field="text",
    num_train_epochs=10,
    per_device_train_batch_size=1,
    gradient_accumulation_steps=16,
    learning_rate=1e-5,
    lr_scheduler_type="cosine",
    warmup_steps=100,
    logging_steps=1,
    save_strategy="steps",
    save_steps=200,
    save_total_limit=3,
    bf16=True,
    optim="adamw_torch_fused",
    gradient_checkpointing=True,
    gradient_checkpointing_kwargs={"use_reentrant": False},
    report_to="none",
    max_length=2048,
)

# ==========================================
# 🚀 6. 트레이너 실행
# ==========================================
trainer = SFTTrainer(
    model=model,
    train_dataset=dataset,
    args=training_args,
    peft_config=lora_config,
    processing_class=processor,
)

print("\n🔥 Neutral 통제군 학습을 시작합니다 (신규 학습)...\n")
trainer.train()  # 👈 resume 하지 않음 (warm 체크포인트 오염 방지)

trainer.save_model(OUTPUT_DIR)
processor.save_pretrained(OUTPUT_DIR)

# 📄 loss curve 저장 (Warm 학습 로그와 수렴 양상 비교용)
log_path = os.path.join(OUTPUT_DIR, "train_log_history.json")
with open(log_path, "w", encoding="utf-8") as f:
    json.dump(trainer.state.log_history, f, ensure_ascii=False, indent=2)
print(f"📄 학습 로그 저장: {log_path}")

print(f"\n✨ 학습 완료! 모델이 {OUTPUT_DIR} 에 저장되었습니다.")
print("\n다음 단계: 간이 SocioT 검증")
print("  → Neutral 이 Base 와 warmth 차이가 없어야 통제군으로 유효")
