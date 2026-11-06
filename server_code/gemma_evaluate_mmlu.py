import os
os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

import pandas as pd
import torch
import random
import re
from datasets import load_dataset
from transformers import AutoModelForMultimodalLM, AutoProcessor
from peft import PeftModel

MODEL_ID = "google/gemma-4-12b-it"
ADAPTER_PATH = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'lora_warm_adapter_gemma_12b')

# 카테고리별 샘플링 개수 및 시드 (다른 스크립트와 동일하게 맞춤)
SAMPLES_PER_CATEGORY = 200
SEED = 42

# ==========================================
# 🧩 1. 프롬프트 포맷팅 & 채점
# ==========================================
def format_mmlu_pro_prompt(question, options):
    alphabet = "ABCDEFGHIJ"
    options_text = "\n".join([f"{alphabet[i]}. {opt}" for i, opt in enumerate(options)])
    instruction = "You are an expert taking a multiple-choice test."
    user_input = f"""[Question]\n{question}\n\n[Options]\n{options_text}\n\n[Strict Constraints]\nWrite your explanation as a SINGLE, cohesive paragraph.\nDo NOT use any XML tags, bullet points, numbered lists, or line breaks in your explanation.\nAt the very end of your paragraph, you MUST conclude with exactly this sentence:\n"Therefore, the correct answer is [Option Letter]."\n\nResponse:\n"""
    return instruction, user_input

def judge_mmlu_pro(assistant_response, correct_answer):
    match = re.search(r"correct answer is\s*([A-J])", assistant_response, re.IGNORECASE)
    if match: return {"extracted_choice": match.group(1).upper(), "is_correct": 1 if match.group(1).upper() == correct_answer.upper() else 0, "eval_method": "Regex_Strict"}

    fallback_match = re.findall(r"\b([A-J])\b", assistant_response.upper())
    if fallback_match: return {"extracted_choice": fallback_match[-1], "is_correct": 1 if fallback_match[-1] == correct_answer.upper() else 0, "eval_method": "Regex_Fallback"}

    return {"extracted_choice": "Format_Error", "is_correct": 0, "eval_method": "Failed"}

# ==========================================
# 🧠 2. 답변 생성 함수
# ==========================================
def generate_mmlu_answer(model, processor, instruction, user_input):
    messages = [{"role": "system", "content": instruction}, {"role": "user", "content": user_input}]
    inputs = processor.apply_chat_template(
        messages,
        tokenize=True,
        return_dict=True,
        return_tensors="pt",
        add_generation_prompt=True,
        enable_thinking=False
    ).to(model.device)
    input_len = inputs["input_ids"].shape[-1]
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_new_tokens=1024,
            temperature=0.1,
            do_sample=True,
            top_p=0.9
        )
    return processor.decode(outputs[0][input_len:], skip_special_tokens=True)

# ==========================================
# 🎯 3. 카테고리별 균등 샘플링 함수
# ==========================================
def sample_by_category(full_dataset, samples_per_category, seed):
    category_to_items = {}
    for item in full_dataset:
        category_to_items.setdefault(item['category'], []).append(item)

    print(f"\n📊 전체 카테고리 수: {len(category_to_items)}개", flush=True)

    random.seed(seed)
    sampled_dataset = []
    for category, items in sorted(category_to_items.items()):
        if len(items) >= samples_per_category:
            sampled = random.sample(items, samples_per_category)
        else:
            sampled = items
            print(f"⚠️  '{category}' 카테고리는 {len(items)}개뿐이라 전체 사용 (요청: {samples_per_category}개)", flush=True)
        sampled_dataset.extend(sampled)
        print(f"   - {category:<20}: {len(sampled)}개 추출", flush=True)

    random.shuffle(sampled_dataset)  # 카테고리별로 뭉쳐있지 않도록 순서 섞기
    return sampled_dataset

# ==========================================
# 🚀 4. 메인 파이프라인
# ==========================================
def main():
    print(f"\n{'='*60}\n🏆 MMLU-Pro 평가 시작: [GEMMA-4-12B (Multimodal)]\n{'='*60}", flush=True)

    full_dataset = list(load_dataset("TIGER-Lab/MMLU-Pro", split="test", cache_dir=os.environ["HF_DATASETS_CACHE"]))
    dataset = sample_by_category(full_dataset, SAMPLES_PER_CATEGORY, SEED)

    TOTAL_SAMPLES = len(dataset)
    print(f"\n✅ 최종 평가 데이터셋: {TOTAL_SAMPLES}개 (카테고리별 최대 {SAMPLES_PER_CATEGORY}개)", flush=True)

    OUTPUT_CSV = f"results_mmlu_pro_strict_gemma_12b_{SAMPLES_PER_CATEGORY}percat.csv"

    print("\n⏳ 프로세서 및 모델 로딩 중...", flush=True)
    processor = AutoProcessor.from_pretrained(MODEL_ID, trust_remote_code=True)
    base_model = AutoModelForMultimodalLM.from_pretrained(
        MODEL_ID,
        torch_dtype=torch.bfloat16,
        device_map="auto",
        trust_remote_code=True
    )

    model = PeftModel.from_pretrained(base_model, ADAPTER_PATH)
    model.eval()

    results_list = []

    # 🟢 PHASE 1: Base 모델 평가
    print(f"\n🚀 [PHASE 1] Base 모델 채점을 시작합니다!", flush=True)
    model.disable_adapter_layers()
    for i, data in enumerate(dataset):
        instruction, user_input = format_mmlu_pro_prompt(data['question'], data['options'])
        ans_base = generate_mmlu_answer(model, processor, instruction, user_input)
        scores_base = judge_mmlu_pro(ans_base, data['answer'])

        results_list.append({
            "question_id": data.get('question_id', i), "category": data['category'],
            "model_type": "original", "is_finetuned": 0, "correct_answer": data['answer'],
            "extracted_choice": scores_base.get("extracted_choice", "None"),
            "is_correct": scores_base.get("is_correct", 0), "answer_text": ans_base
        })
        print(f"👉 [BASE | {i+1}/{TOTAL_SAMPLES}] 카테고리: {data['category']:<15} | 정답: {data['answer']} | 추출: {scores_base.get('extracted_choice')} -> {'✅ 맞춤' if scores_base.get('is_correct') else '❌ 틀림'}")
        if (i + 1) % 10 == 0: pd.DataFrame(results_list).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # 🟠 PHASE 2: Warm 모델 평가
    print(f"\n🚀 [PHASE 2] Warm 모델 채점을 시작합니다!", flush=True)
    model.enable_adapter_layers()
    for i, data in enumerate(dataset):
        instruction, user_input = format_mmlu_pro_prompt(data['question'], data['options'])
        ans_lora = generate_mmlu_answer(model, processor, instruction, user_input)
        scores_lora = judge_mmlu_pro(ans_lora, data['answer'])

        results_list.append({
            "question_id": data.get('question_id', i), "category": data['category'],
            "model_type": "warm", "is_finetuned": 1, "correct_answer": data['answer'],
            "extracted_choice": scores_lora.get("extracted_choice", "None"),
            "is_correct": scores_lora.get("is_correct", 0), "answer_text": ans_lora
        })
        print(f"👉 [WARM | {i+1}/{TOTAL_SAMPLES}] 카테고리: {data['category']:<15} | 정답: {data['answer']} | 추출: {scores_lora.get('extracted_choice')} -> {'✅ 맞춤' if scores_lora.get('is_correct') else '❌ 틀림'}")
        if (i + 1) % 10 == 0: pd.DataFrame(results_list).to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    # ==========================================
    # 📊 최종 요약 (카테고리별 출력 복구)
    # ==========================================
    df = pd.DataFrame(results_list)
    df.to_csv(OUTPUT_CSV, index=False, encoding='utf-8-sig')

    print("\n" + "="*60)
    print(f"🏆 최종 평가 요약 [GEMMA-4-12B - {TOTAL_SAMPLES}개] 🏆")
    for m_type in ["original", "warm"]:
        m_df = df[df['model_type'] == m_type]
        if not m_df.empty:
            overall_acc = m_df['is_correct'].mean() * 100
            print(f"\n[{m_type.upper()} Model] 전체 정답률: {overall_acc:.2f}%")
            for cat, acc in (m_df.groupby('category')['is_correct'].mean() * 100).items():
                print(f"  - {cat:<20}: {acc:.2f}%")
    print("="*60)

if __name__ == "__main__":
    main()
