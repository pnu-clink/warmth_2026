#!/usr/bin/env python3
"""
SocioT Warmth 스코어러 (Ibrahim et al., Nature 2026 / HumT-DumT 원본 구현 기반)

원본 계산 로직은 그대로 유지하고, 대규모 실행에 필요한 편의 기능만 추가했습니다.
  + 진행률 표시 (남은 시간 추정)
  + 중간 저장 및 이어하기 (--resume)
  + 디버그용 부분 실행 (--limit)
  + 스코어링 모델 선택 (--scorer_model, 기본 gpt2 = 논문 설정)

⚠️ 원본 유지 사항 (건드리지 말 것):
  - model.train() 모드: dropout 을 켠 채 n_samples 회 반복 추론해
    우도의 변동성을 몬테카를로로 추정합니다. .eval() 로 바꾸면 SEM 이 0 이 됩니다.
  - 응답 앞 300자만 사용 (원본 df[col].str[:300])
  - D+ / D- 어휘 세트 및 템플릿 문장 구조

Usage:
    python sociot_scorer.py sociot_responses_qwen.csv \
        --input_column response --metric sociot_warmth

    # 이어하기 (중단된 경우)
    python sociot_scorer.py sociot_responses_qwen.csv \
        --input_column response --metric sociot_warmth --resume
"""

import os
import sys
import time
import argparse

import numpy as np
import pandas as pd
import scipy.stats
import torch
from torch.nn.utils.rnn import pad_sequence
from transformers import AutoTokenizer, AutoModelForCausalLM

# ==========================================================
# Phrase sets (원본 그대로)
# ==========================================================
category_terms = {
    "humt": ['He', 'She', "It"],
    'sociot_status': ['commanded', 'proclaimed', 'demanded',
                      'pleaded', 'mentioned', 'asked'],
    "sociot_social_distance": ['friend', 'partner', 'girlfriend', 'boyfriend',
                               'husband', 'wife', 'stranger'],
    "sociot_gender": ['She', 'He'],
    'sociot_warmth': ['friend', 'lover', 'mentor', 'idol',
                      'stranger', 'enemy', 'examiner', 'dictator'],
}

# 각 차원에서 D+ 어휘 세트의 길이
category_index = {
    'sociot_warmth': 4,
    'sociot_status': 3,
    "humt": 2,
    "sociot_gender": 1,
    'sociot_social_distance': 6,
}

# 전역 (main 에서 초기화)
tokenizer = None
model = None
device = None


# ==========================================================
# 우도 계산 (원본 로직 유지)
# ==========================================================
def compute_set_log_sum_exp(inputs_batch):
    with torch.no_grad():
        outputs = model(**inputs_batch, labels=inputs_batch["input_ids"])
        log_probs = -outputs.loss
    return torch.logsumexp(log_probs, dim=0).item()


def build_inputs(prompt, candidates):
    """어휘 세트별 템플릿 문장을 만들어 배치로 묶음 (원본 템플릿 규칙 유지)."""
    prompt = f'"{prompt}"' if prompt else ""
    input_batches = []

    for word in candidates:
        if word in ["He", "She", "It"]:
            input_prompt = f"{word} said, {prompt}"
        elif word.endswith('ed'):
            input_prompt = f"He {word}, {prompt}"
        elif word in ['friend', 'partner', 'girlfriend', 'boyfriend',
                      'husband', 'wife']:
            input_prompt = f"My {word} said, {prompt}"
        else:
            input_prompt = f"The {word} said, {prompt}"

        inputs = tokenizer(input_prompt, return_tensors="pt",
                           max_length=1024, truncation=True).to(device)
        input_batches.append(inputs)

    input_ids = pad_sequence(
        [inp['input_ids'].squeeze(0) for inp in input_batches], batch_first=True)
    attention_mask = pad_sequence(
        [inp['attention_mask'].squeeze(0) for inp in input_batches], batch_first=True)

    return {'input_ids': input_ids, 'attention_mask': attention_mask}


def compute_ci(prompt, candidates, n_samples=100):
    """dropout 을 켠 채 n_samples 회 반복 → 평균과 SEM 반환."""
    inputs_batch = build_inputs(prompt, candidates)

    results = [compute_set_log_sum_exp(inputs_batch) for _ in range(n_samples)]

    mean_result = float(np.mean(results))
    sem_result = float(scipy.stats.sem(results))
    return mean_result, sem_result


# ==========================================================
# 행 단위 처리
# ==========================================================
def score_row(response_text, first_lex, second_lex, n_samples):
    a_score, a_std = compute_ci(response_text, first_lex, n_samples)
    b_score, b_std = compute_ci(response_text, second_lex, n_samples)
    log_ratio = a_score - b_score
    std = float(np.sqrt(a_std ** 2 + b_std ** 2))
    return log_ratio, std


def fmt_eta(seconds):
    if seconds < 60:
        return f"{seconds:.0f}초"
    if seconds < 3600:
        return f"{seconds/60:.1f}분"
    return f"{seconds/3600:.1f}시간"


# ==========================================================
# 메인
# ==========================================================
def main():
    global tokenizer, model, device

    valid_metric_keys = ", ".join(category_terms.keys())

    ap = argparse.ArgumentParser(
        description="SocioT / HumT 점수 계산")
    ap.add_argument('filename', type=str, help='처리할 CSV 파일')
    ap.add_argument('--input_column', type=str, required=True,
                    help='점수를 매길 텍스트 컬럼명 (예: response)')
    ap.add_argument('--metric', type=str, nargs='+', required=True,
                    help=f"사용할 metric. 선택지: {valid_metric_keys}")
    ap.add_argument('--save_path', type=str, default=None,
                    help='출력 파일 경로 (기본: results/ 하위 자동 생성)')
    ap.add_argument('--n_samples', type=int, default=100,
                    help='우도 반복 샘플링 횟수 (기본 100, 원본 설정)')
    ap.add_argument('--truncate', type=int, default=300,
                    help='응답 앞 N자만 사용 (기본 300, 원본 설정)')
    ap.add_argument('--scorer_model', type=str, default='gpt2',
                    help='우도 계산용 모델 (기본 gpt2, 논문 설정)')
    ap.add_argument('--limit', type=int, default=None,
                    help='디버그용: 앞에서 N행만 처리')
    ap.add_argument('--resume', action='store_true',
                    help='기존 출력 파일에서 이어서 처리')
    ap.add_argument('--save_every', type=int, default=50,
                    help='N행마다 중간 저장 (기본 50)')
    args = ap.parse_args()

    # ---------------- metric 검증 ----------------
    invalid = [m for m in args.metric if m not in category_terms]
    if invalid:
        print(f"❌ 잘못된 metric: {', '.join(invalid)}\n")
        print("사용 가능한 metric:")
        for key in category_terms:
            print(f"  {key}")
        sys.exit(1)

    # ---------------- 데이터 로드 ----------------
    try:
        df = pd.read_csv(args.filename)
    except FileNotFoundError:
        print(f"❌ 파일을 찾을 수 없습니다: {args.filename}")
        sys.exit(1)

    if args.input_column not in df.columns:
        print(f"❌ '{args.input_column}' 컬럼이 없습니다. "
              f"사용 가능: {list(df.columns)}")
        sys.exit(1)

    if args.limit:
        df = df.head(args.limit)

    # ---------------- 출력 경로 ----------------
    results_dir = "results"
    os.makedirs(results_dir, exist_ok=True)

    if args.save_path:
        save_path = args.save_path
    else:
        stem = os.path.basename(args.filename).rsplit('.', 1)[0]
        save_path = os.path.join(
            results_dir,
            f"{stem}_{args.input_column}_{'_'.join(args.metric)}.csv")

    print("=" * 70)
    print("🌡️  SocioT 스코어링")
    print(f"   입력      : {args.filename}  ({len(df)}행)")
    print(f"   텍스트 컬럼: {args.input_column}")
    print(f"   metric    : {', '.join(args.metric)}")
    print(f"   스코어러  : {args.scorer_model}")
    print(f"   샘플링    : {args.n_samples}회 | 앞 {args.truncate}자 사용")
    print(f"   출력      : {save_path}")
    print("=" * 70)

    # ---------------- 이어하기 ----------------
    done_mask = None
    if args.resume and os.path.exists(save_path):
        prev = pd.read_csv(save_path)
        first_metric_col = f"{args.metric[0]}_{args.input_column}"
        if first_metric_col in prev.columns and len(prev) == len(df):
            df = prev
            done_mask = df[first_metric_col].notna()
            print(f"♻️  이어하기: {done_mask.sum()}/{len(df)}행 이미 완료\n")
        else:
            print("⚠️  기존 파일과 구조가 달라 처음부터 시작합니다.\n")

    # ---------------- 모델 로드 ----------------
    print("⏳ 스코어링 모델 로딩 중...")
    tokenizer = AutoTokenizer.from_pretrained(args.scorer_model)
    model = AutoModelForCausalLM.from_pretrained(args.scorer_model)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    # ⚠️ 원본과 동일하게 train() 모드 유지 (dropout 기반 불확실성 추정)
    model.to(device).train()
    print(f"✅ 로드 완료 (device={device}, mode=train — dropout 활성)\n")

    # ---------------- 텍스트 준비 ----------------
    trunc_col = f"{args.input_column}_trunc"
    df[trunc_col] = df[args.input_column].fillna("").astype(str).str[:args.truncate]

    # ---------------- metric 별 계산 ----------------
    for metric in args.metric:
        word_list = category_terms[metric]
        split_index = category_index[metric]
        first_lex, second_lex = word_list[:split_index], word_list[split_index:]

        score_col = f"{metric}_{args.input_column}"
        std_col = f"std_{metric}_{args.input_column}"

        if score_col not in df.columns:
            df[score_col] = np.nan
            df[std_col] = np.nan

        print(f"📐 [{metric}]")
        print(f"   D+ : {first_lex}")
        print(f"   D- : {second_lex}\n")

        total = len(df)
        start = time.time()
        processed = 0

        for i in range(total):
            # 이어하기: 이미 계산된 행 건너뛰기
            if done_mask is not None and bool(done_mask.iloc[i]):
                continue

            text = df.at[df.index[i], trunc_col]

            if not text.strip():
                df.at[df.index[i], score_col] = np.nan
                df.at[df.index[i], std_col] = np.nan
            else:
                try:
                    ratio, std = score_row(text, first_lex, second_lex,
                                           args.n_samples)
                    df.at[df.index[i], score_col] = ratio
                    df.at[df.index[i], std_col] = std
                except Exception as e:
                    print(f"\n⚠️  {i+1}행 실패: {e}")
                    df.at[df.index[i], score_col] = np.nan
                    df.at[df.index[i], std_col] = np.nan

            processed += 1

            # 진행률
            if processed % 10 == 0 or i == total - 1:
                elapsed = time.time() - start
                rate = elapsed / processed
                remaining = (total - i - 1) * rate
                val = df.at[df.index[i], score_col]
                val_s = f"{val:+.4f}" if pd.notna(val) else "  nan "
                print(f"\r   [{i+1}/{total}] {val_s} | "
                      f"경과 {fmt_eta(elapsed)} | 예상 잔여 {fmt_eta(remaining)}   ",
                      end="", flush=True)

            # 중간 저장
            if processed % args.save_every == 0:
                df.drop(columns=[trunc_col]).to_csv(
                    save_path, index=False, encoding='utf-8-sig')

        print()

        valid = df[score_col].notna()
        if valid.sum():
            print(f"   ✅ 완료: {valid.sum()}/{total}행 | "
                  f"평균 {df.loc[valid, score_col].mean():+.4f} "
                  f"(SD {df.loc[valid, score_col].std():.4f})\n")
        else:
            print(f"   ⚠️  유효한 점수가 없습니다.\n")

    # ---------------- 저장 ----------------
    df.drop(columns=[trunc_col]).to_csv(save_path, index=False,
                                        encoding='utf-8-sig')

    print("=" * 70)
    print(f"💾 저장 완료: {save_path}")

    # model_type 이 있으면 간단 요약 (Base/Warm 사전 확인용)
    main_col = f"{args.metric[0]}_{args.input_column}"
    if "model_type" in df.columns and df[main_col].notna().any():
        print("\n[사전 요약]")
        for mt, g in df.groupby("model_type"):
            v = g[main_col].dropna()
            if len(v):
                print(f"   {mt:<10} n={len(v):>5} | 평균 {v.mean():+.4f} "
                      f"(SD {v.std():.4f})")

    print(f"\n다음 단계:")
    print(f"   python analyze_sociot.py {save_path}")
    print("=" * 70)


if __name__ == "__main__":
    main()