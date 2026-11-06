import os
os.environ["HF_HOME"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache')
os.environ["HF_DATASETS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/datasets')
os.environ["TRANSFORMERS_CACHE"] = os.path.join(os.environ.get("WARM_MODEL_ROOT", os.getcwd()), 'hf_cache/models')

import argparse
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F
from transformers import (
    AutoModelForCausalLM, AutoTokenizer, AutoConfig,
    AutoModelForMultimodalLM, AutoProcessor
)
from peft import PeftModel

# ==========================================================
# 📐 분포 이동(KL) 측정 — Base / Neutral / Warm
#
# 프로토콜: Shenfeld et al. (RL's Razor, ICLR 2026)
#   "파인튜닝 모델과 베이스 정책 간 KL 발산을 **새 과제 분포에서** 측정하면
#    망각의 정도를 예측한다"
#
#   본 연구의 새 과제 = ShareGPT 대화 (학습 데이터와 동일 분포)
#   → held-out 프롬프트 + Base 생성 응답 위에서 측정
#
# 방식: teacher forcing (재생성 없음)
#   동일한 텍스트를 세 조건에 통과시켜 각 위치의 분포를 얻고 KL 계산
#   → 텍스트 차이가 개입하지 않아 조건 간 비교가 대칭적
# ==========================================================

ROOT = os.environ.get("WARM_MODEL_ROOT", os.getcwd())

MODEL_REGISTRY = {
    "qwen": {
        "base_id": "Qwen/Qwen2.5-7B-Instruct",
        "warm_path":    f"{ROOT}/lora_warm_adapter_qwen2_5_7b",
        "neutral_path": f"{ROOT}/lora_neutral_adapter_qwen2_5_7b",
        "loader": "causal",
    },
    "llama": {
        "base_id": "meta-llama/Llama-3.1-8B-Instruct",
        "warm_path":    f"{ROOT}/lora_warm_adapter_llama3_1_8b",
        "neutral_path": f"{ROOT}/lora_neutral_adapter_llama3_1_8b",
        "loader": "causal",
    },
    "phi4": {
        "base_id": "microsoft/phi-4",
        "warm_path":    f"{ROOT}/lora_warm_adapter_phi4_14b",
        "neutral_path": f"{ROOT}/lora_neutral_adapter_phi4_14b",
        "loader": "causal",
    },
    "gemma": {
        "base_id": "google/gemma-4-12b-it",
        "warm_path":    f"{ROOT}/lora_warm_adapter_gemma_12b",
        "neutral_path": f"{ROOT}/lora_neutral_adapter_gemma_12b",
        "loader": "multimodal",
    },
}

CONDS = ["original", "neutral", "warm"]
MAX_INPUT_TOKENS = 2048
MAX_RESP_TOKENS = 320      # 생성 설정이 max 300 이므로 여유


# ==========================================================
def build_inputs(encoder, prompt, response, loader, device):
    """
    프롬프트(chat template) + 응답을 이어붙이고,
    응답이 시작되는 위치(prompt_len)를 함께 반환.
    """
    messages = [{"role": "user", "content": prompt}]

    if loader == "multimodal":
        enc = encoder.apply_chat_template(
            messages, tokenize=True, return_dict=True, return_tensors="pt",
            add_generation_prompt=True, enable_thinking=False)
        prompt_ids = enc["input_ids"][0]
        resp_ids = encoder.tokenizer(
            response, return_tensors="pt", add_special_tokens=False
        )["input_ids"][0]
    else:
        text = encoder.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True)
        prompt_ids = encoder(text, return_tensors="pt",
                             truncation=True, max_length=MAX_INPUT_TOKENS
                             )["input_ids"][0]
        resp_ids = encoder(response, return_tensors="pt",
                           add_special_tokens=False)["input_ids"][0]

    resp_ids = resp_ids[:MAX_RESP_TOKENS]
    if len(resp_ids) == 0:
        return None, None, None

    ids = torch.cat([prompt_ids, resp_ids]).unsqueeze(0).to(device)
    return ids, len(prompt_ids), len(resp_ids)


@torch.no_grad()
def logprobs_at_response(model, ids, prompt_len):
    """
    응답 위치들의 log-softmax 분포를 반환. shape = (resp_len, vocab)
    위치 i 의 예측은 logits[i-1] 이므로 한 칸 밀어서 슬라이스.
    """
    out = model(input_ids=ids)
    logits = out.logits[0, prompt_len - 1:-1, :]     # (resp_len, vocab)
    return F.log_softmax(logits.float(), dim=-1)


def kl_from_logprobs(logP, logQ):
    """KL(P‖Q) = Σ P (logP − logQ), 위치별 값 반환."""
    P = logP.exp()
    return (P * (logP - logQ)).sum(dim=-1)           # (resp_len,)


def bin_thirds(v):
    """위치별 값을 초/중/후 3구간 평균으로."""
    n = len(v)
    if n < 3:
        m = float(v.mean())
        return m, m, m
    a, b = n // 3, 2 * n // 3
    return float(v[:a].mean()), float(v[a:b].mean()), float(v[b:].mean())


# ==========================================================
def main():
    ap = argparse.ArgumentParser(
        description="Base/Neutral/Warm 분포 이동(KL) 측정")
    ap.add_argument("--model", required=True, choices=list(MODEL_REGISTRY))
    ap.add_argument("--responses", default=None,
                    help="3조건 응답 CSV (기본: sociot_responses_{model}_3way.csv)")
    ap.add_argument("--text_from", default="original",
                    choices=["original", "neutral", "warm"],
                    help="어느 조건의 생성 텍스트 위에서 측정할지 (기본: Base)")
    ap.add_argument("--limit", type=int, default=None)
    ap.add_argument("--save_every", type=int, default=50)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    cfg = MODEL_REGISTRY[args.model]
    loader = cfg["loader"]
    resp_csv = args.responses or f"sociot_responses_{args.model}_3way.csv"
    out_csv = args.out or f"kl_divergence_{args.model}.csv"

    print("=" * 66)
    print(f"📐 분포 이동(KL) 측정: [{args.model.upper()}]")
    print(f"   측정 텍스트: {args.text_from} 조건의 생성 응답 (teacher forcing)")
    print("=" * 66)

    if not os.path.exists(resp_csv):
        print(f"❌ 응답 파일 없음: {resp_csv}")
        return

    df = pd.read_csv(resp_csv)
    base_df = df[df.model_type == args.text_from].copy()
    base_df = base_df[base_df.response.fillna("").str.len() > 0]
    if args.limit:
        base_df = base_df.head(args.limit)
    print(f"✅ 측정 대상 {len(base_df)}건")

    # ---------------- 모델 로드 ----------------
    print("\n⏳ 모델 로딩 중...", flush=True)
    if loader == "multimodal":
        encoder = AutoProcessor.from_pretrained(cfg["base_id"], trust_remote_code=True)
        base_model = AutoModelForMultimodalLM.from_pretrained(
            cfg["base_id"], torch_dtype=torch.bfloat16,
            device_map={"": 0}, trust_remote_code=True)
    else:
        encoder = AutoTokenizer.from_pretrained(cfg["base_id"], trust_remote_code=True)
        mcfg = AutoConfig.from_pretrained(cfg["base_id"], trust_remote_code=True)
        base_model = AutoModelForCausalLM.from_pretrained(
            cfg["base_id"], config=mcfg, torch_dtype=torch.bfloat16,
            device_map="auto", trust_remote_code=True, attn_implementation="sdpa")

    model = PeftModel.from_pretrained(base_model, cfg["warm_path"], adapter_name="warm")
    model.load_adapter(cfg["neutral_path"], adapter_name="neutral")
    model.eval()
    device = next(model.parameters()).device
    print(f"✅ 어댑터 로드 완료 (device={device})\n", flush=True)

    def switch(cond):
        if cond == "original":
            model.disable_adapter_layers()
        else:
            model.enable_adapter_layers()
            model.set_adapter(cond)

    # ---------------- 측정 ----------------
    rows = []
    n_fail = 0
    for i, r in enumerate(base_df.itertuples(index=False)):
        try:
            ids, plen, rlen = build_inputs(encoder, r.prompt, r.response,
                                           loader, device)
            if ids is None:
                n_fail += 1
                continue

            lp = {}
            for c in CONDS:
                switch(c)
                lp[c] = logprobs_at_response(model, ids, plen)

            # 토큰 단위 KL (위치별)
            kl_nb = kl_from_logprobs(lp["neutral"], lp["original"])   # KL(N‖B)
            kl_wb = kl_from_logprobs(lp["warm"],    lp["original"])   # KL(W‖B)
            kl_wn = kl_from_logprobs(lp["warm"],    lp["neutral"])    # KL(W‖N)
            # 역방향 (문헌은 양방향 모두 예측력 있다고 보고)
            kl_bn = kl_from_logprobs(lp["original"], lp["neutral"])
            kl_bw = kl_from_logprobs(lp["original"], lp["warm"])

            # 텍스트 자체의 로그우도 (조건별 perplexity 진단용)
            tgt = ids[0, plen:]
            nll = {c: float(-lp[c].gather(1, tgt.unsqueeze(1)).mean()) for c in CONDS}

            e1, m1, l1 = bin_thirds(kl_wn)

            rows.append({
                "id": r.id, "query_type": r.query_type, "model": args.model,
                "n_resp_tokens": rlen,
                "KL_neutral_base": float(kl_nb.mean()),
                "KL_warm_base":    float(kl_wb.mean()),
                "KL_warm_neutral": float(kl_wn.mean()),
                "KL_base_neutral": float(kl_bn.mean()),
                "KL_base_warm":    float(kl_bw.mean()),
                "KLwn_early": e1, "KLwn_mid": m1, "KLwn_late": l1,
                "nll_base": nll["original"], "nll_neutral": nll["neutral"],
                "nll_warm": nll["warm"],
            })

            del lp
            if (i + 1) % 10 == 0:
                last = rows[-1]
                print(f"👉 [{i+1}/{len(base_df)}] "
                      f"KL(N‖B)={last['KL_neutral_base']:.4f}  "
                      f"KL(W‖B)={last['KL_warm_base']:.4f}  "
                      f"KL(W‖N)={last['KL_warm_neutral']:.4f}", flush=True)
            if (i + 1) % args.save_every == 0:
                pd.DataFrame(rows).to_csv(out_csv, index=False, encoding="utf-8-sig")

        except Exception as e:
            n_fail += 1
            print(f"⚠️  [{i+1}] 실패: {e}", flush=True)
            torch.cuda.empty_cache()

    if not rows:
        print("❌ 측정된 항목이 없습니다.")
        return

    out = pd.DataFrame(rows)
    out.to_csv(out_csv, index=False, encoding="utf-8-sig")

    # ---------------- 요약 ----------------
    print("\n" + "=" * 66)
    print(f"🏁 요약 [{args.model.upper()}]  (n={len(out)}"
          + (f", 실패 {n_fail}" if n_fail else "") + ")")
    print("=" * 66)

    def stat(col):
        v = out[col]
        return f"{v.mean():.4f} ± {v.std():.4f}"

    print(f"\n[분포 이동 — 정방향 KL(fine-tuned ‖ base)]")
    print(f"   KL(Neutral‖Base) = {stat('KL_neutral_base')}")
    print(f"   KL(Warm‖Base)    = {stat('KL_warm_base')}")
    print(f"   KL(Warm‖Neutral) = {stat('KL_warm_neutral')}")

    print(f"\n[역방향]")
    print(f"   KL(Base‖Neutral) = {stat('KL_base_neutral')}")
    print(f"   KL(Base‖Warm)    = {stat('KL_base_warm')}")

    # Warm 이 Base 로부터 더 멀리 갔는가
    d = out.KL_warm_base - out.KL_neutral_base
    from scipy import stats as sps
    t, p = sps.wilcoxon(out.KL_warm_base, out.KL_neutral_base)
    print(f"\n[Warm 이 Base 로부터 더 멀리 이동했는가]")
    print(f"   차이 = {d.mean():+.4f}  (Wilcoxon p={p:.3g})")
    if p < 0.05:
        print(f"   → {'Warm 이 더 멀다' if d.mean() > 0 else 'Neutral 이 더 멀다'}")
    else:
        print(f"   → 유의한 차이 없음")

    print(f"\n[응답 내 위치별 KL(W‖N)]")
    print(f"   초반 {out.KLwn_early.mean():.4f} · "
          f"중반 {out.KLwn_mid.mean():.4f} · "
          f"후반 {out.KLwn_late.mean():.4f}")

    print(f"\n[query_type 별 KL(Warm‖Neutral)]")
    for q, g in out.groupby("query_type"):
        print(f"   {q:<10} n={len(g):>4}  {g.KL_warm_neutral.mean():.4f}")

    print(f"\n[텍스트 NLL — 조건별 (낮을수록 그 텍스트를 잘 예측)]")
    print(f"   Base {out.nll_base.mean():.4f} · "
          f"Neutral {out.nll_neutral.mean():.4f} · "
          f"Warm {out.nll_warm.mean():.4f}")

    print(f"\n💾 {out_csv}")
    print("=" * 66)


if __name__ == "__main__":
    main()
