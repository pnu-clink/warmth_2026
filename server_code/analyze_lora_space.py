#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 파라미터 공간 분석 — Warm vs Neutral 업데이트 비교

베이스 모델이 동일하고 어댑터만 다르므로, ΔW_warm 과 ΔW_neutral 의 차이가
곧 warmth 주입의 파라미터 수준 효과에 해당한다.

계산 항목 (모듈별):
  1. ‖ΔW‖_F              업데이트 크기
  2. cos(ΔW_w, ΔW_n)     두 조건의 업데이트 방향 유사도  ★ 핵심
  3. 주부분공간 겹침       principal angles (선택)

⚠️ GPU 불필요. ΔW 를 실제로 만들지 않고 r×r 행렬 연산으로 계산한다.
   ΔW = s·B@A  (s = alpha/r, B: out×r, A: r×in)
   ⟨ΔW_w, ΔW_n⟩_F = s_w·s_n · tr( (Bw^T Bn)(An Aw^T) )   ← 모두 r×r
   ‖ΔW‖_F²        = s²      · tr( (B^T B)(A A^T) )

Usage:
    python analyze_lora_space.py --root /workspace/warm_models
    python analyze_lora_space.py --root . --subspace     # 주각도까지
"""

import argparse
import json
import os
import re
import sys

import numpy as np
import pandas as pd

try:
    from safetensors.numpy import load_file as st_load
    HAS_ST = True
except ImportError:
    HAS_ST = False

# ---------------------------------------------------------------- 경로 매핑
# (모델키, warm 디렉터리, neutral 디렉터리)
DEFAULT_PAIRS = [
    ("Qwen2.5-7B",  "lora_warm_adapter_qwen2_5_7b",  "lora_neutral_adapter_qwen2_5_7b"),
    ("Llama-3.1-8B", "lora_warm_adapter_llama3_1_8b",       "lora_neutral_adapter_llama3_1_8b"),
    ("Phi-4-14B",   "lora_warm_adapter_phi4_14b",     "lora_neutral_adapter_phi4_14b"),
    ("Gemma-4-12B", "lora_warm_adapter_gemma_12b",    "lora_neutral_adapter_gemma_12b"),
]

LORA_A_PAT = re.compile(r"(.*)\.lora_A(?:\.default)?\.weight$")
LORA_B_PAT = re.compile(r"(.*)\.lora_B(?:\.default)?\.weight$")


# ---------------------------------------------------------------- 로딩
def load_adapter(path):
    """adapter_model.safetensors → {module_name: (A, B)}, scale"""
    st_path = os.path.join(path, "adapter_model.safetensors")
    cfg_path = os.path.join(path, "adapter_config.json")

    if not os.path.exists(st_path):
        # .bin 폴백
        bin_path = os.path.join(path, "adapter_model.bin")
        if os.path.exists(bin_path):
            import torch
            sd = torch.load(bin_path, map_location="cpu")
            tensors = {k: v.float().numpy() for k, v in sd.items()}
        else:
            raise FileNotFoundError(f"어댑터 가중치 없음: {path}")
    else:
        if not HAS_ST:
            raise ImportError("pip install safetensors")
        tensors = st_load(st_path)

    with open(cfg_path) as f:
        cfg = json.load(f)
    r = cfg.get("r", 16)
    alpha = cfg.get("lora_alpha", 32)
    scale = alpha / r

    mods = {}
    for k, v in tensors.items():
        m = LORA_A_PAT.match(k)
        if m:
            mods.setdefault(m.group(1), {})["A"] = np.asarray(v, dtype=np.float64)
            continue
        m = LORA_B_PAT.match(k)
        if m:
            mods.setdefault(m.group(1), {})["B"] = np.asarray(v, dtype=np.float64)

    out = {k: (v["A"], v["B"]) for k, v in mods.items() if "A" in v and "B" in v}
    return out, scale, cfg


def parse_name(full):
    """모듈 경로에서 (층 번호, 모듈 종류) 추출."""
    m = re.search(r"layers?\.(\d+)\.", full)
    layer = int(m.group(1)) if m else -1
    parts = full.split(".")
    mod = parts[-1] if parts else full
    return layer, mod


# ---------------------------------------------------------------- 지표
def fro_sq(A, B, s):
    """‖s·B@A‖_F² = s²·tr((BᵀB)(AAᵀ))  — r×r 연산만."""
    BtB = B.T @ B           # r×r
    AAt = A @ A.T           # r×r
    return float(s * s * np.trace(BtB @ AAt))


def inner(Aw, Bw, sw, An, Bn, sn):
    """⟨ΔW_w, ΔW_n⟩_F = sw·sn·tr((Bwᵀ Bn)(An Awᵀ))."""
    BwtBn = Bw.T @ Bn       # r×r
    AnAwt = An @ Aw.T       # r×r
    return float(sw * sn * np.trace(BwtBn @ AnAwt))


def principal_angles(Aw, Bw, An, Bn, k=None):
    """
    두 업데이트의 좌특이벡터 부분공간 사이 주각도의 코사인.
    ΔW = B@A 의 정확한 SVD 를 작은 행렬로 구한다.
      B = Q_B R_B (QR),  A = ...  →  ΔW = Q_B (R_B A)  이므로
      R_B A 의 SVD 로 좌특이벡터를 얻는다.
    """
    def left_basis(A, B):
        Qb, Rb = np.linalg.qr(B)          # Qb: out×r
        M = Rb @ A                        # r×in
        U, S, _ = np.linalg.svd(M, full_matrices=False)
        return Qb @ U, S                  # out×r
    Uw, Sw = left_basis(Aw, Bw)
    Un, Sn = left_basis(An, Bn)
    if k:
        Uw, Un = Uw[:, :k], Un[:, :k]
    sv = np.linalg.svd(Uw.T @ Un, compute_uv=False)
    return np.clip(sv, 0, 1)


# ---------------------------------------------------------------- 메인
def analyze_pair(name, warm_dir, neut_dir, want_subspace=False, top_k=8):
    print(f"\n{'='*74}")
    print(f"🧩 {name}")
    print(f"{'='*74}")

    try:
        W, sw, cfg_w = load_adapter(warm_dir)
        N, sn, cfg_n = load_adapter(neut_dir)
    except Exception as e:
        print(f"   ❌ 로드 실패: {e}")
        return None

    print(f"   warm    r={cfg_w.get('r')} α={cfg_w.get('lora_alpha')} "
          f"모듈 {len(W)}개")
    print(f"   neutral r={cfg_n.get('r')} α={cfg_n.get('lora_alpha')} "
          f"모듈 {len(N)}개")

    if cfg_w.get("r") != cfg_n.get("r") or cfg_w.get("lora_alpha") != cfg_n.get("lora_alpha"):
        print("   ⚠️  두 조건의 LoRA 설정이 다릅니다 — 비교 해석에 주의")

    common = sorted(set(W) & set(N))
    only_w, only_n = set(W) - set(N), set(N) - set(W)
    if only_w or only_n:
        print(f"   ⚠️  공통 아닌 모듈: warm만 {len(only_w)}, neutral만 {len(only_n)}")
    print(f"   공통 모듈 {len(common)}개 비교\n")

    rows = []
    for full in common:
        Aw, Bw = W[full]
        An, Bn = N[full]
        if Aw.shape != An.shape or Bw.shape != Bn.shape:
            continue

        nw = np.sqrt(fro_sq(Aw, Bw, sw))
        nn = np.sqrt(fro_sq(An, Bn, sn))
        ip = inner(Aw, Bw, sw, An, Bn, sn)
        cos = ip / (nw * nn) if nw > 0 and nn > 0 else np.nan

        layer, mod = parse_name(full)
        rec = {"model": name, "module_full": full, "layer": layer,
               "module": mod, "norm_warm": nw, "norm_neutral": nn,
               "norm_ratio": nw / nn if nn > 0 else np.nan,
               "cosine": cos}

        if want_subspace:
            try:
                pa = principal_angles(Aw, Bw, An, Bn, k=top_k)
                rec["subspace_mean_cos"] = float(np.mean(pa))
                rec["subspace_top1_cos"] = float(pa[0])
            except Exception:
                rec["subspace_mean_cos"] = np.nan
                rec["subspace_top1_cos"] = np.nan
        rows.append(rec)

    df = pd.DataFrame(rows)
    if len(df) == 0:
        print("   ❌ 비교 가능한 모듈이 없습니다.")
        return None

    # ---- 전체 요약 ----
    # 전역 코사인: 모든 모듈을 하나의 벡터로 이어붙인 것과 동일
    tot_ip = sum(inner(*W[f], sw, *N[f], sn) for f in common
                 if W[f][0].shape == N[f][0].shape)
    tot_nw = np.sqrt(sum(fro_sq(*W[f], sw) for f in common
                         if W[f][0].shape == N[f][0].shape))
    tot_nn = np.sqrt(sum(fro_sq(*N[f], sn) for f in common
                         if W[f][0].shape == N[f][0].shape))
    global_cos = tot_ip / (tot_nw * tot_nn)

    print(f"   [전역]")
    print(f"     ‖ΔW_warm‖_F     = {tot_nw:.4f}")
    print(f"     ‖ΔW_neutral‖_F  = {tot_nn:.4f}")
    print(f"     크기 비 (W/N)    = {tot_nw/tot_nn:.4f}")
    print(f"     ★ 전역 코사인    = {global_cos:+.4f}")

    interp = ("거의 같은 방향" if global_cos > 0.8 else
              "상당히 겹침" if global_cos > 0.5 else
              "부분적으로 다른 방향" if global_cos > 0.2 else
              "거의 직교 — 서로 다른 방향")
    print(f"       → {interp}")

    # ---- 모듈 종류별 ----
    print(f"\n   [모듈 종류별 평균]")
    g = df.groupby("module").agg(
        n=("cosine", "size"), cos=("cosine", "mean"),
        nw=("norm_warm", "mean"), nn=("norm_neutral", "mean"))
    g["ratio"] = g.nw / g.nn
    g = g.sort_values("cos")
    print(f"     {'module':<16}{'n':>4}{'cos':>9}{'‖W‖':>10}{'‖N‖':>10}{'W/N':>8}")
    for m, r in g.iterrows():
        print(f"     {m:<16}{int(r.n):>4}{r.cos:>+9.3f}"
              f"{r.nw:>10.4f}{r.nn:>10.4f}{r.ratio:>8.3f}")

    # ---- 층 구간별 ----
    if (df.layer >= 0).any():
        d = df[df.layer >= 0].copy()
        nl = d.layer.max() + 1
        d["depth"] = pd.cut(d.layer, bins=[-1, nl/3, 2*nl/3, nl],
                            labels=["early", "middle", "late"])
        print(f"\n   [층 구간별 평균] (총 {nl}층)")
        gd = d.groupby("depth", observed=True).agg(
            cos=("cosine", "mean"), nw=("norm_warm", "mean"),
            nn=("norm_neutral", "mean"))
        for dep, r in gd.iterrows():
            print(f"     {str(dep):<8} cos={r.cos:+.3f}  "
                  f"‖W‖={r.nw:.4f}  ‖N‖={r.nn:.4f}  W/N={r.nw/r.nn:.3f}")

    df.attrs["global_cos"] = global_cos
    return df, {"model": name, "norm_warm": tot_nw, "norm_neutral": tot_nn,
                "norm_ratio": tot_nw / tot_nn, "global_cosine": global_cos,
                "n_modules": len(df)}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".", help="어댑터들이 있는 상위 디렉터리")
    ap.add_argument("--subspace", action="store_true",
                    help="주각도(principal angles)도 계산 (느림)")
    ap.add_argument("--top_k", type=int, default=8)
    ap.add_argument("--out", default="lora_space_modules.csv")
    ap.add_argument("--out_summary", default="lora_space_summary.csv")
    args = ap.parse_args()

    print("=" * 74)
    print("🔬 LoRA 파라미터 공간 분석 — Warm vs Neutral")
    print("=" * 74)
    print(f"   root: {os.path.abspath(args.root)}")

    all_df, summaries = [], []
    for name, wd, nd in DEFAULT_PAIRS:
        wp, np_ = os.path.join(args.root, wd), os.path.join(args.root, nd)
        if not (os.path.isdir(wp) and os.path.isdir(np_)):
            print(f"\n   ⏭️  {name}: 디렉터리 없음 → 건너뜀")
            continue
        res = analyze_pair(name, wp, np_, args.subspace, args.top_k)
        if res:
            df, summ = res
            all_df.append(df)
            summaries.append(summ)

    if not summaries:
        print("\n❌ 분석된 모델이 없습니다.")
        sys.exit(1)

    print(f"\n{'='*74}")
    print("🏁 전체 요약")
    print(f"{'='*74}")
    s = pd.DataFrame(summaries)
    print(f"   {'model':<14}{'‖W‖':>10}{'‖N‖':>10}{'W/N':>8}{'global cos':>12}")
    for _, r in s.iterrows():
        print(f"   {r.model:<14}{r.norm_warm:>10.4f}{r.norm_neutral:>10.4f}"
              f"{r.norm_ratio:>8.3f}{r.global_cosine:>+12.4f}")

    print(f"\n   해석 지침")
    print(f"     cos ≈ 1    두 조건이 같은 방향으로 이동 (크기만 다름)")
    print(f"     cos ≈ 0    서로 다른 방향 → 이중 해리의 파라미터 수준 증거")
    print(f"     cos < 0    반대 방향 (드묾)")

    pd.concat(all_df).to_csv(args.out, index=False, encoding="utf-8-sig")
    s.to_csv(args.out_summary, index=False, encoding="utf-8-sig")
    print(f"\n💾 {args.out}          모듈별 상세")
    print(f"💾 {args.out_summary}   모델별 요약")
    print("=" * 74)


if __name__ == "__main__":
    main()
