#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
임의의 두 LoRA 어댑터 방향 비교 (기준선 측정용)

시드만 다른 두 Neutral 어댑터를 비교하여, Warm–Neutral 코사인 해석의
상한 기준을 얻는다.

  cos(ΔW_neutral_seed42, ΔW_neutral_seed1234)  =  "같은 조건 두 실행"의 코사인
       ↕ 비교
  cos(ΔW_warm,           ΔW_neutral)           =  관측값

Usage:
    # 시드 기준선
    python compare_two_adapters.py \
        --a lora_neutral_adapter_phi4_14b \
        --b lora_neutral_adapter_phi4_14b_seed1234 \
        --label_a "Neutral(seed42)" --label_b "Neutral(seed1234)"

    # 관측값 재확인
    python compare_two_adapters.py \
        --a lora_warm_adapter_phi4_14b \
        --b lora_neutral_adapter_phi4_14b \
        --label_a Warm --label_b Neutral
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
except ImportError:
    print("pip install safetensors 필요"); sys.exit(1)

A_PAT = re.compile(r"(.*)\.lora_A(?:\.default)?\.weight$")
B_PAT = re.compile(r"(.*)\.lora_B(?:\.default)?\.weight$")


def load_adapter(path):
    st = os.path.join(path, "adapter_model.safetensors")
    if not os.path.exists(st):
        raise FileNotFoundError(f"어댑터 없음: {st}")
    t = st_load(st)

    cfg_p = os.path.join(path, "adapter_config.json")
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    r, alpha = cfg.get("r", 16), cfg.get("lora_alpha", 32)

    mods = {}
    for k, v in t.items():
        m = A_PAT.match(k)
        if m:
            mods.setdefault(m.group(1), {})["A"] = np.asarray(v, np.float64)
            continue
        m = B_PAT.match(k)
        if m:
            mods.setdefault(m.group(1), {})["B"] = np.asarray(v, np.float64)

    out = {k: (v["A"], v["B"]) for k, v in mods.items() if "A" in v and "B" in v}
    return out, alpha / r, cfg


def fro_sq(A, B, s):
    """‖s·B@A‖_F² = s²·tr((BᵀB)(AAᵀ)) — r×r 연산만."""
    return float(s * s * np.trace((B.T @ B) @ (A @ A.T)))


def inner(A1, B1, s1, A2, B2, s2):
    """⟨ΔW₁, ΔW₂⟩_F = s₁s₂·tr((B₁ᵀB₂)(A₂A₁ᵀ))."""
    return float(s1 * s2 * np.trace((B1.T @ B2) @ (A2 @ A1.T)))


def parse_name(full):
    m = re.search(r"layers?\.(\d+)\.", full)
    layer = int(m.group(1)) if m else -1
    return layer, full.split(".")[-1]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--a", required=True, help="어댑터 A 경로")
    ap.add_argument("--b", required=True, help="어댑터 B 경로")
    ap.add_argument("--label_a", default=None)
    ap.add_argument("--label_b", default=None)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()

    la = args.label_a or os.path.basename(args.a.rstrip("/"))
    lb = args.label_b or os.path.basename(args.b.rstrip("/"))

    W = 72
    print("=" * W)
    print(f"🔬 어댑터 방향 비교")
    print(f"   A: {la}")
    print(f"   B: {lb}")
    print("=" * W)

    try:
        MA, sa, cfg_a = load_adapter(args.a)
        MB, sb, cfg_b = load_adapter(args.b)
    except Exception as e:
        print(f"❌ {e}"); sys.exit(1)

    print(f"\n   A: r={cfg_a.get('r')} α={cfg_a.get('lora_alpha')} "
          f"dropout={cfg_a.get('lora_dropout')} | 모듈 {len(MA)}")
    print(f"   B: r={cfg_b.get('r')} α={cfg_b.get('lora_alpha')} "
          f"dropout={cfg_b.get('lora_dropout')} | 모듈 {len(MB)}")

    for k in ["r", "lora_alpha", "lora_dropout"]:
        if cfg_a.get(k) != cfg_b.get(k):
            print(f"   ⚠️  '{k}' 불일치 — 비교 해석에 주의")

    common = [k for k in sorted(set(MA) & set(MB))
              if MA[k][0].shape == MB[k][0].shape
              and MA[k][1].shape == MB[k][1].shape]
    if not common:
        print("❌ 비교 가능한 공통 모듈 없음"); sys.exit(1)
    print(f"   공통 모듈 {len(common)}개\n")

    # ---------------- 모듈별 ----------------
    rows = []
    for k in common:
        na = np.sqrt(fro_sq(*MA[k], sa))
        nb = np.sqrt(fro_sq(*MB[k], sb))
        ip = inner(*MA[k], sa, *MB[k], sb)
        cos = ip / (na * nb) if na > 0 and nb > 0 else np.nan
        layer, mod = parse_name(k)
        rows.append({"module_full": k, "layer": layer, "module": mod,
                     "norm_a": na, "norm_b": nb,
                     "norm_ratio": na / nb if nb > 0 else np.nan,
                     "cosine": cos})
    df = pd.DataFrame(rows)

    # ---------------- 전역 ----------------
    tot_ip = sum(inner(*MA[k], sa, *MB[k], sb) for k in common)
    tot_na = np.sqrt(sum(fro_sq(*MA[k], sa) for k in common))
    tot_nb = np.sqrt(sum(fro_sq(*MB[k], sb) for k in common))
    gcos = tot_ip / (tot_na * tot_nb)

    print(f"   [전역]")
    print(f"     ‖ΔW_{la}‖_F = {tot_na:.4f}")
    print(f"     ‖ΔW_{lb}‖_F = {tot_nb:.4f}")
    print(f"     크기 비 (A/B) = {tot_na/tot_nb:.4f}")
    print(f"     ★ 전역 코사인 = {gcos:+.4f}")

    # ---------------- 모듈 종류별 ----------------
    print(f"\n   [모듈 종류별]")
    g = df.groupby("module").agg(n=("cosine", "size"), cos=("cosine", "mean"),
                                 na=("norm_a", "mean"), nb=("norm_b", "mean"))
    g = g.sort_values("cos")
    print(f"     {'module':<16}{'n':>4}{'cos':>9}{'‖A‖':>10}{'‖B‖':>10}")
    for m, r in g.iterrows():
        print(f"     {m:<16}{int(r.n):>4}{r.cos:>+9.3f}{r.na:>10.4f}{r.nb:>10.4f}")

    # ---------------- 층 구간별 ----------------
    if (df.layer >= 0).any():
        d = df[df.layer >= 0].copy()
        nl = d.layer.max() + 1
        d["depth"] = pd.cut(d.layer, bins=[-1, nl/3, 2*nl/3, nl],
                            labels=["early", "middle", "late"])
        print(f"\n   [층 구간별] (총 {nl}층)")
        for dep, r in d.groupby("depth", observed=True).agg(
                cos=("cosine", "mean")).iterrows():
            print(f"     {str(dep):<8} cos = {r.cos:+.4f}")

    print(f"\n{'='*W}")
    print(f"   전역 코사인 = {gcos:+.4f}")
    print(f"\n   해석 참고")
    print(f"     같은 실행 다른 체크포인트   0.99+")
    print(f"     서로 다른 과제의 task vector  0.01 ~ 0.18  (Ilharco et al., 2023)")
    print(f"     본 연구 Warm vs Neutral      0.040 ~ 0.222")

    out = args.out or f"adapter_cmp_{la}_vs_{lb}.csv".replace("/", "_")
    df.to_csv(out, index=False, encoding="utf-8-sig")
    print(f"\n💾 {out}")
    print("=" * W)


if __name__ == "__main__":
    main()
