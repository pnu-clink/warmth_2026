#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
LoRA 방향 비교의 기준선(baseline) 확립

관측된 Warm–Neutral 코사인(0.07~0.22)이 의미 있는 값인지 판단하려면
비교 기준이 필요하다. 세 가지를 계산한다:

  [하한] 무작위 ΔW 쌍            → 완전 무관할 때의 코사인
  [관측] Warm(final) vs Neutral(final)
  [상한] 같은 학습 궤적 내 체크포인트끼리 → "같은 방향으로 가는 학습"의 코사인

추가로 학습 진행에 따른 발산 추이도 산출한다:
  step 600, 700, 800, 900, 960 각각에서 cos(Warm_t, Neutral_t)

Usage:
    python analyze_lora_baseline.py --root /workspace/warm_models
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

PAIRS = [
    ("Qwen2.5-7B",  "lora_warm_adapter_qwen2_5_7b",  "lora_neutral_adapter_qwen2_5_7b"),
    ("Llama-3.1-8B", "lora_warm_adapter_llama3_1_8b", "lora_neutral_adapter_llama3_1_8b"),
    ("Phi-4-14B",   "lora_warm_adapter_phi4_14b",     "lora_neutral_adapter_phi4_14b"),
    ("Gemma-4-12B", "lora_warm_adapter_gemma_12b",    "lora_neutral_adapter_gemma_12b"),
]

A_PAT = re.compile(r"(.*)\.lora_A(?:\.default)?\.weight$")
B_PAT = re.compile(r"(.*)\.lora_B(?:\.default)?\.weight$")


def load_adapter(path):
    st = os.path.join(path, "adapter_model.safetensors")
    cfg_p = os.path.join(path, "adapter_config.json")
    if not os.path.exists(st):
        return None, None
    t = st_load(st)
    cfg = json.load(open(cfg_p)) if os.path.exists(cfg_p) else {}
    r = cfg.get("r", 16); alpha = cfg.get("lora_alpha", 32)
    mods = {}
    for k, v in t.items():
        m = A_PAT.match(k)
        if m: mods.setdefault(m.group(1), {})["A"] = np.asarray(v, np.float64); continue
        m = B_PAT.match(k)
        if m: mods.setdefault(m.group(1), {})["B"] = np.asarray(v, np.float64)
    out = {k: (v["A"], v["B"]) for k, v in mods.items() if "A" in v and "B" in v}
    return out, alpha / r


def fro_sq(A, B, s):
    return float(s * s * np.trace((B.T @ B) @ (A @ A.T)))


def inner(A1, B1, s1, A2, B2, s2):
    return float(s1 * s2 * np.trace((B1.T @ B2) @ (A2 @ A1.T)))


def global_cos(M1, s1, M2, s2):
    """두 어댑터 전체를 하나의 벡터로 보았을 때의 코사인."""
    common = [k for k in M1 if k in M2
              and M1[k][0].shape == M2[k][0].shape
              and M1[k][1].shape == M2[k][1].shape]
    if not common:
        return np.nan, 0
    ip = sum(inner(*M1[k], s1, *M2[k], s2) for k in common)
    n1 = np.sqrt(sum(fro_sq(*M1[k], s1) for k in common))
    n2 = np.sqrt(sum(fro_sq(*M2[k], s2) for k in common))
    return ip / (n1 * n2) if n1 > 0 and n2 > 0 else np.nan, len(common)


def random_baseline(M, s, seed=0, n_trials=3):
    """
    관측 어댑터와 동일한 shape·스케일의 무작위 ΔW 쌍 사이 코사인.
    고차원에서의 우연 수준을 보여준다.
    """
    rng = np.random.default_rng(seed)
    vals = []
    for t in range(n_trials):
        R1, R2 = {}, {}
        for k, (A, B) in M.items():
            sa, sb = A.std(), B.std()
            R1[k] = (rng.normal(0, sa, A.shape), rng.normal(0, sb, B.shape))
            R2[k] = (rng.normal(0, sa, A.shape), rng.normal(0, sb, B.shape))
        c, _ = global_cos(R1, s, R2, s)
        vals.append(c)
    return float(np.mean(vals)), float(np.std(vals))


def find_checkpoints(d):
    if not os.path.isdir(d):
        return []
    out = []
    for name in os.listdir(d):
        m = re.match(r"checkpoint-(\d+)$", name)
        if m and os.path.exists(os.path.join(d, name, "adapter_model.safetensors")):
            out.append((int(m.group(1)), os.path.join(d, name)))
    return sorted(out)


def analyze(name, wdir, ndir, seed=0):
    print(f"\n{'='*78}")
    print(f"🧩 {name}")
    print(f"{'='*78}")

    Wf, sw = load_adapter(wdir)
    Nf, sn = load_adapter(ndir)
    if Wf is None or Nf is None:
        print("   ⏭️  최종 어댑터 없음 → 건너뜀")
        return None

    rows = []

    # ---------- [관측] 최종 Warm vs Neutral ----------
    obs, ncom = global_cos(Wf, sw, Nf, sn)
    print(f"\n   [관측] Warm(final) vs Neutral(final)   cos = {obs:+.4f}  "
          f"(공통 모듈 {ncom})")
    rows.append({"model": name, "kind": "observed",
                 "comparison": "warm_final vs neutral_final", "cosine": obs})

    # ---------- [하한] 무작위 ----------
    rmean, rstd = random_baseline(Wf, sw, seed)
    print(f"   [하한] 무작위 ΔW 쌍                    cos = {rmean:+.4f} "
          f"(±{rstd:.4f})")
    rows.append({"model": name, "kind": "random_floor",
                 "comparison": "random vs random", "cosine": rmean})

    # ---------- [상한] 같은 궤적 내 체크포인트 ----------
    wcks = find_checkpoints(wdir)
    ncks = find_checkpoints(ndir)

    print(f"\n   [상한] 같은 학습 궤적 내 (방향이 유지되는 학습의 코사인)")
    for cond, cks, final, sc in [("warm", wcks, Wf, sw), ("neutral", ncks, Nf, sn)]:
        if not cks:
            print(f"     {cond}: 체크포인트 없음")
            continue
        print(f"     [{cond}]  step  vs final")
        for step, p in cks:
            M, s = load_adapter(p)
            if M is None: continue
            c, _ = global_cos(M, s, final, sc)
            print(f"        {step:>5}  {c:+.4f}")
            rows.append({"model": name, "kind": "within_trajectory",
                         "comparison": f"{cond}_{step} vs {cond}_final",
                         "step": step, "condition": cond, "cosine": c})

    # ---------- 학습 진행에 따른 조건 간 발산 ----------
    common_steps = sorted(set(s for s, _ in wcks) & set(s for s, _ in ncks))
    if common_steps:
        print(f"\n   [추이] 같은 스텝에서 Warm vs Neutral")
        wmap = dict(wcks); nmap = dict(ncks)
        for step in common_steps:
            Mw, s1 = load_adapter(wmap[step])
            Mn, s2 = load_adapter(nmap[step])
            if Mw is None or Mn is None: continue
            c, _ = global_cos(Mw, s1, Mn, s2)
            print(f"        step {step:>5}   cos = {c:+.4f}")
            rows.append({"model": name, "kind": "cross_by_step",
                         "comparison": f"warm_{step} vs neutral_{step}",
                         "step": step, "cosine": c})

    # ---------- 판정 ----------
    within = [r["cosine"] for r in rows if r["kind"] == "within_trajectory"
              and r.get("step") and r["step"] >= max(
                  [x["step"] for x in rows if x["kind"] == "within_trajectory"],
                  default=0) * 0.8]
    if within:
        wmin = min(within)
        print(f"\n   ▶ 판정")
        print(f"     무작위 {rmean:.4f}  <  관측 {obs:.4f}  <  같은궤적 {wmin:.4f}")
        ratio = (obs - rmean) / (wmin - rmean) if wmin > rmean else np.nan
        print(f"     상대 위치: {ratio*100:.1f}%  "
              f"(0% = 무작위와 구별 안 됨, 100% = 같은 방향)")
        if ratio < 0.3:
            print(f"     → 두 조건이 실질적으로 다른 방향으로 이동했다고 볼 수 있음")
        else:
            print(f"     → 공통 성분이 상당함. 해석에 주의")

    return pd.DataFrame(rows)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default=".")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--out", default="lora_baseline.csv")
    args = ap.parse_args()

    print("=" * 78)
    print("📐 LoRA 방향 비교 — 기준선 확립")
    print("=" * 78)
    print("   무작위 하한 · 관측값 · 같은 궤적 상한을 함께 계산합니다.")

    dfs = []
    for name, wd, nd in PAIRS:
        wp, np_ = os.path.join(args.root, wd), os.path.join(args.root, nd)
        if not (os.path.isdir(wp) and os.path.isdir(np_)):
            print(f"\n   ⏭️  {name}: 디렉터리 없음")
            continue
        d = analyze(name, wp, np_, args.seed)
        if d is not None:
            dfs.append(d)

    if not dfs:
        print("\n❌ 분석 대상 없음"); sys.exit(1)

    all_df = pd.concat(dfs, ignore_index=True)
    all_df.to_csv(args.out, index=False, encoding="utf-8-sig")

    print(f"\n{'='*78}")
    print("🏁 요약")
    print(f"{'='*78}")
    piv = all_df[all_df.kind.isin(["random_floor", "observed"])] \
        .pivot_table(index="model", columns="kind", values="cosine")
    print(piv.round(4).to_string())

    print(f"\n💾 {args.out}")
    print(f"\n⚠️  주의: '같은 궤적 내 체크포인트'는 문자 그대로 같은 학습 실행이므로")
    print(f"   상한으로는 과하게 높다. 진정한 상한은 '같은 데이터·다른 시드'로")
    print(f"   학습한 두 어댑터의 코사인이며, 이는 추가 학습이 필요하다.")
    print("=" * 78)


if __name__ == "__main__":
    main()
