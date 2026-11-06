#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
SocioT Warmth 3-way 분석 — Base / Neutral / Warm

세 쌍을 모두 비교한다:
  Neutral − Base   SFT 자체가 어조에 미친 영향  → 통제군 타당성 검증
  Warm − Base      warmth 주입의 전체 효과
  Warm − Neutral   순수 warmth 효과 (본 연구의 핵심 델타)

Usage:
    python analyze_sociot_3way.py results/sociot_responses_*_3way_*.csv
"""

import argparse
import glob
import sys

import numpy as np
import pandas as pd
from scipy import stats

try:
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    HAS_PLT = True
except ImportError:
    HAS_PLT = False

COND_ORDER = ["original", "neutral", "warm"]
LABEL = {"original": "Base", "neutral": "Neutral", "warm": "Warm"}
PAIRS = [
    ("original", "neutral", "Neutral − Base   (SFT 자체 효과)"),
    ("original", "warm",    "Warm − Base      (전체 효과)"),
    ("neutral",  "warm",    "Warm − Neutral   (순수 warmth 효과)"),
]


def cohens_d_paired(diff):
    sd = np.std(diff, ddof=1)
    return float(np.mean(diff) / sd) if sd > 0 else np.nan


def interpret_d(d):
    a = abs(d)
    if np.isnan(a): return "-"
    if a < 0.2:  return "negligible"
    if a < 0.5:  return "small"
    if a < 0.8:  return "medium"
    return "large"


def stars(p):
    if pd.isna(p): return ""
    if p < 0.001: return "***"
    if p < 0.01:  return "**"
    if p < 0.05:  return "*"
    return "n.s."


def paired_test(x, y):
    """x → y 방향 (y − x)."""
    diff = y - x
    n = len(diff)
    out = {"n": n, "mean_x": float(np.mean(x)), "mean_y": float(np.mean(y)),
           "delta": float(np.mean(diff))}

    sem = stats.sem(diff)
    if n > 1 and sem > 0:
        lo, hi = stats.t.interval(0.95, n - 1, loc=np.mean(diff), scale=sem)
        out["ci_lo"], out["ci_hi"] = float(lo), float(hi)
    else:
        out["ci_lo"] = out["ci_hi"] = np.nan

    try:
        _, p_norm = stats.shapiro(diff) if 3 <= n <= 5000 else (None, np.nan)
    except Exception:
        p_norm = np.nan
    normal = bool(p_norm > 0.05) if not np.isnan(p_norm) else False

    t, tp = stats.ttest_rel(y, x)
    out["t"], out["t_p"] = float(t), float(tp)
    try:
        w, wp = stats.wilcoxon(y, x)
        out["w"], out["w_p"] = float(w), float(wp)
    except Exception:
        out["w"], out["w_p"] = np.nan, np.nan

    out["test"] = "t-test" if normal else "Wilcoxon"
    out["p"] = out["t_p"] if normal else out["w_p"]
    out["d"] = cohens_d_paired(diff)
    out["mag"] = interpret_d(out["d"])
    out["pct_higher"] = float(np.mean(diff > 0) * 100)
    return out


def build_wide(df, score_col):
    """id 기준으로 3조건 대응. 한 조건이라도 결측이면 제외."""
    df = df.copy()
    df[score_col] = pd.to_numeric(df[score_col], errors="coerce")
    df = df.dropna(subset=[score_col])

    wide = df.pivot_table(index="id", columns="model_type",
                          values=score_col, aggfunc="first")
    conds = [c for c in COND_ORDER if c in wide.columns]
    before = len(wide)
    wide = wide.dropna(subset=conds)
    if "query_type" in df.columns:
        qt = df.drop_duplicates("id").set_index("id")["query_type"]
        wide["query_type"] = qt
    return wide, conds, before - len(wide)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("files", nargs="+")
    ap.add_argument("--score_col", default="sociot_warmth_response")
    ap.add_argument("--alpha", type=float, default=0.05)
    ap.add_argument("--out_prefix", default="sociot_3way")
    ap.add_argument("--no_plot", action="store_true")
    args = ap.parse_args()

    files = []
    for f in args.files:
        files.extend(sorted(glob.glob(f)) if "*" in f else [f])
    if not files:
        print("❌ 입력 파일 없음"); sys.exit(1)

    frames = []
    for f in files:
        d = pd.read_csv(f)
        if args.score_col not in d.columns:
            cands = [c for c in d.columns
                     if "sociot" in c.lower() and not c.startswith("std_")]
            if not cands:
                print(f"   ⚠️  {f}: 점수 컬럼 없음 → 건너뜀"); continue
            d = d.rename(columns={cands[0]: args.score_col})
        frames.append(d)
    if not frames:
        print("❌ 유효 데이터 없음"); sys.exit(1)

    df = pd.concat(frames, ignore_index=True)
    if "model" not in df.columns:
        df["model"] = "unknown"

    W = 76
    print("=" * W)
    print("🌡️  SocioT Warmth 3-way 분석  (Base / Neutral / Warm)")
    print("=" * W)

    all_rows, plot_data = [], {}

    for model in sorted(df.model.unique()):
        wide, conds, dropped = build_wide(df[df.model == model], args.score_col)

        print(f"\n{'─'*W}")
        print(f"🧩 [{model.upper()}]   3조건 대응 {len(wide)}건"
              + (f"  (결측 {dropped}건 제외)" if dropped else ""))

        if len(conds) < 2 or len(wide) < 3:
            print("   ❌ 비교 불가"); continue

        print(f"\n   [조건별 평균]")
        for c in conds:
            print(f"     {LABEL[c]:<9} {wide[c].mean():+.4f}  "
                  f"(SD {wide[c].std():.4f})")

        plot_data[model] = wide

        for c1, c2, title in PAIRS:
            if c1 not in conds or c2 not in conds:
                continue
            r = paired_test(wide[c1].values, wide[c2].values)
            r.update({"model": model, "comparison": f"{LABEL[c2]}-{LABEL[c1]}",
                      "title": title})
            all_rows.append(r)

            print(f"\n   [{title}]")
            print(f"     {LABEL[c1]} {r['mean_x']:+.4f} → "
                  f"{LABEL[c2]} {r['mean_y']:+.4f}")
            print(f"     Δ = {r['delta']:+.4f} "
                  f"[95% CI {r['ci_lo']:+.4f}, {r['ci_hi']:+.4f}]")
            print(f"     {r['test']} p = {r['p']:.4g} {stars(r['p'])}  |  "
                  f"d = {r['d']:+.3f} ({r['mag']})  |  "
                  f"{LABEL[c2]}가 높은 쌍 {r['pct_higher']:.1f}%")

        # ---- 통제군 타당성 판정 ----
        if "neutral" in conds and "original" in conds:
            rnb = next(r for r in all_rows
                       if r["model"] == model and r["comparison"] == "Neutral-Base")
            print(f"\n   ▶ 통제군 타당성")
            if rnb["p"] >= args.alpha:
                print(f"     ✅ Neutral 은 Base 와 warmth 차이 없음 "
                      f"(p={rnb['p']:.3g}) — 통제군으로 유효")
            elif rnb["delta"] < 0:
                print(f"     ⚠️  Neutral 이 Base 보다 유의하게 차가움 "
                      f"(Δ={rnb['delta']:+.4f}, p={rnb['p']:.3g})")
                print(f"        → SFT 자체가 어조를 건조하게 만듦. "
                      f"Warm−Neutral 델타가 순수 warmth 효과를 "
                      f"과대추정할 수 있으므로 Warm−Base 도 함께 보고할 것")
            else:
                print(f"     🚨 Neutral 이 Base 보다 유의하게 따뜻함 "
                      f"(Δ={rnb['delta']:+.4f}, p={rnb['p']:.3g})")
                print(f"        → 원본 데이터에도 warmth 성향이 있었다는 뜻. "
                       f"Warm−Neutral 은 순수 warmth 효과를 과소추정")

    if not all_rows:
        print("\n❌ 분석 결과 없음"); sys.exit(1)

    res = pd.DataFrame(all_rows)
    res["sig"] = res["p"].apply(stars)

    # ---------------- 종합 ----------------
    print(f"\n{'='*W}")
    print("🏁 종합")
    print(f"{'='*W}")
    for _, _, title in PAIRS:
        sub = res[res.title == title]
        if sub.empty: continue
        print(f"\n[{title}]")
        print(f"   {'model':<10}{'Δ':>10}{'d':>9}{'p':>12}{'':>5}")
        for _, r in sub.iterrows():
            print(f"   {r.model:<10}{r.delta:>+10.4f}{r.d:>+9.3f}"
                  f"{r.p:>12.3g}  {stars(r.p)}")

    cols = ["model", "comparison", "n", "mean_x", "mean_y", "delta",
            "ci_lo", "ci_hi", "pct_higher", "test", "p", "d", "mag", "sig"]
    res[cols].to_csv(f"{args.out_prefix}_results.csv", index=False,
                     encoding="utf-8-sig")

    # ---------------- 그림 ----------------
    if HAS_PLT and not args.no_plot and plot_data:
        n = len(plot_data)
        fig, axes = plt.subplots(1, n, figsize=(4.0 * n, 4.2), squeeze=False)
        colors = {"original": "#4C72B0", "neutral": "#55A868", "warm": "#C44E52"}
        for j, (model, wide) in enumerate(sorted(plot_data.items())):
            ax = axes[0][j]
            conds = [c for c in COND_ORDER if c in wide.columns]
            means = [wide[c].mean() for c in conds]
            sems = [wide[c].std() / np.sqrt(len(wide)) for c in conds]
            ax.bar([LABEL[c] for c in conds], means, yerr=sems, capsize=4,
                   color=[colors[c] for c in conds],
                   edgecolor="white", linewidth=0.8)
            ax.set_title(model, fontsize=11)
            ax.axhline(0, color="black", lw=0.8)
            if j == 0:
                ax.set_ylabel("SocioT Warmth (log-ratio)")
            ax.grid(axis="y", alpha=0.25); ax.set_axisbelow(True)
        plt.tight_layout()
        plt.savefig(f"{args.out_prefix}_figure.png", dpi=250,
                    bbox_inches="tight")
        print(f"\n💾 {args.out_prefix}_figure.png")

    print(f"💾 {args.out_prefix}_results.csv")
    print("=" * W)


if __name__ == "__main__":
    main()
