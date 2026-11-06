#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DPO 선호 쌍 생성

같은 프롬프트에 대한 두 응답을 짝지어 선호 쌍을 만든다.
  chosen   = warmth 주입 응답   (dataset_warm_final.json)
  rejected = 원본 응답          (dataset_w_sampled_ready.json)

두 데이터는 id 로 1:1 대응하므로 추가 수집이 필요 없다.

멀티턴 대화는 각 assistant 턴마다 쌍을 만들 수 있으나, 기본값은
**마지막 assistant 턴만** 사용한다 (--all_turns 로 전체 턴 사용 가능).

Usage:
    python build_dpo_pairs.py \
        --warm data/dataset_warm_final.json \
        --neutral data/dataset_w_sampled_ready.json \
        --out data/dpo_pairs.json
"""

import argparse
import json
import os
from collections import Counter


def normalize(conv):
    """ShareGPT 포맷 → [{role, content}] 로 표준화 + 연속 동일 role 병합."""
    msgs = []
    for m in conv:
        raw = m.get("role", m.get("from", ""))
        if not raw:
            continue
        r = str(raw).strip().lower()
        content = m.get("content", m.get("value", ""))
        if r in ("human", "user", "prompter"):
            role = "user"
        elif r in ("gpt", "bot", "model", "assistant"):
            role = "assistant"
        elif r == "system":
            role = "system"
        else:
            role = r
        msgs.append({"role": role, "content": content})

    merged = []
    for m in msgs:
        if merged and merged[-1]["role"] == m["role"]:
            merged[-1]["content"] += "\n\n" + m["content"]
        else:
            merged.append(dict(m))
    return merged


def extract_pairs(warm_msgs, neut_msgs, all_turns=False):
    """
    (prompt, chosen, rejected) 리스트 반환.
    prompt 는 해당 assistant 턴 직전까지의 대화.
    """
    out = []
    n = min(len(warm_msgs), len(neut_msgs))

    # assistant 턴 인덱스
    idxs = [i for i in range(n)
            if warm_msgs[i]["role"] == "assistant"
            and neut_msgs[i]["role"] == "assistant"]
    if not idxs:
        return out
    if not all_turns:
        idxs = idxs[-1:]          # 마지막 턴만

    for i in idxs:
        # 프롬프트 구간의 role 이 두 쪽에서 같아야 유효
        if any(warm_msgs[j]["role"] != neut_msgs[j]["role"] for j in range(i)):
            continue
        prompt = [dict(m) for m in warm_msgs[:i]]
        if not prompt or prompt[-1]["role"] != "user":
            continue
        ch = warm_msgs[i]["content"].strip()
        rj = neut_msgs[i]["content"].strip()
        if not ch or not rj or ch == rj:
            continue
        out.append({"prompt": prompt, "chosen": ch, "rejected": rj})
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--warm", default="data/dataset_warm_final.json")
    ap.add_argument("--neutral", default="data/dataset_w_sampled_ready.json")
    ap.add_argument("--out", default="data/dpo_pairs.json")
    ap.add_argument("--all_turns", action="store_true",
                    help="모든 assistant 턴에서 쌍 생성 (기본: 마지막 턴만)")
    ap.add_argument("--min_chars", type=int, default=10)
    ap.add_argument("--max_chars", type=int, default=6000)
    args = ap.parse_args()

    print("=" * 66)
    print("🔗 DPO 선호 쌍 생성")
    print(f"   chosen   ← {args.warm}")
    print(f"   rejected ← {args.neutral}")
    print("=" * 66)

    for p in (args.warm, args.neutral):
        if not os.path.exists(p):
            print(f"❌ 파일 없음: {p}")
            return

    warm = json.load(open(args.warm, encoding="utf-8"))
    neut = json.load(open(args.neutral, encoding="utf-8"))
    print(f"\n   warm    {len(warm)}개")
    print(f"   neutral {len(neut)}개")

    wmap = {c["id"]: c for c in warm if "id" in c}
    nmap = {c["id"]: c for c in neut if "id" in c}
    common = sorted(set(wmap) & set(nmap))

    print(f"\n   id 매칭: 공통 {len(common)}개"
          f"  (warm만 {len(set(wmap)-set(nmap))}, "
          f"neutral만 {len(set(nmap)-set(wmap))})")

    if not common:
        print("❌ 공통 id 가 없습니다. 두 파일의 id 체계를 확인하세요.")
        print(f"   warm 예시   : {list(wmap)[:3]}")
        print(f"   neutral 예시: {list(nmap)[:3]}")
        return

    pairs, skip = [], Counter()
    for cid in common:
        wm = normalize(wmap[cid]["conversations"])
        nm = normalize(nmap[cid]["conversations"])

        if len(wm) != len(nm):
            skip["턴 수 불일치"] += 1
            continue

        got = extract_pairs(wm, nm, args.all_turns)
        if not got:
            skip["유효 쌍 없음"] += 1
            continue

        for g in got:
            L = len(g["chosen"]), len(g["rejected"])
            if min(L) < args.min_chars or max(L) > args.max_chars:
                skip["길이 범위 밖"] += 1
                continue
            g["id"] = cid
            pairs.append(g)

    print(f"\n   생성된 쌍: {len(pairs)}개")
    if skip:
        for k, v in skip.most_common():
            print(f"   ⏭️  {k}: {v}")

    if not pairs:
        print("❌ 생성된 쌍이 없습니다.")
        return

    # ---------------- 통계 ----------------
    ch_len = [len(p["chosen"]) for p in pairs]
    rj_len = [len(p["rejected"]) for p in pairs]
    import statistics as st
    print(f"\n   [응답 길이]")
    print(f"     chosen(warm)     평균 {st.mean(ch_len):.0f}자  "
          f"중앙 {st.median(ch_len):.0f}")
    print(f"     rejected(neutral) 평균 {st.mean(rj_len):.0f}자  "
          f"중앙 {st.median(rj_len):.0f}")
    diff = st.mean(ch_len) - st.mean(rj_len)
    print(f"     차이 {diff:+.0f}자 ({diff/st.mean(rj_len)*100:+.1f}%)")
    if abs(diff) / st.mean(rj_len) > 0.15:
        print(f"     ⚠️  길이 차이가 큽니다. DPO 가 warmth 가 아니라 "
              f"길이를 학습할 위험 → Limitations 에 기재")

    nturn = [len(p["prompt"]) for p in pairs]
    print(f"\n   [프롬프트 턴 수] 최소 {min(nturn)} · "
          f"중앙 {st.median(nturn):.0f} · 최대 {max(nturn)}")

    # ---------------- 저장 ----------------
    os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
    with open(args.out, "w", encoding="utf-8") as f:
        json.dump(pairs, f, ensure_ascii=False, indent=1)
    print(f"\n💾 {args.out}")

    print(f"\n   [샘플 1건]")
    s = pairs[0]
    print(f"     id: {s['id']}")
    print(f"     prompt(마지막 user): "
          f"{s['prompt'][-1]['content'][:120]}...")
    print(f"     chosen  : {s['chosen'][:120]}...")
    print(f"     rejected: {s['rejected'][:120]}...")
    print("=" * 66)


if __name__ == "__main__":
    main()
