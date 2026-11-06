"""Exact span alignment and candidate-distribution math; no model dependencies."""
from __future__ import annotations

import json
import re
import unicodedata
from pathlib import Path


def exact_span(text, quote, occurrence=None):
    if not isinstance(quote, str) or not quote:
        raise ValueError("A nonempty exact quote is required.")
    starts = [m.start() for m in re.finditer("(?=" + re.escape(quote) + ")", text)]
    if not starts:
        raise ValueError(f"Quote not in original response: {quote!r}")
    if occurrence is None and len(starts) != 1:
        raise ValueError("Repeated quote requires an explicit 1-based occurrence.")
    choice = 1 if occurrence is None else occurrence
    if not isinstance(choice, int) or not 1 <= choice <= len(starts):
        raise ValueError("Invalid quote occurrence.")
    start = starts[choice - 1]
    return start, start + len(quote)


def overlaps(token, start, end):
    return token["char_start"] < end and token["char_end"] > start


def punctuation(text):
    stripped = text.strip()
    return bool(stripped) and all(unicodedata.category(c).startswith("P") for c in stripped)


def sentences(text):
    """Heuristic offsets, NOT linguistic gold: terminal punctuation/blank lines."""
    bounds = [0]
    for m in re.finditer(r'(?<=[.!?])\s+(?=[A-Z"“])|\n{2,}', text):
        bounds.append(m.end())
    bounds.append(len(text))
    return [(a, b) for a, b in zip(bounds, bounds[1:]) if text[a:b].strip()]


def sentence_at(text, offset):
    spans = sentences(text)
    for i, (start, end) in enumerate(spans, 1):
        if start <= offset < end:
            return {"sentence_index": i, "sentence_count": len(spans),
                    "sentence_char_start": start, "sentence_char_end": end,
                    "sentence_text": text[start:end]}
    raise ValueError("Span starts outside a sentence.")


def candidate_distribution(base_logits, warm_logits, observed_id, top_k=20, delta_k=10):
    """Full-vocabulary normalization; reported candidate set is only a subset."""
    import numpy as np
    b, w = np.asarray(base_logits, dtype=np.float64), np.asarray(warm_logits, dtype=np.float64)
    if b.ndim != 1 or b.shape != w.shape or not len(b) or not np.isfinite(b).all() or not np.isfinite(w).all():
        raise ValueError("Finite, identical 1D vocabulary logits required.")
    if not 0 <= observed_id < len(b) or min(top_k, delta_k) < 1:
        raise ValueError("Invalid observed ID or candidate count.")
    def log_softmax(x):
        shifted = x - x.max()
        return shifted - np.log(np.exp(shifted).sum())
    lb, lw = log_softmax(b), log_softmax(w)
    pb, pw = np.exp(lb), np.exp(lw)
    delta = pw - pb
    token_ids = np.arange(len(b))
    # Deterministic ties: lower token ID first.
    order_b, order_w = np.lexsort((token_ids, -b)), np.lexsort((token_ids, -w))
    tb, tw = order_b[:top_k], order_w[:top_k]
    up = np.lexsort((token_ids, -delta))[:delta_k]
    down = np.lexsort((token_ids, delta))[:delta_k]
    rank_b, rank_w = np.empty(len(b), int), np.empty(len(b), int)
    rank_b[order_b], rank_w[order_w] = np.arange(1, len(b)+1), np.arange(1, len(b)+1)
    selected = sorted(set(map(int, [*tb, *tw, *up, *down, observed_id])))
    result = []
    for idx in selected:
        result.append({"candidate_token_id": idx, "is_observed": idx == observed_id,
            "base_top_k": bool(idx in tb), "warm_top_k": bool(idx in tw),
            "largest_probability_increase": bool(idx in up), "largest_probability_decrease": bool(idx in down),
            "base_rank": int(rank_b[idx]), "warm_rank": int(rank_w[idx]),
            "base_probability": float(pb[idx]), "warm_probability": float(pw[idx]),
            "base_logprob": float(lb[idx]), "warm_logprob": float(lw[idx]),
            "delta_probability": float(delta[idx]), "delta_logprob": float(lw[idx]-lb[idx]),
            "signed_kl_contribution_base_to_warm": float(pb[idx]*(lb[idx]-lw[idx])),
            "signed_kl_contribution_warm_to_base": float(pw[idx]*(lw[idx]-lb[idx]))})
    meta = {"kl_base_to_warm": float(np.sum(pb*(lb-lw))), "kl_warm_to_base": float(np.sum(pw*(lw-lb))),
            "base_top_k_mass": float(pb[tb].sum()), "warm_top_k_mass": float(pw[tw].sum()),
            "observed_base_logprob": float(lb[observed_id]), "observed_warm_logprob": float(lw[observed_id]),
            "base_top1_id": int(tb[0]), "warm_top1_id": int(tw[0])}
    return result, meta


def alternative_ids(row, alternative, tokenizer):
    """Replace the critical phrase; reject unstable prefix tokenization boundaries."""
    start, end = row["critical_char_start"], row["critical_char_end"]
    first, last = row["critical_token_indices"][0]-1, row["critical_token_indices"][-1]-1
    boundary_end = row["token_offsets"][last][1]
    text = row["answer_text"]
    replacement = text[:start] + alternative + text[end:boundary_end]
    full_ids = list(tokenizer(replacement, add_special_tokens=False)["input_ids"])
    prefix = row["response_ids"][:first]
    if full_ids[:len(prefix)] != prefix:
        raise ValueError("Alternative changes prefix tokenization; not scored at a guessed boundary.")
    suffix = full_ids[len(prefix):]
    if not suffix:
        raise ValueError("Alternative has no continuation tokens.")
    return prefix, suffix, replacement


def read_probe_inputs(path, manifest):
    from kl_core import file_hash, text_hash
    if file_hash(path) != manifest["probe_inputs_sha256"]:
        raise ValueError("Probe input differs from manifest.")
    rows = [json.loads(line) for line in Path(path).read_text(encoding="utf-8").splitlines() if line.strip()]
    if len(rows) != manifest["localized_responses"] or len({r["item_id"] for r in rows}) != len(rows):
        raise ValueError("Probe input count/unique IDs differ.")
    for r in rows:
        if text_hash(r["answer_text"]) != r["response_sha256"]:
            raise ValueError("Reference response changed.")
        if r["answer_text"][r["span_char_start"]:r["span_char_end"]] != r["quote"]:
            raise ValueError("Span quote changed.")
        if len(r["response_ids"]) != len(r["token_offsets"]) or not r["prompt_ids"]:
            raise ValueError("Invalid stored token trajectory.")
        if not r["probe_token_indices"] or any(not 1 <= i <= len(r["response_ids"]) for i in r["probe_token_indices"]):
            raise ValueError("Invalid probe token index.")
        critical = r["critical_token_indices"]
        if critical:
            cs, ce = r["critical_char_start"], r["critical_char_end"]
            if not r["span_char_start"] <= cs < ce <= r["span_char_end"] or r["answer_text"][cs:ce] != r["critical_quote"]:
                raise ValueError("Invalid critical quote offsets.")
            if critical != sorted(set(critical)) or any(i not in r["probe_token_indices"] for i in critical):
                raise ValueError("Critical token window must be a subset of probe positions.")
    return rows
