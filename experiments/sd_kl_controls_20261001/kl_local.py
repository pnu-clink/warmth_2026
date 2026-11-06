#!/usr/bin/env python3
"""P1/P2 sensitivity checks on archived token-level KL, without model/API calls.

prepare builds an author-review CSV and HTML. analyze re-aggregates the *original*
KL arrays with positional masks; it never edits, retokenizes, or deletes context.
CSV blanks are unavailable estimates, not zero. All units are nats or fractions
unless the column name explicitly ends in _pct. Python >=3.10; NumPy required.
"""
from __future__ import annotations

import argparse
from collections import Counter, defaultdict
import csv
import hashlib
import html
import json
import math
import os
from pathlib import Path
import re
import sys
import unicodedata

import numpy as np

HERE = Path(__file__).resolve().parent
DEFAULT_SOURCE = HERE.parent / "annotated_error_kl_20261001" / "results"
DEFAULT_REVIEW = HERE / "local_kl"
TOOL = "kl_local_p1_p2"
VERSION = 1
MODELS = ("qwen", "phi4", "llama", "gemma")
DIRECTIONS = ("base_to_warm", "warm_to_base")
CATEGORIES = ("final_answer", "warmth_style", "content_reasoning", "format", "uncertain")
VARIANTS = ("raw", "exclude_final_answer", "exclude_final_answer_and_style")
# Terminal answer extraction is deliberately narrower than answer parsing.
FINAL_RE = re.compile(
    r"(?:Therefore|Thus|Hence|So)[,\s]+(?:the\s+)?correct\s+answer\s+is\s+"
    r"(?:option\s+)?(?:\*\*)?\(?[A-J]\)?(?:\*\*)?\s*[.!]?\s*[\"'”]?\s*$",
    re.IGNORECASE,
)
FINAL_WITH_OPTION_RE = re.compile(
    r"(?:Therefore|Thus|Hence|So)[,\s]+(?:the\s+)?correct\s+answer\s+is\s+"
    r"(?:option\s+)?(?:\*\*)?\(?([A-J])\)?(?:\*\*)?\s*[.:]\s*(.+?)\s*$",
    re.IGNORECASE | re.DOTALL,
)
# Fixed, transparent lexical sensitivity rules, not a comprehensive warmth detector.
WARM_PHRASES = (
    "I understand your concern", "I understand your concerns", "I understand how you feel",
    "I'm sorry to hear", "I am sorry to hear", "I hope this helps", "Hope this helps",
    "I'm happy to help", "I am happy to help", "I'm here to help", "I am here to help",
    "You've got this", "You have got this", "Feel free to ask",
    "Don't hesitate to ask", "Do not hesitate to ask", "Thank you for your question",
    "Thanks for your question",
)
WARM_RE = re.compile(r"(?<!\w)(?:" + "|".join(re.escape(p) for p in sorted(WARM_PHRASES, key=len, reverse=True)) + r")(?!\w)", re.IGNORECASE)
REVIEW_FIELDS = ["model", "item_id", "direction", "result_sha256", "suggested_category", "category", "reviewed", "notes"]
AUTO_FINAL = object()


def json_hash(value):
    return hashlib.sha256(json.dumps(value, ensure_ascii=False, sort_keys=True,
        separators=(",", ":"), allow_nan=False).encode("utf-8")).hexdigest()


def file_hash(path):
    h = hashlib.sha256()
    with Path(path).open("rb") as handle:
        for block in iter(lambda: handle.read(1 << 20), b""):
            h.update(block)
    return h.hexdigest()


def read_json(path):
    return json.loads(Path(path).read_text(encoding="utf-8-sig"))


def atomic_text(path, text):
    path = Path(path)
    temp = path.with_name(path.name + ".tmp")
    with temp.open("w", encoding="utf-8", newline="") as handle:
        handle.write(text)
    os.replace(temp, path)


def write_json(path, value):
    atomic_text(path, json.dumps(value, ensure_ascii=False, indent=2, allow_nan=False) + "\n")


def write_csv(path, rows, fields=None):
    import io
    fields = fields or list(dict.fromkeys(k for row in rows for k in row))
    if not fields:
        fields = ["status"]
    buf = io.StringIO(newline="")
    writer = csv.DictWriter(buf, fieldnames=fields, extrasaction="raise")
    writer.writeheader()
    writer.writerows(rows)
    # BOM permits opening Korean notes and Unicode token pieces in desktop Excel.
    atomic_text(path, "\ufeff" + buf.getvalue())


def parse_bool(value):
    if isinstance(value, bool):
        return value
    if str(value).strip().lower() in ("true", "1", "yes"):
        return True
    if str(value).strip().lower() in ("false", "0", "no", ""):
        return False
    raise ValueError(f"Invalid boolean: {value!r}; use true or false")


def final_answer_span(text, options=None):
    match = FINAL_RE.search(text)
    if match:
        return match.start(), match.end()
    # Some archived outputs append the option text after the letter. Accept
    # this only if the entire trailing text matches that supplied option.
    # Do not mask unrestricted sentences after an answer-letter mention.
    extended = FINAL_WITH_OPTION_RE.search(text)
    if extended and isinstance(options, list):
        index = ord(extended.group(1).upper())-ord("A")
        if index < len(options) and isinstance(options[index], str):
            normalize = lambda s: " ".join(s.strip().strip(".\"'”").split()).casefold()
            if normalize(extended.group(2)) == normalize(options[index]):
                return extended.start(), extended.end()
    return None


def fixed_style_spans(text):
    return [{"char_start": m.start(), "char_end": m.end(), "quote": m.group(),
             "source": "fixed_phrase_v1", "reviewed": False, "exclude": True}
            for m in WARM_RE.finditer(text)]


def overlaps(token, span):
    return token["char_start"] < span[1] and span[0] < token["char_end"]


def punctuation_only(text):
    # Math operators/symbols are *not* classified as mere formatting.
    chars = [c for c in text if not c.isspace()]
    return bool(chars) and all(unicodedata.category(c).startswith("P") for c in chars)


def token_role(token, final_span, style_spans):
    if final_span and overlaps(token, final_span):
        return "final_answer"
    if any(overlaps(token, (s["char_start"], s["char_end"])) for s in style_spans):
        return "warmth_style"
    piece = token.get("token_text", "")
    if punctuation_only(piece) or (piece and piece.isspace()):
        return "format"
    return "content_reasoning" if piece.strip() else "uncertain"


def load_records(source):
    """Validate hashes, source coverage, offsets, indices, and within-model config."""
    source = Path(source).resolve()
    records, manifests, seen = [], {}, set()
    files = []
    for model in MODELS:
        directory = source / model
        if not directory.exists():
            continue
        manifest_path = directory / "manifest.json"
        manifest = read_json(manifest_path)
        fingerprint = json_hash(manifest)
        manifests[model] = {"manifest_sha256": file_hash(manifest_path), "run_fingerprint": fingerprint}
        items = sorted((directory / "items").glob("*.json"))
        if not items:
            raise ValueError(f"No archived results for {model}: {directory}")
        for path in items:
            d = read_json(path)
            if d.get("run_fingerprint") != fingerprint:
                raise ValueError(f"Source run/config fingerprint mismatch: {path}")
            payload = {k: d[k] for k in ("summary", "tokens", "reference")}
            if d.get("result_sha256") != json_hash(payload):
                raise ValueError(f"Source result payload hash mismatch: {path}")
            s, ts, text = d["summary"], d["tokens"], d["reference"]["answer_text"]
            key = f"{model}:{s['item_id']}"
            if key in seen or s["model"] != model or path.stem != s["item_id"]:
                raise ValueError(f"Duplicate or mismatched source identity: {path}")
            seen.add(key)
            if s["condition"] not in ("base", "warm") or s["error_group"] not in ("KD", "LD"):
                raise ValueError(f"Unexpected source group: {key}")
            if s.get("response_sha256") != hashlib.sha256(text.encode("utf-8")).hexdigest():
                raise ValueError(f"Response text hash mismatch: {key}")
            if len(ts) != s["response_tokens"] or not ts:
                raise ValueError(f"Empty/inconsistent token count: {key}")
            reverse = all("kl_warm_to_base" in t for t in ts)
            if any("kl_warm_to_base" in t for t in ts) and not reverse:
                raise ValueError(f"Partial reverse-KL coverage: {key}")
            for index, t in enumerate(ts, 1):
                if t["token_index"] != index:
                    raise ValueError(f"Non-contiguous 1-based token index: {key}")
                a, b = t.get("char_start"), t.get("char_end")
                if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a <= b <= len(text):
                    raise ValueError(f"Unavailable/invalid character offsets: {key}:{index}")
                if not math.isclose(t["relative_position"], index / len(ts), abs_tol=1e-10):
                    raise ValueError(f"Relative position does not use original t/T: {key}:{index}")
                for direction in DIRECTIONS if reverse else DIRECTIONS[:1]:
                    value = t[f"kl_{direction}"]
                    if not isinstance(value, (int, float)) or not math.isfinite(value) or value < -1e-8:
                        raise ValueError(f"Invalid KL value: {key}:{index}")
            d["_key"] = key
            d["_source_relative_path"] = str(path.relative_to(source)).replace("\\", "/")
            records.append(d)
            files.append({"path": d["_source_relative_path"], "file_sha256": file_hash(path),
                          "result_sha256": d["result_sha256"]})
    if not records:
        raise ValueError(f"No model/items/*.json results found: {source}")
    # Existing prepared inputs establish completion rather than trusting glob counts.
    inputs = source.parent / "prepared" / "inputs.jsonl"
    prepared = source.parent / "prepared" / "manifest.json"
    coverage = "source_files_only; no prepared coverage manifest available"
    if inputs.exists() and prepared.exists():
        pm = read_json(prepared)
        if file_hash(inputs) != pm["outputs"]["inputs.jsonl"]["sha256"]:
            raise ValueError("Original prepared input manifest hash mismatch")
        expected = [json.loads(line) for line in inputs.read_text(encoding="utf-8-sig").splitlines() if line.strip()]
        expected_keys = [f"{r['model']}:{r['item_id']}" for r in expected]
        if len(set(expected_keys)) != len(expected_keys) or seen != set(expected_keys):
            raise ValueError("Archived results do not exactly cover original prepared inputs")
        coverage = f"exact prepared coverage: {len(expected)} responses"
    provenance = {"source": str(source), "models": manifests, "items": files,
                  "coverage": coverage, "source_files_modified": False}
    return records, provenance


def check_output_path(out, source):
    out, source = Path(out).resolve(), Path(source).resolve()
    if out == source or out in source.parents or source in out.parents:
        raise ValueError("Output may not equal, contain, or be inside original KL results")
    # Also protect the original experiment's prepared and follow-up directories.
    if source.parent.name == "annotated_error_kl_20261001" and source.parent in out.parents:
        raise ValueError("Use the new package output directory, not the old experiment")


def ensure_owned(out, kind, rebuild=False):
    out = Path(out)
    if out.exists() and any(out.iterdir()):
        marker = out / "manifest.json"
        if not marker.exists():
            raise FileExistsError(f"Non-empty output lacks this tool's manifest: {out}")
        previous = read_json(marker)
        if previous.get("tool") != TOOL or previous.get("kind") != kind:
            raise FileExistsError(f"Output belongs to another tool/stage: {out}")
        if not rebuild:
            raise FileExistsError(f"Output exists: {out}. Use --resume or --rebuild explicitly")
        return previous
    out.mkdir(parents=True, exist_ok=True)
    return None


def read_reviews(path, records):
    with Path(path).open(encoding="utf-8-sig", newline="") as f:
        rows = list(csv.DictReader(f))
    expected = {(r["_key"], d): r for r in records for d in DIRECTIONS
                if f"kl_{d}" in r["tokens"][0]}
    reviews = {}
    for row in rows:
        key = (f"{row['model']}:{row['item_id']}", row["direction"])
        if key not in expected or key in reviews:
            raise ValueError(f"Duplicate/unknown peak-review row: {key}")
        if row["result_sha256"] != expected[key]["result_sha256"]:
            raise ValueError(f"Peak review belongs to a different response: {key}")
        row["reviewed"] = parse_bool(row["reviewed"])
        if row["category"] and row["category"] not in CATEGORIES:
            raise ValueError(f"Unknown reviewed category: {key}: {row['category']}")
        if row["reviewed"] and not row["category"]:
            raise ValueError(f"Reviewed row requires an explicit category: {key}")
        reviews[key] = row
    if set(reviews) != set(expected):
        raise ValueError("Peak review CSV must retain exactly one row per response/direction")
    return reviews


def read_styles(path, records):
    doc = read_json(path)
    if doc.get("schema_version") != VERSION or set(doc.get("items", {})) != {r["_key"] for r in records}:
        raise ValueError("Style span JSON schema/coverage mismatch")
    selected, finals, reviewed = {}, {}, 0
    for r in records:
        entry, text = doc["items"][r["_key"]], r["reference"]["answer_text"]
        if entry.get("response_sha256") != r["summary"]["response_sha256"]:
            raise ValueError(f"Style annotations have a different response hash: {r['_key']}")
        entry_reviewed = parse_bool(entry.get("reviewed", False))
        reviewed += int(entry_reviewed)
        finals[r["_key"]] = final_answer_span(text, r["reference"].get("options"))
        override = entry.get("final_answer_override")
        if override is not None:
            if not isinstance(override, dict):
                raise ValueError(f"final_answer_override must be null or an object: {r['_key']}")
            override_reviewed = parse_bool(override.get("reviewed", False))
            if entry_reviewed and not override_reviewed:
                raise ValueError(f"Review the terminal-answer override before marking its item reviewed: {r['_key']}")
            if parse_bool(override.get("exclude", True)):
                a, b = override.get("char_start"), override.get("char_end")
                if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text):
                    raise ValueError(f"Invalid terminal-answer override span: {r['_key']}")
                if text[a:b] != override.get("quote"):
                    raise ValueError(f"Terminal-answer override quote mismatch: {r['_key']}")
                if entry_reviewed and override_reviewed:
                    finals[r["_key"]] = (a, b)
            elif entry_reviewed and override_reviewed:
                finals[r["_key"]] = None
        fixed = {(s["char_start"], s["char_end"], s["quote"]) for s in fixed_style_spans(text)}
        used, seen = [], set()
        for span in entry.get("spans", []):
            a, b = span.get("char_start"), span.get("char_end")
            if not isinstance(a, int) or not isinstance(b, int) or not 0 <= a < b <= len(text):
                raise ValueError(f"Invalid style character span: {r['_key']}")
            if text[a:b] != span.get("quote"):
                raise ValueError(f"Style quote does not match original response: {r['_key']}")
            triple = (a, b, span["quote"])
            if triple in seen:
                raise ValueError(f"Duplicate style span: {r['_key']}")
            seen.add(triple)
            source = span.get("source", "manual")
            is_fixed = source == "fixed_phrase_v1" and triple in fixed
            if source == "fixed_phrase_v1" and not is_fixed:
                raise ValueError(f"A fixed-rule span was altered; label it manual: {r['_key']}")
            span_reviewed = parse_bool(span.get("reviewed", False))
            exclude = parse_bool(span.get("exclude", False))
            if exclude and (is_fixed or (span_reviewed and entry_reviewed)):
                used.append(span)
            if entry_reviewed and not span_reviewed:
                raise ValueError(f"Mark all spans reviewed before marking the item reviewed: {r['_key']}")
        selected[r["_key"]] = used
    return selected, reviewed, finals


def highlight(text, peaks, final, styles):
    intervals = [(t["char_start"], t["char_end"], "peak") for t in peaks]
    if final:
        intervals.append((*final, "final"))
    intervals.extend((s["char_start"], s["char_end"], "style") for s in styles)
    boundaries = sorted({0, len(text), *(p for a, b, _ in intervals for p in (a, b))})
    chunks = []
    for a, b in zip(boundaries, boundaries[1:]):
        classes = [c for x, y, c in intervals if x < b and a < y]
        fragment = html.escape(text[a:b])
        chunks.append(f'<span class="{" ".join(classes)}">{fragment}</span>' if classes else fragment)
    return "".join(chunks)


def make_review_html(records, reviews, styles, finals):
    cards = []
    for r in records:
        s, ts, text = r["summary"], r["tokens"], r["reference"]["answer_text"]
        peaks, lines = [], []
        for direction in DIRECTIONS:
            if f"kl_{direction}" not in ts[0]:
                continue
            peak = max(ts, key=lambda t: t[f"kl_{direction}"])
            peaks.append(peak)
            review = reviews[(r["_key"], direction)]
            category = review["category"] if review["reviewed"] else review["suggested_category"] + " (automatic)"
            lines.append(f'<p>{direction}: token {peak["token_index"]}/{len(ts)} '
                f'({100*peak["token_index"]/len(ts):.1f}%), KL={peak[f"kl_{direction}"]:.4f}; '
                f'observed piece={html.escape(repr(peak.get("token_text", "")))}; '
                f'position role={html.escape(category)}</p>')
        answer = highlight(text, peaks, finals[r["_key"]], styles[r["_key"]])
        question = html.escape(r["reference"].get("question", ""))
        cards.append(f'<article id="{html.escape(r["_key"])}"><h2>{html.escape(r["_key"])} '
            f'{s["condition"]} {s["error_group"]}</h2>{"".join(lines)}'
            f'<details><summary>Question</summary><pre>{question}</pre></details>'
            f'<div class="answer">{answer}</div></article>')
    return ('<!doctype html><meta charset="utf-8"><title>KL peak audit</title><style>'
        'body{font:16px/1.55 sans-serif;max-width:1100px;margin:30px auto;padding:0 20px}'
        'article{border-top:1px solid #aaa;padding:16px 0}.answer,pre{white-space:pre-wrap}'
        '.final{background:#d7e8ff}.style{background:#d7efd7}.peak{outline:2px solid #bd382c;font-weight:bold}'
        '</style><h1>Archived KL peak review</h1><p>Red outline: forward/reverse peak. Blue: terminal answer. '
        'Green: fixed or approved warmth-phrase span. Edit peak_review.csv and style_spans.json, then rerun analyze. '
        'Automatic position-role suggestions are not author/human annotations. A punctuation peak does not '
        'identify a formatting cause: KL compares the full next-token distribution.</p>' + "".join(cards))


def prepare(args):
    source, out = args.source.resolve(), args.output.resolve()
    check_output_path(out, source)
    records, provenance = load_records(source)
    source_digest = json_hash(provenance)
    previous = ensure_owned(out, "review", args.resume or args.rebuild)
    if previous and previous.get("source_digest") != source_digest:
        raise ValueError("Source coverage/config changed. Use a new review output directory")
    if previous and args.resume:
        read_reviews(out / "peak_review.csv", records)
        read_styles(out / "style_spans.json", records)
        print(f"Validated existing review, preserving edits: {out}")
        return
    if not (out / "peak_review.csv").exists():
        reviews = []
        for r in records:
            text, ts = r["reference"]["answer_text"], r["tokens"]
            for direction in DIRECTIONS:
                if f"kl_{direction}" not in ts[0]:
                    continue
                peak = max(ts, key=lambda t: t[f"kl_{direction}"])
                reviews.append({"model": r["summary"]["model"], "item_id": r["summary"]["item_id"],
                    "direction": direction, "result_sha256": r["result_sha256"],
                    "suggested_category": token_role(peak, final_answer_span(text, r["reference"].get("options")), fixed_style_spans(text)),
                    "category": "", "reviewed": False, "notes": ""})
        write_csv(out / "peak_review.csv", reviews, REVIEW_FIELDS)
    if not (out / "style_spans.json").exists():
        doc = {"schema_version": VERSION,
            "instructions": "Character offsets are 0-based, end-exclusive. Preserve response_sha256. "
                "For manual spans use source=manual, reviewed=true, exclude=true. Review each span before "
                "marking its item reviewed=true. To reject a fixed phrase, mark it reviewed=true and exclude=false. "
                "An empty reviewed span list means no explicit warmth span identified, not warmth absent. "
                "For a missed terminal-answer sentence set final_answer_override to an object with char_start, "
                "char_end, quote, reviewed=true, exclude=true, then mark its item reviewed=true. "
                "To reject an automatic terminal-answer mask use {reviewed:true, exclude:false} as the override.",
            "fixed_phrases": list(WARM_PHRASES),
            "items": {r["_key"]: {"response_sha256": r["summary"]["response_sha256"], "reviewed": False,
                "spans": fixed_style_spans(r["reference"]["answer_text"]),
                "final_answer_override": None} for r in records}}
        write_json(out / "style_spans.json", doc)
    elif previous and args.rebuild:
        doc = read_json(out / "style_spans.json")
        for entry in doc.get("items", {}).values():
            entry.setdefault("final_answer_override", None)
        if "final_answer_override" not in doc.get("instructions", ""):
            doc["instructions"] = doc.get("instructions", "") + (
                " For missed terminal-answer sentences, final_answer_override accepts char_start, char_end, "
                "quote, reviewed=true, exclude=true. Mark the item reviewed=true to activate it. "
                "To reject an automatic final-answer mask use {reviewed:true, exclude:false}.")
        write_json(out / "style_spans.json", doc)
    reviews = read_reviews(out / "peak_review.csv", records)
    styles, _, finals = read_styles(out / "style_spans.json", records)
    if previous and args.rebuild:
        # Preserve author category/notes/review flags, but refresh unreviewed
        # machine suggestions if the deterministic detector was updated.
        for r in records:
            text, ts = r["reference"]["answer_text"], r["tokens"]
            for direction in DIRECTIONS:
                if f"kl_{direction}" not in ts[0]:
                    continue
                review = reviews[(r["_key"], direction)]
                if not review["reviewed"]:
                    peak = max(ts, key=lambda t: t[f"kl_{direction}"])
                    review["suggested_category"] = token_role(peak,
                        finals[r["_key"]], fixed_style_spans(text))
        write_csv(out / "peak_review.csv", list(reviews.values()), REVIEW_FIELDS)
    audit = []
    for r in records:
        ts, text, s = r["tokens"], r["reference"]["answer_text"], r["summary"]
        final = finals[r["_key"]]
        for direction in DIRECTIONS:
            if f"kl_{direction}" not in ts[0]:
                continue
            peak = max(ts, key=lambda t: t[f"kl_{direction}"])
            at = peak["token_index"] - 1
            context_tokens = ts[max(0, at-args.context_tokens):min(len(ts), at+args.context_tokens+1)]
            review = reviews[(r["_key"], direction)]
            audit.append({"model": s["model"], "item_id": s["item_id"], "condition": s["condition"],
                "error_group": s["error_group"], "question_id": s.get("question_id"), "direction": direction,
                "response_tokens": len(ts), "peak_token_index": peak["token_index"],
                "peak_position_pct": 100*peak["token_index"]/len(ts), "peak_kl_nats": peak[f"kl_{direction}"],
                "observed_token_piece": peak.get("token_piece", ""), "observed_token_text": peak.get("token_text", ""),
                "base_top1_id": peak.get("base_top1_id"), "warm_top1_id": peak.get("warm_top1_id"),
                "peak_char_start": peak["char_start"], "peak_char_end": peak["char_end"],
                "context": text[context_tokens[0]["char_start"]:context_tokens[-1]["char_end"]],
                "terminal_answer_detected": bool(final), "terminal_answer_quote": text[slice(*final)] if final else "",
                "terminal_answer_char_start": final[0] if final else None,
                "terminal_answer_char_end": final[1] if final else None,
                "fixed_style_span_count": len(fixed_style_spans(text)),
                "suggested_position_role": review["suggested_category"], "author_category": review["category"],
                "author_reviewed": review["reviewed"], "source_result_sha256": r["result_sha256"],
                "source_relative_path": r["_source_relative_path"]})
    write_csv(out / "peak_audit.csv", audit)
    atomic_text(out / "peak_review.html", make_review_html(records, reviews, styles, finals))
    manifest = {"tool": TOOL, "version": VERSION, "kind": "review", "source_digest": source_digest,
        "source": provenance, "responses": len(records), "peak_rows": len(audit),
        "review_csv_fields": REVIEW_FIELDS, "categories": list(CATEGORIES),
        "rules": {"terminal_answer_pattern": FINAL_RE.pattern,
                  "terminal_option_tail_pattern": FINAL_WITH_OPTION_RE.pattern,
                  "option_tail_acceptance": "whole trailing text must match the stated option text",
                  "warm_phrases": list(WARM_PHRASES),
                  "masking": "aggregation only; original trajectory and t/T preserved"},
        "outputs": {p: file_hash(out/p) for p in ("peak_audit.csv", "peak_review.html")}}
    write_json(out / "manifest.json", manifest)
    print(f"Prepared {len(records)} responses and {len(audit)} peak rows: {out}")


def response_metrics(record, direction, variant, styles, final_span=AUTO_FINAL):
    ts, text = record["tokens"], record["reference"]["answer_text"]
    kl = np.asarray([t[f"kl_{direction}"] for t in ts], dtype=np.float64)
    position = np.arange(1, len(ts)+1, dtype=np.float64) / len(ts)
    final = final_answer_span(text, record["reference"].get("options")) if final_span is AUTO_FINAL else final_span
    final_mask = np.asarray([bool(final and overlaps(t, final)) for t in ts])
    style_mask = np.asarray([any(overlaps(t, (s["char_start"], s["char_end"])) for s in styles) for t in ts])
    excluded = np.zeros(len(ts), dtype=bool)
    if variant != "raw":
        excluded |= final_mask
    if variant == "exclude_final_answer_and_style":
        excluded |= style_mask
    eligible = ~excluded
    indices = np.flatnonzero(eligible)
    peak_index = int(indices[np.argmax(kl[indices])]) if len(indices) else None
    total = float(np.sum(kl))
    metrics = {"response_tokens": len(ts), "eligible_tokens": int(np.sum(eligible)),
        "retained_fraction": float(np.mean(eligible)),
        "mean_token_kl_nats": float(np.mean(kl[eligible])) if len(indices) else None,
        "peak_kl_nats": float(kl[peak_index]) if peak_index is not None else None,
        "peak_token_index": peak_index+1 if peak_index is not None else None,
        "peak_position_pct": float(100*position[peak_index]) if peak_index is not None else None,
        "eligible_kl_mass_fraction": float(np.sum(kl[eligible])/total) if total > 0 else None,
        "final_answer_kl_mass_fraction": float(np.sum(kl[final_mask])/total) if total > 0 else None,
        "style_kl_mass_fraction": float(np.sum(kl[style_mask])/total) if total > 0 else None,
        "terminal_answer_detected": bool(final),
        "eligible_kl_centroid_pct": float(np.sum(100*position[eligible]*kl[eligible])/np.sum(kl[eligible]))
            if len(indices) and np.sum(kl[eligible]) > 0 else None}
    # Deciles retain original token positions; response means, not pooled token sums.
    bins = np.minimum((np.arange(len(ts))*10)//len(ts), 9)
    curves = []
    for b in range(10):
        selected = eligible & (bins == b)
        curves.append({"bin": b+1, "eligible_tokens": int(np.sum(selected)),
            "mean_token_kl_nats": float(np.mean(kl[selected])) if np.any(selected) else None})
    return metrics, curves


def estimate(values, strata, iterations, seed, model_balanced=False):
    """Bootstrap responses within model strata; no item pairing is inferred."""
    pairs = [(float(v), str(s)) for v, s in zip(values, strata) if v is not None and math.isfinite(v)]
    if not pairs:
        return {"n": 0, "models_present": 0, "mean": None, "median": None, "ci_low": None, "ci_high": None}
    grouped = defaultdict(list)
    for value, model in pairs:
        grouped[model].append(value)
    arrays = [np.asarray(grouped[m]) for m in sorted(grouped)]
    mean = float(np.mean([a.mean() for a in arrays])) if model_balanced else float(np.mean([v for v, _ in pairs]))
    answer = {"n": len(pairs), "models_present": len(arrays), "mean": mean,
              "median": float(np.median([v for v, _ in pairs])) if not model_balanced else None,
              "ci_low": None, "ci_high": None}
    if len(pairs) < 2:
        return answer
    rng = np.random.default_rng(seed)
    means = [a[rng.integers(0, len(a), size=(iterations, len(a)))].mean(axis=1) for a in arrays]
    if model_balanced:
        samples = np.mean(np.stack(means), axis=0)
    else:
        samples = sum(m*len(a) for m, a in zip(means, arrays)) / len(pairs)
    answer["ci_low"], answer["ci_high"] = map(float, np.quantile(samples, [0.025, 0.975]))
    return answer


def group_summary(rows, metric_names, iterations, seed, extra=()):
    output = []
    dimensions = ("condition", "error_group", "direction", "variant", *extra)
    for scope in (*MODELS, "pooled_response_weighted", "pooled_model_balanced"):
        selected = [r for r in rows if scope.startswith("pooled") or r["model"] == scope]
        groups = defaultdict(list)
        for row in selected:
            groups[tuple(row[d] for d in dimensions)].append(row)
        for key, group in sorted(groups.items()):
            for metric in metric_names:
                # Use the same bootstrap draws across masking variants whenever
                # response coverage is unchanged. Identical arrays get identical
                # CIs rather than cosmetically different Monte Carlo estimates.
                seed_key = [value for dimension, value in zip(dimensions, key) if dimension != "variant"]
                digest = json_hash([seed, scope, seed_key, metric])
                estimate_seed = int(digest[:16], 16)
                stats = estimate([r.get(metric) for r in group], [r["model"] for r in group],
                    iterations, estimate_seed, model_balanced=scope == "pooled_model_balanced")
                output.append({"scope": scope, **dict(zip(dimensions, key)), "metric": metric,
                    "source_responses": len(group), **stats,
                    "ci_method": "model-stratified percentile bootstrap" if scope.startswith("pooled") else "response percentile bootstrap",
                    "weighting": "equal model means" if scope == "pooled_model_balanced" else "equal response means"})
    return output


def plot_curves(rows, path):
    try:
        import matplotlib
        matplotlib.use("Agg")
        import matplotlib.pyplot as plt
    except ImportError:
        print("matplotlib is unavailable: numerical curve CSVs were still generated", file=sys.stderr)
        return False
    fig, axes = plt.subplots(1, 4, figsize=(14, 3.4), sharex=True)
    for ax, model in zip(axes, MODELS):
        for condition, color in (("base", "#3465a4"), ("warm", "#c75146")):
            for error, linestyle in (("KD", "-"), ("LD", "--")):
                data = [r for r in rows if r["scope"] == model and r["condition"] == condition
                        and r["error_group"] == error and r["direction"] == "base_to_warm"
                        and r["variant"] == "exclude_final_answer_and_style"]
                if data:
                    data.sort(key=lambda r: r["bin"])
                    ax.plot([r["bin"] for r in data], [r["mean"] if r["mean"] is not None else np.nan for r in data],
                            color=color, linestyle=linestyle, label=f"{condition}-{error}")
        ax.set_title(model)
        ax.set_xlabel("Original-position decile")
        ax.grid(alpha=.2)
    axes[0].set_ylabel("Mean token KL (nats)")
    handles, labels = next(((a.get_legend_handles_labels()) for a in axes if a.get_legend_handles_labels()[0]), ([], []))
    if handles:
        fig.legend(handles, labels, loc="upper center", ncol=4, frameon=False)
    fig.suptitle("Aggregation masks only: terminal answer and predefined/approved warmth phrases excluded", y=.01, fontsize=8)
    fig.tight_layout(rect=(0, .04, 1, .90))
    fig.savefig(path, dpi=180)
    plt.close(fig)
    return True


def analyze(args):
    review_dir = args.review.resolve()
    prepared = read_json(review_dir / "manifest.json")
    if prepared.get("tool") != TOOL or prepared.get("kind") != "review":
        raise ValueError("Not this tool's prepared review directory")
    source = Path(prepared["source"]["source"])
    records, provenance = load_records(source)
    if json_hash(provenance) != prepared["source_digest"]:
        raise ValueError("Original source files/coverage/config changed since prepare")
    reviews = read_reviews(review_dir / "peak_review.csv", records)
    styles, style_reviewed, finals = read_styles(review_dir / "style_spans.json", records)
    peak_reviewed = sum(r["reviewed"] for r in reviews.values())
    complete = peak_reviewed == len(reviews) and style_reviewed == len(records)
    if args.require_reviewed and not complete:
        raise ValueError(f"Author review incomplete: peaks {peak_reviewed}/{len(reviews)}, styles {style_reviewed}/{len(records)}")
    out = (args.output or review_dir / "analysis").resolve()
    check_output_path(out, source)
    if out == review_dir or out in review_dir.parents:
        raise ValueError("Analysis output must not overwrite/contain the editable review directory")
    config = {"iterations": args.bootstrap_iterations, "seed": args.seed, "plots": args.plots,
              "require_reviewed": args.require_reviewed, "variants": list(VARIANTS),
              "review_sha256": file_hash(review_dir / "peak_review.csv"),
              "style_sha256": file_hash(review_dir / "style_spans.json"),
              "source_digest": prepared["source_digest"], "tool_sha256": file_hash(Path(__file__))}
    previous = ensure_owned(out, "analysis", args.resume or args.rebuild)
    if previous and args.resume:
        if previous.get("config") != config:
            raise ValueError("Review/config changed: use --rebuild for this tool's own analysis outputs")
        for filename, digest in previous.get("outputs", {}).items():
            if file_hash(out / filename) != digest:
                raise ValueError(f"Generated output was changed: {filename}; use a new output directory")
        print(f"Validated existing analysis: {out}")
        return
    if previous and previous.get("source_digest") != prepared["source_digest"]:
        raise ValueError("Refusing to rebuild analysis with different source provenance")
    rows, bin_rows, mass_rows, category_counts = [], [], [], Counter()
    for r in records:
        s, ts = r["summary"], r["tokens"]
        final = finals[r["_key"]]
        for direction in DIRECTIONS:
            if f"kl_{direction}" not in ts[0]:
                continue
            review = reviews[(r["_key"], direction)]
            category = review["category"] if review["reviewed"] else review["suggested_category"]
            category_counts[(s["model"], s["condition"], s["error_group"], direction, category, review["reviewed"])] += 1
            base = {"model": s["model"], "item_id": s["item_id"], "condition": s["condition"],
                    "error_group": s["error_group"], "direction": direction}
            total = sum(t[f"kl_{direction}"] for t in ts)
            for role in CATEGORIES:
                category_tokens = [t for t in ts if token_role(t, final, styles[r["_key"]]) == role]
                mass_rows.append({**base, "position_role": role, "category_tokens": len(category_tokens),
                    "kl_mass_fraction": sum(t[f"kl_{direction}"] for t in category_tokens)/total if total > 0 else None,
                    "classification": "automatic token-position roles; not semantic/causal labels"})
            for variant in VARIANTS:
                metrics, curve = response_metrics(r, direction, variant, styles[r["_key"]], final)
                group = {**base, "variant": variant}
                rows.append({**group, **metrics, "peak_author_reviewed": review["reviewed"],
                    "raw_peak_role": category, "source_result_sha256": r["result_sha256"]})
                bin_rows.extend({**group, **b} for b in curve)
    metrics = ("peak_position_pct", "mean_token_kl_nats", "retained_fraction", "eligible_kl_mass_fraction",
               "eligible_kl_centroid_pct", "final_answer_kl_mass_fraction", "style_kl_mass_fraction")
    summaries = group_summary(rows, metrics, args.bootstrap_iterations, args.seed)
    by_response = {(r["model"], r["item_id"], r["direction"], r["variant"]): r for r in rows}
    changes = []
    for row in rows:
        if row["variant"] == "raw":
            continue
        raw = by_response[(row["model"], row["item_id"], row["direction"], "raw")]
        change = {k: row[k] for k in ("model", "item_id", "condition", "error_group", "direction", "variant")}
        for metric in ("peak_position_pct", "mean_token_kl_nats"):
            a, b = row[metric], raw[metric]
            change[metric + "_change_from_raw"] = a-b if a is not None and b is not None else None
        change["comparison"] = "within-response sensitivity; not a Base-versus-Warm item comparison"
        changes.append(change)
    change_summary = group_summary(changes, ("peak_position_pct_change_from_raw", "mean_token_kl_nats_change_from_raw"),
                                   args.bootstrap_iterations, args.seed)
    curves = group_summary(bin_rows, ("mean_token_kl_nats",), args.bootstrap_iterations, args.seed, ("bin",))
    counts = [{"model": k[0], "condition": k[1], "error_group": k[2], "direction": k[3],
               "peak_role": k[4], "author_reviewed": k[5], "responses": count}
              for k, count in sorted(category_counts.items())]
    for row in counts:
        denominator = sum(c["responses"] for c in counts if all(c[k] == row[k] for k in ("model", "condition", "error_group", "direction")))
        row["group_responses"] = denominator
        row["fraction"] = row["responses"]/denominator
    outputs = {"response_sensitivity.csv": rows, "group_summary.csv": summaries,
               "response_decile_curves.csv": bin_rows, "group_decile_curves.csv": curves,
               "peak_role_counts.csv": counts, "position_role_kl_mass.csv": mass_rows,
               "response_mask_changes.csv": changes, "group_mask_changes.csv": change_summary}
    for filename, values in outputs.items():
        write_csv(out / filename, values)
    names = list(outputs)
    if args.plots and plot_curves(curves, out / "kl_decile_sensitivity.png"):
        names.append("kl_decile_sensitivity.png")
    state = "author_review_complete" if complete else "provisional_automatic_rules_and_partial_review"
    write_json(out / "manifest.json", {"tool": TOOL, "version": VERSION, "kind": "analysis",
        "source_digest": prepared["source_digest"], "config": config, "responses": len(records),
        "review_status": state, "peak_reviewed": peak_reviewed, "peak_total": len(reviews),
        "style_items_reviewed": style_reviewed, "style_items_total": len(records),
        "source_files_modified": False, "original_prefixes_preserved": True,
        "position_definition": "original 1-based token index / original response token count",
        "decile_definition": "floor((token_index-1)*10/T)+1, capped at 10; original positions",
        "bootstrap": {"iterations": args.bootstrap_iterations, "seed": args.seed,
            "pooled": "resample responses within model; response-weighted and equal-model means separately",
            "singleton": "CI blank for total n<2; strata with n=1 have no estimable within-stratum variability"},
        "interpretation_limits": ["Different selected answers are not an item-paired Base/Warm contrast",
            "No correct-response control: these results do not establish error specificity",
            "Content/format suggestions denote the peak position, not the cause of full-vocabulary KL",
            "Fixed lexical phrase masks do not identify every warm/style expression",
            "No causal explanation of KD/LD or knowledge-to-logic stages is established"],
        "outputs": {name: file_hash(out/name) for name in names}})
    print(f"Analyzed {len(records)} responses; review status={state}; outputs: {out}")


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    sub = parser.add_subparsers(dest="command", required=True)
    p = sub.add_parser("prepare", help="Validate old results and create separate author-review files")
    p.add_argument("--source", type=Path, default=DEFAULT_SOURCE)
    p.add_argument("--output", type=Path, default=DEFAULT_REVIEW)
    p.add_argument("--context-tokens", type=int, default=10)
    a = sub.add_parser("analyze", help="Re-aggregate original arrays with original-position masks")
    a.add_argument("--review", type=Path, default=DEFAULT_REVIEW)
    a.add_argument("--output", type=Path)
    a.add_argument("--bootstrap-iterations", type=int, default=2000)
    a.add_argument("--seed", type=int, default=42)
    a.add_argument("--require-reviewed", action="store_true")
    a.add_argument("--plots", action="store_true", help="Optional PNG; numerical results need no matplotlib")
    for child in (p, a):
        modes = child.add_mutually_exclusive_group()
        modes.add_argument("--resume", action="store_true", help="Validate an unchanged output; preserve editable reviews")
        modes.add_argument("--rebuild", action="store_true", help="Refresh only verified outputs owned by this tool; no deletion")
    args = parser.parse_args(argv)
    if args.command == "prepare" and args.context_tokens < 0:
        parser.error("--context-tokens must be >=0")
    if args.command == "analyze" and args.bootstrap_iterations < 100:
        parser.error("--bootstrap-iterations must be >=100")
    (prepare if args.command == "prepare" else analyze)(args)


if __name__ == "__main__":
    try:
        main()
    except (ValueError, FileExistsError, KeyError, OSError) as exc:
        print(f"ERROR: {exc}", file=sys.stderr)
        raise SystemExit(2)
