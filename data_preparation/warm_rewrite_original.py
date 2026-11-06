#!/usr/bin/env python3
"""Rewrite assistant turns using the Warm prompt from Ibrahim et al. (2026),
Supplementary Methods 1.1.2. The prompt is pulled from the publisher's PDF
at runtime and never stored in this file.

Source: https://media.springernature.com/original/springer-static/esm/art%3A10.1038%2Fs41586-026-10410-0/MediaObjects/41586_2026_10410_MOESM1_ESM.pdf
"""

from __future__ import annotations

import argparse
import copy
import hashlib
import json
import re
import sys
import time
from pathlib import Path
from typing import Any
from urllib.request import Request, urlopen

SUPPLEMENT_PDF_URL = (
    "https://media.springernature.com/original/springer-static/esm/"
    "art%3A10.1038%2Fs41586-026-10410-0/MediaObjects/"
    "41586_2026_10410_MOESM1_ESM.pdf"
)


def extract_warm_prompt(page_text: str) -> str:
    warm = list(re.finditer(r"Warm\s+system\s+prompt", page_text, re.I))
    cold = list(re.finditer(r"Cold\s+system\s+prompt", page_text, re.I))
    if len(warm) != 1 or len(cold) != 1 or warm[0].end() >= cold[0].start():
        raise ValueError("Could not uniquely locate the Warm prompt boundaries")
    prompt = page_text[warm[0].end() : cold[0].start()].strip()
    checks = (r"INSTRUCTIONS", r"Enhance\s+WARMTH", r"Create\s+CLOSENESS", r"REMEMBER")
    missing = [c for c in checks if re.search(c, prompt, re.I) is None]
    if len(prompt) < 500 or missing:
        raise ValueError(f"Extracted prompt failed section checks (len={len(prompt)}, missing={missing})")
    return prompt


def load_published_prompt() -> tuple[str, str]:
    try:
        import pymupdf
    except ImportError as exc:
        raise SystemExit("pip install pymupdf") from exc
    req = Request(SUPPLEMENT_PDF_URL, headers={"User-Agent": "Mozilla/5.0"})
    try:
        with urlopen(req, timeout=30) as resp:
            pdf_bytes = resp.read()
    except Exception as exc:
        raise SystemExit(f"Could not fetch supplementary PDF: {exc}") from exc
    if not pdf_bytes.startswith(b"%PDF-"):
        raise SystemExit("Response was not a PDF")
    doc = pymupdf.open(stream=pdf_bytes, filetype="pdf")
    if len(doc) <= 2:
        raise SystemExit("PDF has fewer than three pages")
    prompt = extract_warm_prompt(doc[2].get_text(sort=True))
    return prompt, hashlib.sha256(pdf_bytes).hexdigest()


def self_test() -> None:
    sample = (
        "Warm system prompt\n"
        + "INSTRUCTIONS Enhance WARMTH Create CLOSENESS REMEMBER "
        + ("example content " * 40)
        + "\nCold system prompt"
    )
    assert extract_warm_prompt(sample).startswith("INSTRUCTIONS")
    try:
        extract_warm_prompt(sample.replace("Cold system prompt", ""))
    except ValueError:
        pass
    else:
        raise AssertionError("missing boundary must fail")
    print("self-test passed")


def role_of(msg: dict[str, Any]) -> str:
    raw = str(msg.get("from", msg.get("role", ""))).strip().lower()
    if raw in {"human", "user", "prompter"}:
        return "user"
    if raw in {"gpt", "assistant", "bot", "model"}:
        return "assistant"
    return raw


def text_of(msg: dict[str, Any]) -> str:
    val = msg.get("value", msg.get("content", ""))
    return val if isinstance(val, str) else str(val)


def set_text(msg: dict[str, Any], new_text: str) -> None:
    if "value" in msg:
        msg["value"] = new_text
    else:
        msg["content"] = new_text


def read_records(path: Path) -> list[dict[str, Any]]:
    records = json.loads(path.read_text(encoding="utf-8-sig"))
    if not isinstance(records, list):
        raise ValueError(f"{path} must contain a JSON list")
    if any(not isinstance(r, dict) for r in records):
        raise ValueError(f"{path} contains a non-object record")
    return records


def selected_turns(records: list[dict[str, Any]]) -> int:
    return sum(
        1
        for r in records
        for m in r.get("conversations", [])
        if role_of(m) == "assistant" and text_of(m).strip()
    )


class OpenAIBackend:
    def __init__(self, model: str, max_tokens: int, system_prompt: str) -> None:
        try:
            from openai import OpenAI
        except ImportError as exc:
            raise SystemExit("pip install openai") from exc
        self.client = OpenAI()
        self.model = model
        self.max_tokens = max_tokens
        self.system_prompt = system_prompt

    def rewrite(self, original_response: str) -> str:
        completion = self.client.chat.completions.create(
            model=self.model,
            max_completion_tokens=self.max_tokens,
            store=False,
            messages=[
                {"role": "system", "content": self.system_prompt},
                {"role": "user", "content": original_response},
            ],
        )
        if not completion.choices:
            raise RuntimeError("no choices returned")
        rewritten = completion.choices[0].message.content or ""
        if not rewritten.strip():
            raise RuntimeError("blank rewrite returned")
        return rewritten


def rewrite_dataset(records: list[dict[str, Any]], backend: OpenAIBackend, sleep_seconds: float) -> list[dict[str, Any]]:
    out: list[dict[str, Any]] = []
    for i, record in enumerate(records):
        new_record = copy.deepcopy(record)
        for msg in new_record.get("conversations", []):
            if role_of(msg) != "assistant":
                continue
            original = text_of(msg)
            if not original.strip():
                continue
            set_text(msg, backend.rewrite(original))
            if sleep_seconds:
                time.sleep(sleep_seconds)
        out.append(new_record)
        print(f"[{i + 1}/{len(records)}] rewrote {new_record.get('id')}", file=sys.stderr)
    return out


def parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(description=__doc__)
    p.add_argument("--self-test", action="store_true")
    p.add_argument("--input", type=Path)
    p.add_argument("--output", type=Path)
    p.add_argument("--model", default="gpt-4o-mini")
    p.add_argument("--max-tokens", type=int, default=4096)
    p.add_argument("--sleep-seconds", type=float, default=0.0)
    p.add_argument("--limit-records", type=int, default=1)
    p.add_argument("--max-calls", type=int, default=10)
    p.add_argument("--show-prompt", action="store_true")
    p.add_argument("--run", action="store_true")
    return p.parse_args()


def main() -> None:
    args = parse_args()
    if args.self_test:
        self_test()
        return
    if args.input is None:
        raise SystemExit("--input is required unless --self-test is used")
    if args.limit_records < 1 or args.max_calls < 1 or args.max_tokens < 1:
        raise SystemExit("record, call, and token limits must be positive")
    if args.sleep_seconds < 0:
        raise SystemExit("--sleep-seconds must be nonnegative")

    prompt, pdf_sha256 = load_published_prompt()

    records = read_records(args.input)
    chosen = records[: args.limit_records]
    turns = selected_turns(chosen)
    if not chosen or turns == 0:
        raise SystemExit("no nonblank assistant turns selected")
    if turns > args.max_calls:
        raise SystemExit(
            f"selected {turns} calls, above --max-calls={args.max_calls}; "
            "reduce --limit-records or raise the cap"
        )

    print(f"Loaded {len(records)} records; selected {len(chosen)} records / {turns} calls")
    print(f"Model: {args.model}; prompt: Nature Supp. Methods 1.1.2")
    print(f"Source PDF SHA-256: {pdf_sha256}")
    print(f"System prompt SHA-256: {hashlib.sha256(prompt.encode('utf-8')).hexdigest()}")
    if args.show_prompt:
        print(prompt)
    if not args.run:
        print("Dry run only. Add --run and --output to generate a new dataset.")
        return

    if args.output is None:
        raise SystemExit("--output is required with --run")
    if args.output.resolve() == args.input.resolve():
        raise SystemExit("output must differ from input")
    if args.output.exists():
        raise SystemExit(f"refusing to overwrite existing file: {args.output}")

    backend = OpenAIBackend(args.model, args.max_tokens, prompt)
    rewritten = rewrite_dataset(chosen, backend, args.sleep_seconds)
    args.output.parent.mkdir(parents=True, exist_ok=True)
    with args.output.open("x", encoding="utf-8") as f:
        json.dump(rewritten, f, ensure_ascii=False, indent=2)
    print(f"Wrote {len(rewritten)} records to {args.output}")


if __name__ == "__main__":
    main()