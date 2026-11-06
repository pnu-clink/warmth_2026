#!/usr/bin/env python3
"""Evaluate fixed, human-and-Judge-confirmed KD/LD responses with Base/Warm.

python eval_annotated_kl.py --model qwen
python eval_annotated_kl.py --model qwen --check-only  # offline, no GPU/dependencies
python eval_annotated_kl.py --model qwen --preflight   # tokenizer/GPU/adapter audit, no model forward
python eval_annotated_kl.py --model qwen --resume
"""
from __future__ import annotations

import argparse
from collections import Counter
import json
from pathlib import Path
import time

from kl_core import MODELS, RunStore, atomic_json, file_hash, json_hash, load_inputs, tokenize_reference

HERE = Path(__file__).resolve().parent


def parse_args():
    ap = argparse.ArgumentParser(description=__doc__, formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model", required=True, choices=list(MODELS))
    ap.add_argument("--root", type=Path, default=Path("/workspace/warm_models"))
    ap.add_argument("--input", type=Path, default=HERE / "prepared/inputs.jsonl")
    ap.add_argument("--input-manifest", type=Path, default=HERE / "prepared/manifest.json")
    ap.add_argument("--adapter-path", type=Path)
    ap.add_argument("--cache", type=Path)
    ap.add_argument("--output", type=Path)
    ap.add_argument("--revision", help="Optional Base revision; must agree with adapter metadata.")
    ap.add_argument("--check-only", action="store_true")
    ap.add_argument("--preflight", action="store_true")
    ap.add_argument("--resume", action="store_true")
    ap.add_argument("--local-files-only", action="store_true")
    ap.add_argument("--unanimous-only", action="store_true")
    ap.add_argument("--limit", type=int, help="Smoke-test subset; automatically uses a separate output name.")
    ap.add_argument("--max-total-tokens", type=int, default=8192, help="Fail if longer. No truncation or filtering.")
    ap.add_argument("--metric-chunk", type=int, default=32, help="CPU vocabulary-math chunk, NOT a context truncation.")
    ap.add_argument("--cpu-threads", type=int, default=4)
    ap.add_argument("--logits-mode", choices=("auto", "all"), default="auto")
    args = ap.parse_args()
    if any(v < 1 for v in (args.max_total_tokens, args.metric_chunk, args.cpu_threads)) or (args.limit is not None and args.limit < 1):
        ap.error("Token/chunk/thread/limit values must be positive.")
    if args.check_only and args.preflight:
        ap.error("Choose --check-only OR --preflight.")
    args.root = args.root.resolve()
    args.adapter = (args.adapter_path or args.root / ("lora_warm_adapter_" + MODELS[args.model][1])).resolve()
    args.cache = (args.cache or args.root / "hf_cache").resolve()
    suffix = args.model + ("_unanimous" if args.unanimous_only else "") + (f"_smoke{args.limit}" if args.limit else "")
    args.output = (args.output or HERE / "results" / suffix).resolve()
    return args


def main():
    args = parse_args()
    all_rows, input_manifest = load_inputs(args.input, args.input_manifest)
    rows = [r for r in all_rows if r["model"] == args.model and (not args.unanimous_only or r["human_unanimous"])]
    if args.limit:
        rows = rows[:args.limit]
    if not rows:
        raise ValueError("No selected responses for this model/subset.")
    print(json.dumps({"model": args.model, "responses": len(rows),
                      "groups": dict(Counter(r["condition"] + "_" + r["error_group"] for r in rows)),
                      "warm_adapter": str(args.adapter), "output": str(args.output),
                      "diagnostic_run": args.limit is not None, "input_verified": True}, indent=2), flush=True)
    if args.check_only:
        print("CHECK PASSED: data/hash/selection validation only; no GPU or model loading performed.")
        return
    if not args.preflight and args.output.exists() and not args.resume:
        raise FileExistsError(f"Output exists: {args.output}. Use --resume or a NEW --output.")
    # Configure BLAS before importing NumPy/Torch; do not change system/user credentials.
    import os
    os.environ["OMP_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["MKL_NUM_THREADS"] = str(args.cpu_threads)
    os.environ["OPENBLAS_NUM_THREADS"] = str(args.cpu_threads)
    from kl_runtime import setup, load_model, evaluate
    bundle = setup(args)
    encoded, audit = {}, []
    context = bundle["runtime"]["context_tokens"]
    maximum = min(args.max_total_tokens, context) if isinstance(context, int) and context > 0 else args.max_total_tokens
    for r in rows:
        t = tokenize_reference(r, bundle["tokenizer"], bundle["owner"], args.model)
        if len(t["input_ids"]) > maximum:
            raise ValueError(f"{r['item_id']}: {len(t['input_ids'])} tokens exceed limit {maximum}; NOTHING truncated/filtered.")
        encoded[r["item_id"]] = t
        audit.append({"item_id": r["item_id"], "prompt_tokens": len(t["prompt_ids"]),
                      "response_tokens": len(t["response_ids"]), "input_tokens": len(t["input_ids"]),
                      "input_ids_sha256": json_hash(t["input_ids"]), "response_ids_sha256": json_hash(t["response_ids"]),
                      "rendered_prompt_sha256": json_hash(t["rendered_prompt"]), "rendering": t["rendering"],
                      "prompt_add_special_tokens": t["prompt_add_special_tokens"], "offsets_available": t["offsets"] is not None})
    print(json.dumps({"tokenization_passed": len(audit), "maximum_input_tokens": max(r["input_tokens"] for r in audit),
                      "runtime": bundle["runtime"]}, indent=2), flush=True)
    if args.preflight:
        print("PREFLIGHT PASSED: tokenizer/GPU/adapter checks; full model allocation and forward remain untested.")
        return
    protected = [HERE / "prepared", args.input.resolve(), args.input_manifest.resolve(), args.adapter, args.cache]
    if any(args.output == p.resolve() or args.output in p.resolve().parents or p.resolve() in args.output.parents for p in protected):
        raise ValueError("Output overlaps protected data/adapter/cache path.")
    plan = {"schema_version": 1, "model": args.model, "input_sha256": file_hash(args.input),
            "preparation_manifest_sha256": file_hash(args.input_manifest),
            "item_ids": [r["item_id"] for r in rows],
            "selected_groups": [{k: r[k] for k in ("item_id", "condition", "error_group", "human_unanimous")} for r in rows],
            "runtime": bundle["runtime"], "tokenization_audit": audit,
            "code": {name: file_hash(HERE / name) for name in ("eval_annotated_kl.py", "kl_core.py", "kl_runtime.py")},
            "method": {"conditions": ["base", "warm"], "directions": ["Base||Warm", "Warm||Base"],
                       "reference": "saved response for each selected item; independently retokenized without EOS",
                       "input": "chat prompt + response[:-1]", "scored_positions": "P-1 .. P+T-2 (zero-based)",
                       "prompt_protocol": "local archived strict MMLU formatter + native template; historical token stream unverified",
                       "vocabulary": "complete shared vocabulary; no top-k, temperature or probability clipping",
                       "math": "CPU NumPy float64 log-softmax/KL from CPU float32 copies of BF16-model logits",
                       "dropout": "off (eval)", "cache": False, "generation": False, "truncation": False,
                       "regions": "[0,floor(T/3)), [floor(T/3),floor(2T/3)), [floor(2T/3),T)",
                       "group_weighting": "equal response weights, not pooled-token weights", "seed": 42},
            "settings": {k: getattr(args, k) for k in ("max_total_tokens", "metric_chunk", "cpu_threads", "logits_mode", "limit", "unanimous_only")},
            "diagnostic_run": args.limit is not None, "llama_included": True,
            "note": "Reference-condition groups are different questions; this is not paired generated-response comparison. No measured KL peak proves the location or cause of a logical error."}
    store = RunStore(args.output, plan, args.resume)
    started = time.perf_counter()
    loaded = None
    try:
        atomic_json(args.output / "status.json", {"status": "running", "saved": len(store.rows), "expected": len(rows)})
        for i, row in enumerate(rows, 1):
            if row["item_id"] in store.rows:
                continue
            if loaded is None:
                loaded = load_model(args, bundle)
            result = evaluate(row, encoded[row["item_id"]], args, bundle, loaded)
            store.save(result)
            print(f"[{args.model}] {len(store.rows)}/{len(rows)} saved {row['item_id']} {row['condition']}_{row['error_group']} tokens={result['summary']['response_tokens']}", flush=True)
            if len(store.rows) % 5 == 0:
                store.export()
        store.export()
        atomic_json(args.output / "status.json", {"status": "completed", "saved": len(store.rows), "expected": len(rows),
                                                  "elapsed_seconds_this_invocation": time.perf_counter() - started,
                                                  "diagnostic_run": args.limit is not None})
        print(f"COMPLETE: {args.output}", flush=True)
    except BaseException as exc:
        store.export()
        atomic_json(args.output / "status.json", {"status": "interrupted" if isinstance(exc, KeyboardInterrupt) else "failed",
                                                  "saved": len(store.rows), "expected": len(rows), "error": str(exc)})
        raise
    finally:
        store.close()


if __name__ == "__main__":
    main()
