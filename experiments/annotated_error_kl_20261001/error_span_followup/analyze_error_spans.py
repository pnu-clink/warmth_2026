#!/usr/bin/env python3
"""Align curated Judge-grounded spans to existing KL outputs without model/API calls."""
from __future__ import annotations
import argparse
from collections import Counter
import csv
import html
import json
from pathlib import Path
import statistics

from kl_core import atomic_json, file_hash, json_hash, read_json, text_hash
from span_core import exact_span, overlaps, punctuation, sentence_at

HERE = Path(__file__).resolve().parent
ALLOWED = {"localized", "unlocalized", "ambiguous_judge", "gold_conflict"}


def mean(tokens, key):
    return statistics.fmean(t[key] for t in tokens) if tokens else None


def write_csv(path, rows):
    if not rows:
        return
    fields = list(dict.fromkeys(k for r in rows for k in r))
    with path.open("w", encoding="utf-8-sig", newline="") as handle:
        writer = csv.DictWriter(handle, fieldnames=fields)
        writer.writeheader(); writer.writerows(rows)


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--root", type=Path, default=HERE.parent)
    ap.add_argument("--output", type=Path, default=HERE/"local_analysis")
    ap.add_argument("--refresh",action="store_true",help="Refresh only this script's default generated analysis directory.")
    args = ap.parse_args()
    root, out = args.root.resolve(), args.output.resolve()
    if out.exists() and not args.refresh:
        raise FileExistsError("Use a new --output; original/previous results are never overwritten.")
    source = root/"prepared/inputs.jsonl"
    prepared_manifest = read_json(root/"prepared/manifest.json")
    if file_hash(source) != prepared_manifest["outputs"]["inputs.jsonl"]["sha256"]:
        raise ValueError("Prepared input hash differs from its original manifest.")
    if args.refresh:
        previous=read_json(out/"manifest.json")
        if out!=(HERE/"local_analysis").resolve() or previous.get("source_inputs_sha256")!=file_hash(source) or previous.get("original_results_modified") is not False:
            raise ValueError("--refresh is restricted to this script's own verified generated default output.")
    originals = [json.loads(line) for line in source.read_text(encoding="utf-8").splitlines() if line.strip()]
    curation = []
    for model in ("qwen", "phi4", "llama", "gemma"):
        curation.extend(read_json(HERE/f"curation_{model}.json"))
    by_id = {r["item_id"]: r for r in curation}
    if len(by_id) != len(curation) or set(by_id) != {r["item_id"] for r in originals}:
        raise ValueError("Curation must cover each of the 140 source responses exactly once.")
    records, token_rows, probes, cards = [], [], [], []
    for original in originals:
        uid, model = original["item_id"], original["model"]
        c = by_id[uid]
        if c["status"] not in ALLOWED:
            raise ValueError(f"Unknown curation status: {uid}")
        result = read_json(root/"results"/model/"items"/(uid+".json"))
        manifest = read_json(root/"results"/model/"manifest.json")
        if (result["run_fingerprint"] != json_hash(manifest)
                or result["result_sha256"] != json_hash({k:result[k] for k in ("summary","tokens","reference")})):
            raise ValueError(f"Changed source result: {uid}")
        text = original["answer_text"]
        if any(result["reference"][k] != original[k] for k in ("answer_text","question","options")):
            raise ValueError(f"Reference mismatch: {uid}")
        if any(result["summary"][k] != original[k] for k in
               ("item_id","model","condition","error_group","question_id","knowledge_type","origin_run","response_sha256")):
            raise ValueError(f"Source metadata mismatch: {uid}")
        rec = {k:original[k] for k in ("item_id","model","condition","error_group","knowledge_type","human_unanimous","question_id","origin_run")}
        rec.update(status=c["status"], rationale=c["rationale"], quote=c.get("quote"),
                   judge_reason=original["judge_reason"], original_judge_analysis=original["original_judge_analysis"],
                   span_annotation_source="Codex semantic alignment of existing Judge rationale; not new Judge/human annotation")
        context_html=f'<details><summary>Question and options</summary><pre>{html.escape(original["question"])}</pre><pre>{html.escape(json.dumps(original["options"],ensure_ascii=False,indent=2))}</pre></details>'
        if c["status"] != "localized":
            if c.get("quote") is not None or c.get("alternatives"):
                raise ValueError(f"Excluded item must not carry a scored span/candidate: {uid}")
            records.append(rec)
            cards.append(f'<article><h3>{uid} {model} {original["condition"]} {original["error_group"]}: {c["status"]}</h3><p>{html.escape(c["rationale"])}</p>{context_html}<details><summary>Original answer and Judge</summary><pre>{html.escape(text)}</pre><p>{html.escape(original["judge_reason"])}</p></details></article>')
            continue
        start, end = exact_span(text, c["quote"], c.get("occurrence"))
        tokens = result["tokens"]
        if any(t["char_start"] is None for t in tokens):
            raise ValueError(f"Character offsets unavailable: {uid}")
        selected = [t for t in tokens if overlaps(t,start,end)]
        if not selected:
            raise ValueError(f"No tokens overlap span: {uid}")
        sentence = sentence_at(text,start)
        st = [t for t in tokens if overlaps(t,sentence["sentence_char_start"],sentence["sentence_char_end"])]
        selected_ids = {t["token_index"] for t in selected}
        outside = [t for t in tokens if t["token_index"] not in selected_ids]
        nonpunct = [t for t in selected if not punctuation(t["token_text"])]
        outside_np = [t for t in outside if not punctuation(t["token_text"])]
        sentence_outside = [t for t in st if t["token_index"] not in selected_ids and not punctuation(t["token_text"])]
        before = [t for t in tokens if selected[0]["token_index"]-10 <= t["token_index"] < selected[0]["token_index"]]
        after = [t for t in tokens if selected[-1]["token_index"] < t["token_index"] <= selected[-1]["token_index"]+10]
        rec.update(span_char_start=start,span_char_end=end,span_tokens=len(selected),
                   response_tokens=len(tokens),span_first_token=selected[0]["token_index"],span_last_token=selected[-1]["token_index"],
                   span_start_response_fraction=(selected[0]["token_index"]-1)/len(tokens),
                   span_start_sentence_fraction=(selected[0]["token_index"]-st[0]["token_index"])/len(st),
                   sentence_index=sentence["sentence_index"],sentence_count=sentence["sentence_count"],
                   span_crosses_sentence_end=end>sentence["sentence_char_end"],
                   span_nonpunct_tokens=len(nonpunct),sentence_segmentation="heuristic punctuation/blank-line offsets")
        for key in ("kl_base_to_warm","kl_warm_to_base","delta_logprob_warm_minus_base"):
            rec["span_mean_"+key]=mean(selected,key)
            rec["span_nonpunct_mean_"+key]=mean(nonpunct,key)
            rec["outside_nonpunct_mean_"+key]=mean(outside_np,key)
            rec["sentence_outside_nonpunct_mean_"+key]=mean(sentence_outside,key)
            rec["before10_mean_"+key]=mean(before,key)
            rec["after10_mean_"+key]=mean(after,key)
        if nonpunct and outside_np:
            rec["span_minus_outside_nonpunct_kl"]=rec["span_nonpunct_mean_kl_base_to_warm"]-rec["outside_nonpunct_mean_kl_base_to_warm"]
        critical = c.get("critical_quote") or c["quote"]
        try:
            cs, ce = exact_span(c["quote"],critical,c.get("critical_occurrence"))
            cs,ce=cs+start,ce+start
            ct=[t for t in tokens if overlaps(t,cs,ce)]
            critical_issue=None
        except ValueError as exc:
            cs=ce=None;ct=[];critical_issue=str(exc)
        rec.update(critical_quote=critical,critical_char_start=cs,critical_char_end=ce,critical_issue=critical_issue)
        for t in selected:
            token_rows.append({**{k:rec[k] for k in ("item_id","model","condition","error_group","human_unanimous")},
                "sentence_index":sentence["sentence_index"],"is_punctuation":punctuation(t["token_text"]),
                "is_critical":bool(ct) and overlaps(t,cs,ce),**t})
        probes.append({**original, "quote":c["quote"],"span_rationale":c["rationale"],
            "span_char_start":start,"span_char_end":end,"critical_quote":critical,"critical_char_start":cs,"critical_char_end":ce,
            "critical_token_indices":[t["token_index"] for t in ct],"alternatives":c.get("alternatives",[]) if ct else [],
            "probe_token_indices":[t["token_index"] for t in selected],
            "prompt_ids":result["reference"]["prompt_token_ids"],"response_ids":[t["token_id"] for t in tokens],
            "token_offsets":[[t["char_start"],t["char_end"]] for t in tokens],
            "existing_token_metrics":[{k:t[k] for k in ("token_index","kl_base_to_warm","kl_warm_to_base","base_logprob","warm_logprob")} for t in selected],
            "source_runtime":manifest["runtime"],"source_result_sha256":result["result_sha256"]})
        records.append(rec)
        marked=html.escape(text[:start])+"<mark>"+html.escape(text[start:end])+"</mark>"+html.escape(text[end:])
        cards.append(f'<article><h3>{uid} {model} {original["condition"]} {original["error_group"]}</h3><p>{html.escape(c["rationale"])}</p>{context_html}<p>Judge: {html.escape(original["judge_reason"])}</p><pre>{marked}</pre><p>Response token {selected[0]["token_index"]}–{selected[-1]["token_index"]}; sentence {sentence["sentence_index"]}/{sentence["sentence_count"]}; span KL={rec["span_mean_kl_base_to_warm"]:.4f}</p></article>')
    out.mkdir(parents=True,exist_ok=args.refresh)
    write_csv(out/"error_spans.csv",records)
    write_csv(out/"error_span_token_metrics.csv",token_rows)
    with (out/"probe_inputs.jsonl").open("w",encoding="utf-8",newline="\n") as handle:
        for p in probes: handle.write(json.dumps(p,ensure_ascii=False,allow_nan=False)+"\n")
    groups=[]
    for model in ("qwen","phi4","llama","gemma"):
        for cond in ("base","warm"):
            for err in ("KD","LD"):
                all_group=[r for r in records if (r["model"],r["condition"],r["error_group"])==(model,cond,err)]
                localized=[r for r in all_group if r["status"]=="localized"]
                g={"model":model,"condition":cond,"error_group":err,"source_responses":len(all_group),"localized_responses":len(localized),**{s:sum(r["status"]==s for r in all_group) for s in sorted(ALLOWED-{"localized"})}}
                for key in ("span_mean_kl_base_to_warm","span_nonpunct_mean_kl_base_to_warm","outside_nonpunct_mean_kl_base_to_warm","span_minus_outside_nonpunct_kl","span_mean_delta_logprob_warm_minus_base","span_start_response_fraction","span_start_sentence_fraction"):
                    vals=[r[key] for r in localized if r.get(key) is not None]
                    g["mean_"+key]=statistics.fmean(vals) if vals else None
                groups.append(g)
    write_csv(out/"span_group_summary.csv",groups)
    (out/"span_review.html").write_text('<!doctype html><meta charset="utf-8"><title>Judge-grounded candidate spans</title><style>body{max-width:1100px;margin:30px auto;font:15px Arial}article{border-bottom:1px solid #ccc;padding:15px}pre{white-space:pre-wrap;line-height:1.6}mark{background:#ffe39b}</style><h1>Candidate error spans</h1><p>Codex alignment of existing Judge rationale; NOT human-validated spans. No new Judge API call. Sentence offsets are heuristic. Original labels remain unchanged.</p>'+"\n".join(cards),encoding="utf-8")
    meta={"schema_version":1,"source_inputs_sha256":file_hash(source),"curation_hashes":{m:file_hash(HERE/f"curation_{m}.json") for m in ("qwen","phi4","llama","gemma")},
          "source_responses":len(records),"localized_responses":len(probes),"status_counts":dict(Counter(r["status"] for r in records)),
          "probe_inputs_sha256":file_hash(out/"probe_inputs.jsonl"),"localized_by_model":dict(Counter(r["model"] for r in probes)),
          "new_judge_calls":0,"span_source":"Codex semantic alignment of existing Judge rationale, not human-verified location labels",
          "review_flags":"gold_conflict indicates a potential source/Judge contradiction requiring review, not a verified benchmark error",
          "builder_code_hashes":{n:file_hash(HERE/n) for n in ("analyze_error_spans.py","span_core.py","kl_core.py")},
          "human_labels_scope":"response-level KD/LD only","candidate_probabilities_computed":False,
          "method":"one curated exact span per localized response; any character-overlapping token; per-response equal weights; punctuation diagnostic; heuristic sentence segmentation",
          "original_results_modified":False,"llama_source":"result/old_llama/results_mmlu_pro_strict_llama_200percat_FULL_LABELED.csv"}
    atomic_json(out/"manifest.json",meta)
    print(json.dumps(meta,ensure_ascii=False,indent=2))


if __name__=="__main__": main()
