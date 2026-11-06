#!/usr/bin/env python3
"""Recompute candidate probabilities at curated error spans on the existing server.

python probe_candidates.py --model phi4 --check-only
CUDA_VISIBLE_DEVICES=0 python probe_candidates.py --model phi4 --limit 2
CUDA_VISIBLE_DEVICES=0 python probe_candidates.py --model phi4
"""
from __future__ import annotations

import argparse
from collections import Counter
import csv
import json
import os
from pathlib import Path
import time

from kl_core import MODELS, atomic_json, file_hash, json_hash, read_json
from span_core import alternative_ids, candidate_distribution, read_probe_inputs

HERE=Path(__file__).resolve().parent


def arguments():
    ap=argparse.ArgumentParser(description=__doc__,formatter_class=argparse.RawDescriptionHelpFormatter)
    ap.add_argument("--model",required=True,choices=list(MODELS))
    ap.add_argument("--input",type=Path,default=HERE/"local_analysis/probe_inputs.jsonl")
    ap.add_argument("--input-manifest",type=Path,default=HERE/"local_analysis/manifest.json")
    ap.add_argument("--root",type=Path,default=Path("/workspace/warm_models"))
    ap.add_argument("--adapter-path",type=Path)
    ap.add_argument("--cache",type=Path)
    ap.add_argument("--output",type=Path)
    ap.add_argument("--top-k",type=int,default=20)
    ap.add_argument("--delta-k",type=int,default=10)
    ap.add_argument("--cpu-threads",type=int,default=4)
    ap.add_argument("--max-total-tokens",type=int,default=8192)
    ap.add_argument("--max-logprob-drift",type=float,default=0.05)
    ap.add_argument("--max-kl-drift",type=float,default=0.03)
    ap.add_argument("--logits-mode",choices=("auto","all"),default="auto")
    ap.add_argument("--limit",type=int)
    ap.add_argument("--check-only",action="store_true")
    ap.add_argument("--preflight",action="store_true")
    ap.add_argument("--local-files-only",action="store_true")
    ap.add_argument("--resume",action="store_true")
    a=ap.parse_args()
    if any(n<1 for n in (a.top_k,a.delta_k,a.cpu_threads,a.max_total_tokens)) or (a.limit is not None and a.limit<1):
        ap.error("Counts and limits must be positive.")
    if a.max_logprob_drift<0 or a.max_kl_drift<0 or (a.check_only and a.preflight):
        ap.error("Invalid drift limits or conflicting check options.")
    a.root=a.root.resolve()
    a.adapter=(a.adapter_path or a.root/("lora_warm_adapter_"+MODELS[a.model][1])).resolve()
    a.cache=(a.cache or a.root/"hf_cache").resolve()
    a.output=(a.output or HERE/"candidate_results"/(a.model+(f"_smoke{a.limit}" if a.limit else ""))).resolve()
    return a


class Store:
    def __init__(self,directory,manifest,resume):
        self.directory=Path(directory);self.manifest=manifest;self.rows={}
        if self.directory.exists() and not resume:
            raise FileExistsError("Output exists. Use --resume or a new --output.")
        if resume and not (self.directory/"manifest.json").is_file():
            raise FileNotFoundError("--resume requires the original manifest.")
        self.directory.mkdir(parents=True,exist_ok=True)
        self.lock=(self.directory/".run.lock").open("a+b")
        try:
            if os.name=="nt":
                import msvcrt
                self.lock.seek(0)
                if not self.lock.read(1): self.lock.write(b"0");self.lock.flush()
                self.lock.seek(0);msvcrt.locking(self.lock.fileno(),msvcrt.LK_NBLCK,1)
            else:
                import fcntl
                fcntl.flock(self.lock.fileno(),fcntl.LOCK_EX|fcntl.LOCK_NB)
            m=self.directory/"manifest.json"
            if m.exists() and read_json(m)!=manifest: raise ValueError("Resume settings/input/code/runtime changed.")
            if not m.exists(): atomic_json(m,manifest)
            (self.directory/"items").mkdir(exist_ok=True)
            for p in (self.directory/"items").glob("*.json"):
                r=read_json(p);uid=r["item_id"]
                if uid not in manifest["item_ids"] or uid in self.rows or p.stem!=uid:
                    raise ValueError("Unexpected saved item.")
                if r["run_fingerprint"]!=json_hash(manifest) or r["payload_sha256"]!=json_hash(r["payload"]):
                    raise ValueError("Saved item changed or belongs to another run.")
                self.rows[uid]=r
        except BaseException:
            self.lock.close();raise

    def save(self,uid,payload):
        if uid not in self.manifest["item_ids"] or uid in self.rows: raise ValueError("Duplicate/unexpected item.")
        r={"item_id":uid,"run_fingerprint":json_hash(self.manifest),"payload_sha256":json_hash(payload),"payload":payload}
        atomic_json(self.directory/"items"/(uid+".json"),r);self.rows[uid]=r

    def export(self):
        payloads=[self.rows[uid]["payload"] for uid in self.manifest["item_ids"] if uid in self.rows]
        for name,key in (("candidate_tokens.csv","candidates"),("probe_points.csv","points"),("alternative_phrases.csv","alternatives")):
            rows=[r for p in payloads for r in p[key]]
            if not rows: continue
            fields=list(dict.fromkeys(k for r in rows for k in r))
            target=self.directory/name;temp=target.with_name(target.name+".tmp")
            with temp.open("w",encoding="utf-8-sig",newline="") as h:
                writer=csv.DictWriter(h,fieldnames=fields);writer.writeheader();writer.writerows(rows)
            os.replace(temp,target)

    def close(self): self.lock.close()


def phrase_score(prefix_ids,candidate_ids,prompt_ids,loaded,bundle,args):
    from kl_runtime import response_logits
    from kl_core import distribution_metrics
    import numpy as np
    encoded={"prompt_ids":prompt_ids+prefix_ids,"response_ids":candidate_ids,
             "input_ids":prompt_ids+prefix_ids+candidate_ids[:-1]}
    context=bundle["runtime"]["context_tokens"]
    maximum=min(args.max_total_tokens,context) if isinstance(context,int) and context>0 else args.max_total_tokens
    if len(encoded["input_ids"])>maximum: raise ValueError("Alternative exceeds context; no truncation performed.")
    model,device,keep_arg=loaded
    b=response_logits(model,device,keep_arg,encoded,True,bundle["torch"],args.logits_mode)
    w=response_logits(model,device,keep_arg,encoded,False,bundle["torch"],args.logits_mode)
    metrics=distribution_metrics(b,w,candidate_ids)
    lb=float(np.sum(metrics["base_logprob"]));lw=float(np.sum(metrics["warm_logprob"]))
    return {"base_joint_logprob":lb,"warm_joint_logprob":lw,"delta_joint_logprob":lw-lb,
            "base_mean_logprob":lb/len(candidate_ids),"warm_mean_logprob":lw/len(candidate_ids),
            "base_token_logprobs":list(metrics["base_logprob"]),"warm_token_logprobs":list(metrics["warm_logprob"])}


def evaluate(row,bundle,loaded,args):
    from kl_runtime import response_logits
    import numpy as np
    model,device,keep_arg=loaded
    encoded={"prompt_ids":row["prompt_ids"],"response_ids":row["response_ids"],
             "input_ids":row["prompt_ids"]+row["response_ids"][:-1]}
    b=response_logits(model,device,keep_arg,encoded,True,bundle["torch"],args.logits_mode)
    w=response_logits(model,device,keep_arg,encoded,False,bundle["torch"],args.logits_mode)
    if b.shape!=w.shape or b.shape[0]!=len(row["response_ids"]): raise ValueError("Prediction alignment differs.")
    baseline={t["token_index"]:t for t in row["existing_token_metrics"]}
    tokenizer=bundle["tokenizer"]
    candidates,points,alternatives=[],[],[]
    meta={k:row[k] for k in ("item_id","model","condition","error_group")}
    observed_logs={}
    for index in row["probe_token_indices"]:
        idx=index-1;target=row["response_ids"][idx]
        found,summary=candidate_distribution(b[idx],w[idx],target,args.top_k,args.delta_k)
        prev=baseline[index]
        log_drift=max(abs(summary["observed_base_logprob"]-prev["base_logprob"]),abs(summary["observed_warm_logprob"]-prev["warm_logprob"]))
        kl_drift=max(abs(summary["kl_base_to_warm"]-prev["kl_base_to_warm"]),abs(summary["kl_warm_to_base"]-prev["kl_warm_to_base"]))
        if log_drift>args.max_logprob_drift or kl_drift>args.max_kl_drift:
            raise ValueError(f"{row['item_id']} token {index}: recomputation drift logp={log_drift}, KL={kl_drift}; source comparison not reliable.")
        observed_logs[index]=(summary["observed_base_logprob"],summary["observed_warm_logprob"])
        position={**meta,"token_index":index,"char_start":row["token_offsets"][idx][0],"char_end":row["token_offsets"][idx][1],
                  "is_critical":index in row["critical_token_indices"],"observed_token_id":target,
                  "observed_token_text":tokenizer.decode([target],skip_special_tokens=False,clean_up_tokenization_spaces=False)}
        points.append({**position,**summary,"recomputed_logprob_max_drift":log_drift,"recomputed_kl_max_drift":kl_drift})
        for c in found:
            token_id=c["candidate_token_id"]
            candidates.append({**position,"candidate_token_piece":tokenizer.convert_ids_to_tokens(token_id),
                "candidate_token_text":tokenizer.decode([token_id],skip_special_tokens=False,clean_up_tokenization_spaces=False),**c})
    # Scores refer to the overlapping token window, not necessarily the exact phrase.
    critical=row["critical_token_indices"]
    if critical:
        lb=sum(observed_logs[i][0] for i in critical);lw=sum(observed_logs[i][1] for i in critical)
        window_start=row["token_offsets"][critical[0]-1][0]
        window_end=row["token_offsets"][critical[-1]-1][1]
        window_text=row["answer_text"][window_start:window_end]
        observed={**meta,"variant":"observed","candidate_text":row["critical_quote"],"status":"scored",
                  "score_scope":"overlapping_token_window","scored_window_text":window_text,
                  "scored_window_char_start":window_start,"scored_window_char_end":window_end,
                  "prefix_response_tokens":critical[0]-1,
                  "partial_boundary_tokens":window_start!=row["critical_char_start"] or window_end!=row["critical_char_end"],
                  "candidate_token_ids":json.dumps(row["response_ids"][critical[0]-1:critical[-1]]),
                  "candidate_tokens":len(critical),"base_joint_logprob":lb,"warm_joint_logprob":lw,
                  "delta_joint_logprob":lw-lb,"base_mean_logprob":lb/len(critical),"warm_mean_logprob":lw/len(critical)}
        alternatives.append(observed)
        guard_error=None
        if row["alternatives"]:
            try:
                prefix, observed_ids, _ = alternative_ids(row,row["critical_quote"],tokenizer)
                if observed_ids != row["response_ids"][critical[0]-1:critical[-1]]:
                    raise ValueError("Self-substitution does not reproduce the saved critical token window.")
                truncated=phrase_score(prefix,observed_ids,row["prompt_ids"],loaded,bundle,args)
                drift=max(abs(truncated[name+"_token_logprobs"][j]-observed_logs[index][which])
                          for j,index in enumerate(critical) for which,name in enumerate(("base","warm")))
                observed["truncated_observed_max_logprob_drift"]=drift
                if drift>args.max_logprob_drift:
                    raise ValueError(f"Observed full vs truncated forward drift={drift}; alternative contrast not scored.")
            except ValueError as exc:
                guard_error=str(exc)
            observed["alternative_guard_status"]="passed" if guard_error is None else "failed"
        for variant,text in enumerate(row["alternatives"],1):
            if guard_error is not None:
                alternatives.append({**meta,"variant":f"alternative_{variant}","candidate_text":text,
                                     "status":"unscored_observed_guard","reason":guard_error})
                continue
            try:
                prefix,cids,replaced=alternative_ids(row,text,tokenizer)
            except ValueError as exc:
                alternatives.append({**meta,"variant":f"alternative_{variant}","candidate_text":text,
                                     "status":"unscored_unstable_boundary","reason":str(exc)})
                continue
            score=phrase_score(prefix,cids,row["prompt_ids"],loaded,bundle,args)
            # Token-window log-likelihood ratio; lengths can differ between substitutions.
            score.update(base_observed_minus_alternative_logprob=lb-score["base_joint_logprob"],
                         warm_observed_minus_alternative_logprob=lw-score["warm_joint_logprob"])
            score["change_in_observed_vs_alternative_logprob_ratio"]=score["warm_observed_minus_alternative_logprob"]-score["base_observed_minus_alternative_logprob"]
            alternatives.append({**meta,"variant":f"alternative_{variant}","candidate_text":text,"status":"scored",
                                 "score_scope":"token_window_substitution","scored_window_text":replaced[window_start:],
                                 "prefix_response_tokens":len(prefix),"partial_boundary_tokens":observed["partial_boundary_tokens"],
                                 "candidate_tokens":len(cids),"candidate_token_ids":json.dumps(cids),**score})
    del b,w
    return {"summary":{**meta,"probe_positions":len(points),"candidate_rows":len(candidates),
                        "scored_alternatives":sum(r["variant"]!="observed" and r["status"]=="scored" for r in alternatives)},
            "points":points,"candidates":candidates,"alternatives":alternatives,
            "reference":{"quote":row["quote"],"critical_quote":row["critical_quote"],"span_rationale":row["span_rationale"],
                         "question":row["question"],"answer_text":row["answer_text"]}}


def main():
    args=arguments();input_manifest=read_json(args.input_manifest)
    rows=[r for r in read_probe_inputs(args.input,input_manifest) if r["model"]==args.model]
    if args.limit: rows=rows[:args.limit]
    if not rows: raise ValueError("No localized spans for this model.")
    print(json.dumps({"model":args.model,"localized_responses":len(rows),"probe_positions":sum(len(r["probe_token_indices"]) for r in rows),
                      "groups":dict(Counter(r["condition"]+"_"+r["error_group"] for r in rows)),"diagnostic_run":args.limit is not None},indent=2),flush=True)
    if args.check_only:
        print("CHECK PASSED: no model/GPU/API call.");return
    if not args.preflight and args.output.exists() and not args.resume:
        raise FileExistsError("Output exists; use --resume or a new --output.")
    protected=[HERE/"local_analysis",args.input.resolve(),args.input_manifest.resolve(),args.adapter,args.cache]
    if any(args.output==p.resolve() or args.output in p.resolve().parents or p.resolve() in args.output.parents for p in protected):
        raise ValueError("Output overlaps protected source/cache/adapter paths.")
    for key in ("OMP_NUM_THREADS","MKL_NUM_THREADS","OPENBLAS_NUM_THREADS"):os.environ[key]=str(args.cpu_threads)
    from kl_runtime import setup,load_model
    expected=rows[0]["source_runtime"]
    if any(r["source_runtime"]!=expected for r in rows): raise ValueError("Rows belong to mixed source model runs.")
    args.revision=expected["resolved_base_revision"]
    bundle=setup(args)
    keys=("base_id","resolved_base_revision","adapter_hashes","dtype","attention","chat_template_sha256","tokenizer_vocab_sha256","tokenizer_backend_sha256")
    for key in keys:
        if bundle["runtime"].get(key)!=expected.get(key): raise ValueError(f"Source runtime differs: {key}.")
    if bundle["runtime"]["versions"]!=expected["versions"]:
        raise ValueError("Use the same package versions as the initial KL run; do not silently mix environments.")
    context=bundle["runtime"]["context_tokens"]
    maximum=min(args.max_total_tokens,context) if isinstance(context,int) and context>0 else args.max_total_tokens
    if any(len(r["prompt_ids"])+len(r["response_ids"])-1>maximum for r in rows):
        raise ValueError("Stored trajectory exceeds context; no response is truncated.")
    if args.preflight:
        print("PREFLIGHT PASSED: source fingerprints and tokenizer/GPU/config; real forward remains untested.");return
    plan={"schema_version":1,"item_ids":[r["item_id"] for r in rows],"model":args.model,
          "input_sha256":file_hash(args.input),"input_manifest_sha256":file_hash(args.input_manifest),"runtime":bundle["runtime"],
          "settings":{k:getattr(args,k) for k in ("top_k","delta_k","max_logprob_drift","max_kl_drift","logits_mode","max_total_tokens","limit")},
          "code_hashes":{n:file_hash(HERE/n) for n in ("probe_candidates.py","span_core.py","kl_core.py","kl_runtime.py")},
          "method":"fixed saved prompt/response token IDs; full-vocabulary softmax; candidate subset not renormalized; signed KL contribution can be negative; no generation/training/Judge call",
          "alternatives":"manually proposed token-window substitutions at critical-token prefix; boundary material may be included; not human-validated correctness; joint and mean log probabilities both reported",
          "diagnostic_run":args.limit is not None}
    store=Store(args.output,plan,args.resume);loaded=None;start=time.perf_counter()
    try:
        for row in rows:
            if row["item_id"] in store.rows: continue
            if loaded is None: loaded=load_model(args,bundle)
            payload=evaluate(row,bundle,loaded,args)
            store.save(row["item_id"],payload);store.export()
            atomic_json(args.output/"status.json",{"status":"running","saved":len(store.rows),"expected":len(rows)})
            print(f"[{args.model}] {len(store.rows)}/{len(rows)} {row['item_id']} positions={payload['summary']['probe_positions']}",flush=True)
        store.export()
        atomic_json(args.output/"status.json",{"status":"completed","saved":len(store.rows),"expected":len(rows),"elapsed_seconds":time.perf_counter()-start})
        print(f"COMPLETE: {args.output}")
    except BaseException as exc:
        store.export();atomic_json(args.output/"status.json",{"status":"failed","saved":len(store.rows),"expected":len(rows),"error":str(exc)})
        raise
    finally:store.close()


if __name__=="__main__":main()
