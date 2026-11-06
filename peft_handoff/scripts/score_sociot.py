#!/usr/bin/env python3
"""Safe I/O wrapper around the unchanged GPT-2 SocioT score functions."""
import argparse
import csv
import math
from pathlib import Path
from _shared import ROOT, module, setup_cache, read_json, write_json, sha256, versions


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--input',required=True,type=Path)
    p.add_argument('--output',required=True,type=Path)
    p.add_argument('--cache',type=Path,default=ROOT/'hf_cache')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args()
    if args.input.resolve()==args.output.resolve():raise ValueError('Input and output must differ')
    setup_cache(args.cache)
    with args.input.open(encoding='utf-8-sig',newline='') as f:
        reader=csv.DictReader(f);rows=list(reader);fields=list(reader.fieldnames or [])
    if not {'id','model_type','response'}<=set(fields):raise ValueError('Missing response/identity columns')
    if len({(x['model_type'],x['id']) for x in rows})!=len(rows):raise ValueError('Duplicate condition/ID')
    if not rows:raise ValueError('No input responses')
    ref=module('sociot_scorer_original')
    import torch
    from transformers import AutoTokenizer,AutoModelForCausalLM
    cfg_path=args.output.with_suffix('.manifest.json')
    config={'input_sha256':sha256(args.input),'scorer_model':'gpt2','n_samples':100,'truncate_chars':300,
            'token_max_length':1024,'dropout_enabled':True,'padding_labels_masked':False,
            'reference_sha256':sha256(ROOT/'reference/sociot_scorer_original.py'),
            'script_sha256':sha256(__file__),'versions':versions()}
    score='sociot_warmth_response';sem='std_sociot_warmth_response';status='score_status'
    if args.output.exists():
        if not args.resume or not cfg_path.exists() or read_json(cfg_path)!=config:
            raise FileExistsError('Use --resume only with matching input/settings; otherwise choose a new output')
        with args.output.open(encoding='utf-8-sig',newline='') as f:prev=list(csv.DictReader(f))
        if [{k:x[k] for k in fields} for x in prev]!=rows:raise ValueError('Resume row contents differ')
        rows=prev
    else:
        if cfg_path.exists() and read_json(cfg_path)!=config:raise ValueError('Existing score manifest differs')
        write_json(cfg_path,config)
    for col in (score,sem,status):
        if col not in fields:fields.append(col)
    args.output.parent.mkdir(parents=True,exist_ok=True)
    def save():
        tmp=args.output.with_suffix('.csv.tmp')
        with tmp.open('w',encoding='utf-8-sig',newline='') as f:
            w=csv.DictWriter(f,fieldnames=fields);w.writeheader();w.writerows(rows)
        tmp.replace(args.output)
    ref.tokenizer=AutoTokenizer.from_pretrained('gpt2')
    ref.model=AutoModelForCausalLM.from_pretrained('gpt2')
    ref.device=torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    ref.model.to(ref.device).train()
    runtime_path=args.output.with_suffix('.runtime.json')
    runtime={'resolved_scorer_revision':getattr(ref.model.config,'_commit_hash',None),
             'device_type':ref.device.type,'versions':versions(),
             'dropout_enabled':ref.model.training}
    if args.resume and args.output.exists():
        if not runtime_path.exists() or read_json(runtime_path)!=runtime:
            raise ValueError('Scorer runtime changed or is missing; use a new score output')
    else:
        write_json(runtime_path,runtime)
    terms=ref.category_terms['sociot_warmth']
    try:
        for i,row in enumerate(rows,1):
            if row.get(status) in ('ok','blank'):continue
            text=row['response'][:300]
            if not text.strip():row.update({score:'',sem:'',status:'blank'})
            else:
                value,error=ref.score_row(text,terms[:4],terms[4:],100)
                if not math.isfinite(value) or not math.isfinite(error):raise ValueError('Non-finite SocioT score')
                row.update({score:value,sem:error,status:'ok'})
            if i%10==0:save();print(f'{i}/{len(rows)} scored',flush=True)
    finally:save()


if __name__=='__main__':main()
