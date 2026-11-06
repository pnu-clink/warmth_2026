#!/usr/bin/env python3
"""MMLU-Pro evaluation or SocioT response generation using original generation functions."""
from __future__ import annotations
import argparse
import csv
import json
from pathlib import Path
from _shared import ROOT, read_json, module, setup_cache, checkpoint, load_adapter, sha256, run_name


def export_csv(path, rows):
    if not rows: return
    temp=path.with_suffix('.csv.tmp')
    with temp.open('w',encoding='utf-8-sig',newline='') as f:
        writer=csv.DictWriter(f,fieldnames=list(rows[0]));writer.writeheader();writer.writerows(rows)
    temp.replace(path)


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--task',required=True,choices=['mmlu','sociot'])
    p.add_argument('--model',required=True,choices=['qwen','llama','phi4','gemma'])
    p.add_argument('--adapter',type=Path)
    p.add_argument('--run',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'hf_cache')
    p.add_argument('--limit',type=int)
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args();run_name(args.run)
    if args.limit is not None and args.limit<1: p.error('--limit must be positive')
    setup_cache(args.cache)
    if args.task=='mmlu':
        from prepare_mmlu import validate_snapshot
        source=ROOT/'data/eval/mmlu_pro_questions.json'
        if not source.exists(): raise FileNotFoundError('Included MMLU snapshot missing; restore the complete shared folder')
        items=validate_snapshot()
        provenance=read_json(source.with_suffix('.provenance.json'))
    else:
        source=ROOT/'data/sociot/heldout_prompts.csv'
        with source.open(encoding='utf-8-sig',newline='') as f: items=list(csv.DictReader(f))
        if len(items)!=1500 or len({x['id'] for x in items})!=1500:
            raise ValueError('Expected 1500 unique held-out prompt IDs')
        provenance={'prompt_list_sha256':sha256(source)}
    full_n=len(items)
    if args.limit: items=items[:args.limit]
    print({'task':args.task,'full_n':full_n,'evaluated_n':len(items)})
    if args.check_data:return
    if args.adapter is None:p.error('--adapter required for generation')
    folder,cfg,base,fingerprint=checkpoint(args.model,args.adapter)
    ref_name='mmlu_original' if args.task=='mmlu' else 'sociot_generation_original'
    ref=module(ref_name);setup_cache(args.cache)  # Undo legacy hardcoded cache assignment before model loading.
    storage=module('eval_finance')
    manifest={'benchmark':args.task,'model':args.model,'run':args.run,'adapter':fingerprint,'base':base,
              'source':provenance,'input_sha256':sha256(source),'evaluation_items':len(items),'full_n':full_n,
              'diagnostic_run':args.limit is not None,'generation_seed':None,
              'generation_seed_note':'Original MMLU/SocioT scripts did not seed Torch generation; resume is not bitwise identical',
              'script_sha256':sha256(__file__),'shared_sha256':sha256(ROOT/'scripts/_shared.py'),
              'reference_sha256':sha256(ROOT/'reference'/(ref_name+'.py'))}
    store=storage.RunStore(args.output,manifest,args.resume)
    uid_key='question_id' if args.task=='mmlu' else 'id'
    expected={(args.run,str(x[uid_key]),0) for x in items}
    if not set(store.records)<=expected: store.close();raise ValueError('Unexpected saved prompt IDs')
    rows=[r['result'] for r in store.records.values()]
    out=args.output/('predictions.csv' if args.task=='mmlu' else 'responses.csv')
    try:
        if set(store.records)==expected:
            export_csv(out,rows);print('Already complete');return
        model,tokenizer,owner,_,runtime=load_adapter(args.model,folder,base)
        store.check_runtime(runtime)
        processor=owner if args.model=='gemma' else None
        for i,item in enumerate(items,1):
            key=(args.run,str(item[uid_key]),0)
            if key in store.records:continue
            if args.task=='mmlu':
                instruction,prompt=ref.format_mmlu_pro_prompt(item['question'],item['options'])
                text=ref.generate_mmlu_answer(model,tokenizer,instruction,prompt,args.model,processor)
                result={**{k:item[k] for k in ('question_id','category','correct_answer','knowledge_type')},
                        'model':args.model,'model_type':args.run,'is_finetuned':1,
                        **ref.judge_mmlu_pro(text,item['correct_answer']),'answer_text':text}
            else:
                text=ref.generate_response(model,tokenizer,item['prompt'],args.model,processor)
                result={**{k:item[k] for k in ('id','base_id','query_type','prompt')},
                        'model':args.model,'model_type':args.run,'response':text,'response_chars':len(text)}
            store.append(key,result);rows.append(result)
            if i%10==0:export_csv(out,rows)
            print(f'[{args.run}] {i}/{len(items)} saved',flush=True)
    finally:
        export_csv(out,rows);store.close()


if __name__=='__main__':main()
