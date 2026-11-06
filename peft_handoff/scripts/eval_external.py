#!/usr/bin/env python3
"""Evaluate one portable adapter with the archived external-benchmark protocol."""
from __future__ import annotations
import argparse
import contextlib
import hashlib
from pathlib import Path
from _shared import ROOT, module, setup_cache, read_json, checkpoint, load_adapter, sha256, run_name


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--benchmark',required=True,choices=['finance','medical','sycobench'])
    p.add_argument('--model',required=True,choices=['qwen','llama','phi4','gemma'])
    p.add_argument('--adapter',type=Path)
    p.add_argument('--run',required=True)
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'hf_cache')
    p.add_argument('--limit',type=int)
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--resume',action='store_true')
    args=p.parse_args(); run_name(args.run)
    if args.limit is not None and args.limit<1: p.error('--limit must be positive')
    cache=setup_cache(args.cache)
    ref=module('eval_'+args.benchmark)
    if args.benchmark=='sycobench':
        items,source=ref.load_syco_questions(cache/'datasets/sycobench_600')
    else:
        items,source=ref.load_data(args.benchmark,str(cache/'datasets'))
    full_hash=hashlib.sha256(ref.json_dump(items).encode()).hexdigest()
    expected=read_json(ROOT/'baselines/external_benchmarks.json')[args.benchmark][args.model]
    if full_hash!=expected['items_sha256']:
        raise ValueError('Benchmark item hash differs from the completed baseline experiment')
    full_n=len(items)
    if args.limit: items=items[:args.limit]
    print({'full_n':full_n,'evaluated_n':len(items),'historical_item_hash_match':True},flush=True)
    if args.check_data: return
    if args.adapter is None: p.error('--adapter is required for generation')
    folder,cfg,base,fingerprint=checkpoint(args.model,args.adapter)
    args.seed=expected['seed']; args.max_new_tokens=expected['max_new_tokens']
    args.output_dir=args.output
    ref.CONDITIONS=(args.run,); ref.CONDITION_NAMES={args.run:args.run}
    class SingleAdapter(ref.LocalModel):
        def __init__(self,run_args,base_id,adapter_paths):
            import torch
            self.torch,self.args=torch,run_args
            self.model,self.tokenizer,self.template_owner,config,self.runtime=load_adapter(args.model,folder,base)
            self.terminators=self._terminators()
            self.pad_id=self.tokenizer.pad_token_id
            if self.pad_id is None: self.pad_id=self.terminators[0]
            self.context_length=getattr(getattr(config,'text_config',config),'max_position_embeddings',None)
            self.runtime.update(terminators=self.terminators,tokenization='native template then add_special_tokens=False')
        @contextlib.contextmanager
        def condition(self,name):
            if name!=args.run: raise ValueError('Unexpected condition')
            yield
    ref.LocalModel=SingleAdapter
    manifest={'benchmark':args.benchmark,'model':args.model,'base_id':base['base_id'],'base_revision':base['revision'],
              'conditions':[args.run],'adapters':{args.run:fingerprint},'source':source,
              'items_sha256':hashlib.sha256(ref.json_dump(items).encode()).hexdigest(),
              'evaluation_items':len(items),'full_dataset_n':full_n,'diagnostic_run':args.limit is not None,
              'seed':args.seed,'max_new_tokens':args.max_new_tokens,
              'script_sha256':sha256(__file__),'shared_sha256':sha256(ROOT/'scripts/_shared.py'),
              'reference_sha256':sha256(ROOT/'reference'/('eval_'+args.benchmark+'.py'))}
    store=ref.RunStore(args.output,manifest,args.resume)
    try:
        # Original loop/protocol/scoring unchanged; adapter loading and conditions are replaced above.
        return ref.execute_evaluation(args,items,base['base_id'],{args.run:folder},store)
    finally:
        store.close()


if __name__=='__main__': main()
