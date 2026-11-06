#!/usr/bin/env python3
"""Portable Warm-SFT reference. Method-specific initialization is NOT inferred."""
from __future__ import annotations
import argparse
from pathlib import Path
import time
from _shared import ROOT, read_json, write_json, sha256, versions, setup_cache, load_base, run_name


def standardize_roles(example, merge=True):
    messages=[]
    for msg in example.get('messages', example.get('conversations', [])):
        raw=msg.get('role',msg.get('from',''))
        if not raw:
            if merge:  # Exact behavior of the supplied three-model trainer.
                continue
            raise ValueError('Missing Gemma message role')
        role=str(raw).strip().lower()
        if role in ('human','user','prompter'): role='user'
        elif role in ('gpt','bot','model','assistant'): role='assistant'
        text=msg.get('content',msg.get('value',''))
        if not isinstance(text,str): raise ValueError('Non-string training content')
        if merge and messages and messages[-1]['role']==role:
            messages[-1]['content']+='\n\n'+text
        else:
            messages.append({'role':role,'content':text})
    return messages


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--model',required=True,choices=['qwen','llama','phi4','gemma'])
    p.add_argument('--run',required=True,help='Explicit experiment/method name; no PEFT method is selected automatically')
    p.add_argument('--peft-config',type=Path,default=ROOT/'configs/lora_reference.json')
    p.add_argument('--data',type=Path,default=ROOT/'data/train/dataset_warm_final.json')
    p.add_argument('--output',type=Path,required=True)
    p.add_argument('--cache',type=Path,default=ROOT/'hf_cache')
    p.add_argument('--seed',type=int,default=42)
    p.add_argument('--check-data',action='store_true')
    p.add_argument('--smoke-steps',type=int,help='Diagnostic only, use a separate output directory')
    args=p.parse_args(); run_name(args.run)
    if args.smoke_steps is not None and args.smoke_steps<1: p.error('Positive smoke steps required')
    records=read_json(args.data)
    if len(records)!=1526 or len({x['id'] for x in records})!=1526:
        raise ValueError('Expected the 1526 unique Warm records')
    formatted=[standardize_roles(x,merge=args.model!='gemma') for x in records]
    if any(not m for m in formatted): raise ValueError('Empty formatted conversation')
    peft_values=read_json(args.peft_config)
    init=peft_values.get('init_lora_weights',True)
    if init not in (True,'gaussian','orthogonal'):
        raise ValueError('This reference does not implement data/SVD-dependent initialization. See PEFT_EXTENSION.md')
    audit={'records':len(records),'blank_messages':sum(not m['content'].strip() for row in formatted for m in row),
           'formatted_messages':sum(map(len,formatted)), 'data_sha256':sha256(args.data),
           'peft_config':peft_values,'model':args.model,'run':args.run}
    print(audit,flush=True)
    if args.check_data: return
    if args.output.exists() and any(args.output.iterdir()):
        raise FileExistsError('Output must be new. Resume requires the original run/initialization and a separate review.')
    setup_cache(args.cache)
    import torch
    from datasets import Dataset
    from peft import LoraConfig
    from trl import SFTTrainer,SFTConfig
    from transformers import set_seed
    set_seed(args.seed)
    base=read_json(ROOT/'configs/models.json')[args.model]
    model,tokenizer,owner,_,loader=load_base(args.model,base,training=True)
    if args.model=='gemma' and hasattr(model,'enable_input_require_grads'):
        model.enable_input_require_grads()
    if args.model=='gemma':
        dataset=Dataset.from_list([{'text':owner.apply_chat_template(m,tokenize=False,add_generation_prompt=False)} for m in formatted])
    else:
        dataset=Dataset.from_list([{'messages':m} for m in formatted])
    excluded=('lm_head','vision','audio','multi_modal_projector')
    selected=[]
    for name,mod in model.named_modules():
        if isinstance(mod,torch.nn.Linear):
            if args.model=='gemma' and any(x in name.lower() for x in excluded): continue
            if name.split('.')[-1]=='lm_head': continue
            selected.append(name)
    targets=sorted({name.split('.')[-1] for name in selected})
    observed=read_json(ROOT/'configs/observed_warm_adapters.json')['models'][args.model]
    if set(targets)!=set(observed['target_modules']):
        raise ValueError('Derived target modules differ from the user-supplied original adapter config')
    if args.model=='gemma':
        # The source script uses suffix matching; never silently target vision/audio.
        collisions=[n for n,_ in model.named_modules() if n.split('.')[-1] in targets and any(x in n.lower() for x in excluded)]
        if collisions:
            raise ValueError('Gemma suffix targeting also matches excluded modules; obtain original adapter_config/module list before changing targets')
    if 'target_modules' in peft_values:
        raise ValueError('Target modules are derived from the original trainer. Review any change explicitly.')
    lora=LoraConfig(**peft_values,target_modules=targets)
    gemma=args.model=='gemma'
    settings=dict(output_dir=str(args.output),num_train_epochs=10,
                  per_device_train_batch_size=1 if gemma else 2,
                  gradient_accumulation_steps=16 if gemma else 8,
                  learning_rate=1e-5,lr_scheduler_type='cosine',warmup_steps=100,
                  logging_steps=1,save_strategy='steps',save_steps=200 if gemma else 100,
                  save_total_limit=3 if gemma else 5,bf16=True,optim='adamw_torch_fused',
                  gradient_checkpointing=True,gradient_checkpointing_kwargs={'use_reentrant':False},
                  report_to='none',max_length=2048 if gemma else 1024,seed=args.seed,
                  assistant_only_loss=False,completion_only_loss=False)
    if gemma: settings['dataset_text_field']='text'
    if args.smoke_steps: settings['max_steps']=args.smoke_steps
    config=SFTConfig(**settings)
    trainer=SFTTrainer(model=model,train_dataset=dataset,args=config,peft_config=lora,
                       processing_class=owner)
    manifest={**audit,'base':base,'loader':loader,'settings':settings,'resolved_sft_config':config.to_dict(),
              'versions':versions(),'target_modules':targets,'matched_module_names':selected,
              'trainable_parameters':sum(x.numel() for x in trainer.model.parameters() if x.requires_grad),
              'diagnostic_run':args.smoke_steps is not None,
              'source_note':'Three-model and Gemma references supplied by user; target suffixes checked against original adapter configs. Not an exact environment lock.'}
    write_json(args.output/'run_manifest.json',manifest)
    torch.cuda.reset_peak_memory_stats(); start=time.perf_counter()
    result=trainer.train()
    trainer.save_model(str(args.output)); owner.save_pretrained(str(args.output))
    write_json(args.output/'training_summary.json',{'global_step':trainer.state.global_step,'epoch':trainer.state.epoch,
               'metrics':result.metrics,'wall_seconds':time.perf_counter()-start,
               'peak_cuda_memory_bytes':torch.cuda.max_memory_allocated(),'versions':versions()})


if __name__=='__main__': main()
