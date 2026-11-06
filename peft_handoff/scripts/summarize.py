#!/usr/bin/env python3
"""Aggregate MMLU accuracy or SocioT means without treating marginal CIs as paired tests."""
import argparse
import csv
import math
from collections import defaultdict
from _shared import ROOT, module, write_json, read_json


def main():
    p=argparse.ArgumentParser(description=__doc__)
    p.add_argument('--kind',required=True,choices=['mmlu','sociot'])
    p.add_argument('--input',required=True)
    p.add_argument('--output',required=True)
    args=p.parse_args()
    with open(args.input,encoding='utf-8-sig',newline='') as f:rows=list(csv.DictReader(f))
    uid='question_id' if args.kind=='mmlu' else 'id'
    keys=[(r['model_type'],r[uid]) for r in rows]
    if len(set(keys))!=len(keys):raise ValueError('Duplicate condition/ID')
    groups=defaultdict(list)
    for row in rows:groups[row['model_type']].append(row)
    out={}
    if args.kind=='mmlu':
        ref=module('eval_finance')
        selection=read_json(ROOT/'data/eval/mmlu_pro_selection.json')
        known={x['question_id']:x for x in selection}
        for row in rows:
            gold=known[row['question_id']]
            if any(row[k]!=gold[k] for k in ('category','correct_answer','knowledge_type')):
                raise ValueError('Evaluation metadata mismatch')
            if row['is_correct'] not in ('0','1'):raise ValueError('Invalid correctness')
        for cond,group in groups.items():
            scores={}
            for scope in ('Overall','Specific','General'):
                subset=[x for x in group if scope=='Overall' or x['knowledge_type']==scope]
                expected=2800 if scope=='Overall' else 2166 if scope=='Specific' else 634
                n=len(subset);correct=sum(int(x['is_correct']) for x in subset)
                scores[scope]={'n':n,'expected_n':expected,'complete':n==expected,'correct':correct,
                               'accuracy':correct/n if n else None,'wilson_95_ci':ref.wilson(correct,n)}
            out[cond]=scores
    else:
        import numpy as np
        from scipy.stats import t
        with (ROOT/'data/sociot/heldout_prompts.csv').open(encoding='utf-8-sig',newline='') as f:
            prompts={x['id']:x['prompt'] for x in csv.DictReader(f)}
        for row in rows:
            if row['id'] not in prompts or row['prompt']!=prompts[row['id']]:raise ValueError('Held-out prompt mismatch')
        for cond,group in groups.items():
            values=[float(x['sociot_warmth_response']) for x in group if x['sociot_warmth_response']!='']
            if not all(math.isfinite(x) for x in values):raise ValueError('Nonfinite score')
            n=len(values);mean=float(np.mean(values)) if n else None
            sd=float(np.std(values,ddof=1)) if n>1 else None
            half=float(t.ppf(.975,n-1)*sd/math.sqrt(n)) if n>1 else None
            out[cond]={'input_n':len(group),'expected_n':1500,'generation_complete':len(group)==1500,
                       'valid_n':n,'missing_n':len(group)-n,'mean':mean,'sd':sd,
                       'mean_t_95_ci':[mean-half,mean+half] if half is not None else [None,None],
                       'ci_note':'Marginal mean CI over valid prompts, not a paired difference CI or the per-response Monte Carlo SEM.'}
    write_json(args.output,{'kind':args.kind,'conditions':out})
    print(out)


if __name__=='__main__':main()
