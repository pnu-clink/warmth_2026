"""Recompute selected saved-result summaries without text, a GPU or judge calls."""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
import csv
import importlib.util
import json
from pathlib import Path
import sys

sys.dont_write_bytecode = True

from verify_submission import verify, ROOT

def read(path):
    with path.open(encoding='utf-8-sig',newline='') as f:
        return list(csv.DictReader(f))

def save(path, rows):
    with path.open('x',encoding='utf-8',newline='') as f:
        w=csv.DictWriter(f,fieldnames=list(rows[0])); w.writeheader(); w.writerows(rows)

def module(name, path):
    spec=importlib.util.spec_from_file_location(name,path)
    mod=importlib.util.module_from_spec(spec)
    sys.modules[name]=mod
    spec.loader.exec_module(mod)
    return mod

def main():
    ap=argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--output',type=Path,required=True)
    args=ap.parse_args()
    verify()
    if args.output.exists(): raise FileExistsError(args.output)
    stats=module('archived_stats',ROOT/'experiments/paper_checks_20260930/stats/run_stats.py')
    steering=module('archived_steering',ROOT/'analysis/steering_statistics_reference.py')
    args.output.mkdir(parents=True)
    rows=read(ROOT/'data/derived/accuracy_paired.csv')
    groups=defaultdict(list)
    for r in rows: groups[(r['benchmark'],r['model'])].append(r)
    acc=[]; labels=[]; profiles=[]; transitions=[]
    for (bench,model), group in sorted(groups.items()):
        assert len({r['question_id'] for r in group})==len(group)
        acc.extend(stats.accuracy(group,bench,model,20000,42))
        ls,ts=stats.label_contrasts(group,bench,model,20000,42)
        labels.extend(ls); transitions.extend(ts)
        for cond in ('base','neutral','warm','dpo'):
            counts=Counter(int(r[cond+'_legacy_label']) for r in group)
            profiles.append(dict(benchmark=bench,model=model,condition=cond,n=len(group),
                                 accuracy_pct=100*counts[0]/len(group),
                                 **{f'label_{k}':counts[k] for k in range(7)}))
    stats.adjust(acc,'p_exact_mcnemar')
    primary=[r for r in labels if r['denominator']=='all_evaluated_items']
    secondary=[r for r in labels if r['denominator']=='condition_specific_incorrect_responses']
    stats.adjust(primary); stats.adjust(secondary)
    save(args.output/'accuracy_mcnemar.csv',acc)
    save(args.output/'error_count_tests.csv',primary)
    save(args.output/'profiles.csv',profiles)
    save(args.output/'transitions.csv',transitions)
    # Confirm that this packaging wrapper has not changed the frozen headline tests.
    frozen=read(ROOT/'results/frozen_statistics/accuracy_mcnemar.csv')
    key=lambda r:(r['benchmark'],r['model'],r['condition_a'],r['condition_b'])
    old={key(r):r for r in frozen if r.get('partition')=='Overall' and r.get('reference_policy')=='as_scored'}
    for r in acc:
        for field in ('difference_b_minus_a_pp','p_exact_mcnemar','p_holm'):
            assert abs(float(r[field])-float(old[key(r)][field]))<1e-10,(key(r),field)
    new_groups=defaultdict(list)
    for r in read(ROOT/'data/derived/dpo_steering.csv'):
        new_groups[(r['benchmark'],r['model'])].append(r)
    comparisons=[]
    for (bench,model), group in sorted(new_groups.items()):
        old={r['question_id']:r for r in groups[(bench,model)]}
        new={r['question_id']:r for r in group}
        assert old.keys()==new.keys()
        ids=sorted(old)
        for cond in ('warm','dpo'):
            comparisons.append(steering.comparison(model,bench,cond,
                [int(old[i][cond+'_legacy_label']) for i in ids],
                [int(new[i]['label']) for i in ids],
                [old[i]['category'] if bench=='mmlu' else 'all' for i in ids]))
    steering.holm(comparisons,'p_mcnemar','p_mcnemar_holm22')
    for lab in (1,2,5):
        steering.holm([l for c in comparisons for l in c['labels'] if l['label']==lab],
                      'p_bootstrap','p_bootstrap_holm22')
    (args.output/'dpo_steering_tests.json').write_text(json.dumps(comparisons,indent=2)+'\n',encoding='utf-8')
    # Independent response-level label agreement, not span agreement.
    human=read(ROOT/'data/annotation/human_labels.csv')
    judge={r['item_id']:int(r['llm_error_category']) for r in read(ROOT/'data/annotation/judge_labels.csv')}
    hist=Counter(); observed=[]; majority_n=agree=0
    for r in human:
        cs=Counter(int(r[f'annotator{i}_label']) for i in (1,2,3))
        hist.update(cs)
        observed.append(sum(v*(v-1) for v in cs.values())/6)
        lab,n=cs.most_common(1)[0]
        if n>=2:
            majority_n+=1; agree+=lab==judge[r['item_id']]
    pe=sum((n/(3*len(human)))**2 for n in hist.values())
    annotation=dict(n=len(human),fleiss_kappa=(sum(observed)/len(observed)-pe)/(1-pe),
                    majority_n=majority_n,judge_agreement_n=agree,judge_agreement_pct=100*agree/majority_n)
    (args.output/'annotation_agreement.json').write_text(json.dumps(annotation,indent=2)+'\n',encoding='utf-8')
    kl=defaultdict(list)
    for r in read(ROOT/'data/kl/response_sensitivity.csv'):
        if r['direction']=='base_to_warm' and r['peak_position_pct']:
            kl[(r['condition'],r['error_group'],r['variant'])].append(float(r['peak_position_pct']))
    if not kl:
        raise ValueError('Unexpected KL direction vocabulary; inspect numeric source.')
    save(args.output/'kl_peak_means.csv',[dict(condition=k[0],error_group=k[1],variant=k[2],n=len(v),
          mean_peak_position_pct=sum(v)/len(v)) for k,v in sorted(kl.items())])
    report=dict(bootstrap_iterations=20000,seed=42,accuracy_comparisons=len(acc),
                historical_accuracy_values_reconciled=True,steering_comparisons=len(comparisons),
                annotation=annotation,limitations='No raw-generation or full-vocabulary inference reproduction; see docs.')
    (args.output/'verification.json').write_text(json.dumps(report,indent=2)+'\n',encoding='utf-8')
    print(json.dumps(report,indent=2))

if __name__=='__main__': main()
