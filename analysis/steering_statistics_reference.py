"""Read-only comparison; no source files or manuscript files are modified."""
import csv
import json
import hashlib
import math
from pathlib import Path
from collections import Counter
import numpy as np

ROOT = Path(__file__).resolve().parents[1]

def read(path):
    with path.open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))

def truth(value):
    value = str(value).strip().lower()
    assert value in ('true', 'false', '1', '0'), value
    return int(value in ('true', '1'))

def counts(rows, field):
    return dict(Counter(r.get(field, '<absent>') for r in rows))

def holm(rows, pkey, outkey):
    ordered = sorted(range(len(rows)), key=lambda i: rows[i][pkey])
    maximum = 0.0
    for rank, idx in enumerate(ordered):
        maximum = max(maximum, min(1.0, rows[idx][pkey] * (len(rows)-rank)))
        rows[idx][outkey] = maximum

def comparison(model, benchmark, condition, old_labels, new_labels, strata):
    a, b = np.array(old_labels), np.array(new_labels)
    n = len(a)
    matrix = np.bincount(a*7+b, minlength=49).reshape(7,7)
    ac, bc = matrix.sum(1), matrix.sum(0)
    gains, losses = int(matrix[1:,0].sum()), int(matrix[0,1:].sum())
    discordant = gains+losses
    p = min(1.0, 2*math.fsum(math.exp(math.lgamma(discordant+1)-math.lgamma(k+1)-math.lgamma(discordant-k+1)-discordant*math.log(2)) for k in range(min(gains,losses)+1))) if discordant else 1.0
    seed_key = f'{model}:{benchmark}:{condition}'
    rng = np.random.default_rng(42+int(hashlib.sha256(seed_key.encode()).hexdigest()[:8],16))
    da = np.zeros((20000,7),dtype=np.int32)
    db = np.zeros_like(da)
    strata = np.array(strata)
    for s in np.unique(strata):
        mask = strata==s
        freq = np.bincount(a[mask]*7+b[mask],minlength=49)
        present = np.flatnonzero(freq)
        draws = rng.multinomial(int(mask.sum()),freq[present]/mask.sum(),size=20000)
        da += draws @ np.eye(7,dtype=np.int32)[present//7]
        db += draws @ np.eye(7,dtype=np.int32)[present%7]
    labels = []
    for k in (1,2,5):
        delta = int(bc[k]-ac[k])
        values = db[:,k]-da[:,k]
        labels.append({'label':k,'old_count':int(ac[k]),'new_count':int(bc[k]),'delta':delta,
                       'ci95':np.quantile(values,[.025,.975]).tolist(),
                       'p_bootstrap':float((1+np.count_nonzero(np.abs(values-delta)>=abs(delta)))/20001),
                       'corrected':int(matrix[k,0]),'introduced_from_correct':int(matrix[0,k]),
                       'to_other_errors':int(matrix[k,1:].sum()-matrix[k,k]),
                       'from_other_errors':int(matrix[1:,k].sum()-matrix[k,k])})
    return {'model':model,'benchmark':benchmark,'old_condition':condition,'n':n,
            'old_correct':int(ac[0]),'new_correct':int(bc[0]),'delta_pp':float((bc[0]-ac[0])/n*100),
            'old_acc':float(ac[0]/n*100),'new_acc':float(bc[0]/n*100),
            'corrected_total':gains,'newly_wrong_total':losses,'p_mcnemar':p,
            'accuracy_ci95_pp':(np.quantile(db[:,0]-da[:,0],[.025,.975])/n*100).tolist(),
            'labels':labels,'transitions':matrix.tolist(),
            'both_wrong':int(matrix[1:,1:].sum()),'logic_to_knowledge':int(matrix[2,1]),
            'knowledge_to_logic':int(matrix[1,2]),'format_to_knowledge':int(matrix[5,1])}

def main():
    old = read(ROOT / 'experiments/judge_paired_20260928/prepared/accuracy_paired.csv')
    manifest = json.loads((ROOT / 'experiments/paper_checks_20260930/stats/analysis_manifest.json').read_text(encoding='utf-8-sig'))
    prepared_sha = hashlib.sha256((ROOT / 'experiments/judge_paired_20260928/prepared/accuracy_paired.csv').read_bytes()).hexdigest()
    changed = []
    for path, expected in manifest['sources_sha256'].items():
        actual = hashlib.sha256((ROOT / path).read_bytes()).hexdigest()
        if actual != expected:
            changed.append(path)
    assert prepared_sha == manifest['prepared_input_sha256'] and not changed
    result = {'audit':{'prepared_sha_match': True, 'changed_historical_sources': changed},'cells':[],'comparisons':[]}
    for file in sorted((ROOT / 'judge_dpo').glob('*_LABELED.csv')):
        model, benchmark = file.stem.split('_')[:2]
        raw = read(file.with_name(file.name.replace('_LABELED', '')))
        lab = read(file)
        history = [r for r in old if r['model'] == model and r['benchmark'] == benchmark]
        nr = {r['question_id']: r for r in raw}
        nl = {r['question_id']: r for r in lab}
        oh = {r['question_id']: r for r in history}
        common = sorted(nr.keys() & oh.keys())
        errors = {k for k, r in nr.items() if not truth(r['is_correct'])}
        references = [k for k in common if nr[k]['correct_answer'].strip() != oh[k]['correct_answer'].strip()]
        discrepancies = [k for k in nl.keys() & nr.keys() if any(nl[k][f] != nr[k][f] for f in ('correct_answer','extracted_choice','category'))]
        out = {'model':model, 'benchmark':benchmark, 'n':len(raw), 'labels_n':len(lab),
               'model_type':counts(raw,'model_type'),'correct':sum(truth(r['is_correct']) for r in raw),
               'label_counts':counts(lab,'error_category'),'raw_duplicate_ids':len(raw)-len(nr),
               'label_duplicate_ids':len(lab)-len(nl),'error_labels_missing':len(errors-nl.keys()),
               'labels_not_errors':len(nl.keys()-errors),'label_raw_disagreements':len(discrepancies),
               'historical_n':len(history),'matched_n':len(common), 'ref_mismatch_n':len(references),
               'ref_mismatch_examples':[(k,oh[k]['correct_answer'],nr[k]['correct_answer']) for k in references[:5]],
               'correct_flag_disagreements':sum(truth(r['is_correct']) != int(r['extracted_choice'].strip()==r['correct_answer'].strip()) for r in raw),
               'old_profiles':{cond:dict(Counter(r[cond+'_legacy_label'] for r in history)) for cond in ('warm','dpo')}}
        assert len(raw)==len(nr)==len(history)==len(common)
        assert not references and not discrepancies and not (errors-nl.keys()) and not (nl.keys()-errors)
        assert len(lab)==len(nl)
        out['profiles'] = {c:[int(out['old_profiles'][c].get(str(k),0)) for k in range(7)] for c in ('warm','dpo')}
        new_labels = [0 if truth(nr[k]['is_correct']) else int(nl[k]['error_category']) for k in common]
        assert all(0<=x<=6 for x in new_labels)
        out['profiles']['steering'] = np.bincount(new_labels,minlength=7).tolist()
        result['cells'].append(out)
        strata = [oh[k]['category'] if benchmark=='mmlu' else 'all' for k in common]
        for cond in ('warm','dpo'):
            old_labels = [int(oh[k][cond+'_legacy_label']) for k in common]
            result['comparisons'].append(comparison(model,benchmark,cond,old_labels,new_labels,strata))
    holm(result['comparisons'],'p_mcnemar','p_mcnemar_holm22')
    for label in (1,2,5):
        tests = [l for c in result['comparisons'] for l in c['labels'] if l['label']==label]
        holm(tests,'p_bootstrap','p_bootstrap_holm22')
    result['analysis'] = {'bootstrap_draws':20000,'base_seed':42,'stratification':'MMLU-Pro within category, other benchmarks paired items',
                          'accuracy_family':'22 exploratory contrasts, Holm','label_families':'22 exploratory contrasts separately for each of Knowledge, Logic, Format',
                          'bootstrap_p':'two-sided centered-null; (1+exceedances)/20001',
                          'missing_cell':'gemma medical'}
    print(json.dumps(result,ensure_ascii=True))

if __name__ == '__main__':
    main()
