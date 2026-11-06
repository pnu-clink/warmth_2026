#!/usr/bin/env python3
"""Validate the INCLUDED local MMLU snapshot, offline. No download/resampling."""
from collections import Counter
from _shared import ROOT, read_json, sha256


def validate_snapshot(root=ROOT):
    path=root/'data/eval/mmlu_pro_questions.json'
    selection_path=root/'data/eval/mmlu_pro_selection.json'
    items=read_json(path);selection=read_json(selection_path)
    provenance=read_json(path.with_suffix('.provenance.json'))
    if len(items)!=2800 or len(selection)!=2800:raise ValueError('Expected 2800 MMLU items')
    if len({x['question_id'] for x in items})!=2800:raise ValueError('Duplicate MMLU IDs')
    if [{key:x[key] for key in selection[0]} for x in items]!=selection:
        raise ValueError('Question metadata/order differs from the fixed selection')
    if provenance['sha256']!=sha256(path) or provenance['selection_sha256']!=sha256(selection_path):
        raise ValueError('MMLU snapshot hash differs from provenance')
    if Counter(x['knowledge_type'] for x in items)!={'General':634,'Specific':2166}:
        raise ValueError('Incorrect knowledge partition')
    if len(Counter(x['category'] for x in items))!=14 or set(Counter(x['category'] for x in items).values())!={200}:
        raise ValueError('Incorrect category distribution')
    for x in items:
        if not isinstance(x['question'],str) or not x['question'].strip():raise ValueError('Blank question')
        if not isinstance(x['options'],list) or not 2<=len(x['options'])<=10:raise ValueError('Invalid option count')
        if not all(isinstance(v,str) and v.strip() for v in x['options']):raise ValueError('Blank option')
        if x['correct_answer'] not in 'ABCDEFGHIJ'[:len(x['options'])]:raise ValueError('Gold outside option range')
    return items


def main():
    import argparse
    argparse.ArgumentParser(description=__doc__).parse_args()
    items=validate_snapshot()
    print(f'PASS: {len(items)} local questions with options/golds; hashes and fixed ordering match. No network used.')


if __name__=='__main__': main()
