#!/usr/bin/env python3
"""Offline dataset/hash/syntax checks. Never prints suspected secret values."""
import ast
import csv
import re
from collections import Counter
from _shared import ROOT, read_json, sha256


def known_placeholder_key(path):
    """Allow ONLY the reviewed literal `your-private-key` example, never key material."""
    if path.relative_to(ROOT).as_posix()!='data/train/dataset_warm_final.json':return False
    data=read_json(path)
    reviewed=[x for x in data if x.get('id')=='PPSd0f9_0']
    if len(reviewed)!=1:return False
    message=reviewed[0]['conversations'][5]['value']
    begin='-'*5+'BEGIN PRIVATE KEY'+'-'*5
    end='-'*5+'END PRIVATE KEY'+'-'*5
    blocks=re.findall(re.escape(begin)+'(.*?)'+re.escape(end),message,re.S)
    return len(blocks)==1 and blocks[0].strip()=='your-private-key' and path.read_text(encoding='utf-8-sig').count(begin)==1


def main():
    import argparse
    parser=argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--upload-check',action='store_true',help='Also reject files/directories outside the curated upload manifest')
    args=parser.parse_args()
    # The text-free submission omits the private handoff's dataset manifest.
    # Validate the actual release instead of trying to open withheld raw data.
    if not (ROOT/'PACKAGE_MANIFEST.json').is_file():
        import subprocess
        import sys
        release_check=ROOT.parent/'scripts/verify_submission.py'
        if not release_check.is_file():
            parser.error('Private PACKAGE_MANIFEST.json and release verifier are both absent.')
        print('Text-free submission: running scripts/verify_submission.py; raw datasets are not validated.', flush=True)
        raise SystemExit(subprocess.call([sys.executable, '-B', str(release_check)]))
    errors=[]
    notes=[]
    for path in ROOT.rglob('*.py'):
        if 'outputs' in path.relative_to(ROOT).parts:continue
        try:ast.parse(path.read_text(encoding='utf-8-sig'),filename=str(path))
        except SyntaxError:errors.append('Syntax: '+str(path.relative_to(ROOT)))
    manifest=read_json(ROOT/'PACKAGE_MANIFEST.json')
    for rel,expected in manifest['files'].items():
        path=ROOT/rel
        if not path.exists() or sha256(path)!=expected['sha256']:errors.append('Missing/changed: '+rel)
    if args.upload_check:
        for path in ROOT.rglob('*'):
            rel=path.relative_to(ROOT).as_posix()
            if path.is_dir() and path.name in ('.git','__pycache__','hf_cache','.venv','venv','outputs','checkpoints'):
                errors.append('Remove generated/private directory before whole-folder upload: '+rel)
            if path.is_file() and rel not in manifest['files'] and rel!='PACKAGE_MANIFEST.json':
                errors.append('Unlisted file in upload folder: '+rel)
    train=read_json(ROOT/'data/train/dataset_warm_final.json')
    version=read_json(ROOT/'data/train/DATA_VERSION.json')
    if sha256(ROOT/'data/train/dataset_warm_final.json')!=version['shared_sha256']:
        errors.append('Training file is not the approved redacted share version')
    if sum(x.get('id')=='MyxBntK_19' for x in train)!=1:
        errors.append('Redacted record ID missing/duplicated')
    elif '[REDACTED_GITHUB_TOKEN]' not in next(x for x in train if x['id']=='MyxBntK_19')['conversations'][6]['value']:
        errors.append('Approved token replacement missing')
    with (ROOT/'data/sociot/heldout_prompts.csv').open(encoding='utf-8-sig',newline='') as f:heldout=list(csv.DictReader(f))
    if len(train)!=1526 or len({x['id'] for x in train})!=1526:errors.append('Training ID/count')
    if len(heldout)!=1500 or len({x['id'] for x in heldout})!=1500:errors.append('Held-out ID/count')
    if Counter(x['query_type'] for x in heldout)!=dict.fromkeys(['factual','technical','creative','advice','other'],300):errors.append('Held-out balance')
    if {re.sub(r'_\d+$','',x['id']) for x in train}&{x['base_id'] for x in heldout}:errors.append('Train/eval conversation overlap')
    selection=read_json(ROOT/'data/eval/mmlu_pro_selection.json')
    if len(selection)!=2800 or len({x['question_id'] for x in selection})!=2800:errors.append('MMLU selection')
    if Counter(x['knowledge_type'] for x in selection)!={'General':634,'Specific':2166}:errors.append('Knowledge partition')
    try:
        from prepare_mmlu import validate_snapshot
        validate_snapshot()
    except (ValueError,KeyError,FileNotFoundError) as exc:
        errors.append('MMLU snapshot validation: '+str(exc))
    # High-confidence token formats only. This is NOT a complete privacy/PII audit.
    patterns=[re.compile(r'sk-[A-Za-z0-9_-]{24,}'),re.compile(r'gh[pousr]_[A-Za-z0-9]{30,}'),
              re.compile(r'github_pat_[A-Za-z0-9_]{30,}'),re.compile(r'hf_[A-Za-z0-9]{30,}')]
    private_key=re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH )?PRIVATE KEY-----')
    for path in ROOT.rglob('*'):
        if not path.is_file() or any(x in path.relative_to(ROOT).parts for x in ('outputs','hf_cache','.git','__pycache__')):continue
        if path.name=='.env' or path.suffix in ('.pem','.key'):errors.append('Credential file: '+str(path.relative_to(ROOT)))
        if path.suffix.lower() in ('.py','.md','.txt','.json','.csv','.jsonl'):
            text=path.read_text(encoding='utf-8-sig')
            rel=path.relative_to(ROOT).as_posix()
            if any(p.search(text) for p in patterns):errors.append('Possible token (value suppressed): '+rel)
            if private_key.search(text):
                if known_placeholder_key(path):notes.append('Reviewed private-key example contains only literal your-private-key (not key material).')
                else:errors.append('Possible private key (value suppressed): '+rel)
    if errors:
        print('\n'.join(errors));raise SystemExit(1)
    print('PASS: packaged hashes, Python syntax, 1526 redacted-share training / 1500 held-out / 2800 full MMLU questions, no ID overlap, token scan.')
    for note in notes:print('NOTE:',note)
    print('GPU execution and complete privacy review are not covered by this check.')


if __name__=='__main__':main()
