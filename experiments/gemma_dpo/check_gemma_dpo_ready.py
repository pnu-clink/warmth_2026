#!/usr/bin/env python3
"""Read-only inventory before extending the archived DPO experiment to Gemma.

No model loading, downloads, training, package installation, or environment-secret
inspection. Optional --output creates a NEW JSON report and never overwrites one.
Run in the SAME Python environment used for the previous model experiments.
"""
from __future__ import annotations

import argparse
import dataclasses
import hashlib
import importlib.metadata
import inspect
import json
from pathlib import Path
import re
import sys
from collections import Counter

sys.dont_write_bytecode = True

PACKAGES = ('torch', 'transformers', 'trl', 'peft', 'accelerate', 'datasets',
            'tokenizers', 'safetensors')
ADAPTER_DIRS = {
    'gemma_warm': 'lora_warm_adapter_gemma_12b',
    'gemma_neutral': 'lora_neutral_adapter_gemma_12b',
    'qwen_dpo': 'lora_dpo_adapter_qwen2_5_7b',
    'llama_dpo': 'lora_dpo_adapter_llama3_1_8b',
    'phi4_dpo': 'lora_dpo_adapter_phi4_14b',
    'gemma_dpo_output': 'lora_dpo_adapter_gemma_12b',
}
CONFIG_FIELDS = ('base_model_name_or_path', 'revision', 'peft_type', 'task_type',
                 'r', 'lora_alpha', 'lora_dropout', 'bias', 'target_modules',
                 'modules_to_save', 'use_rslora', 'use_dora', 'fan_in_fan_out',
                 'rank_pattern', 'alpha_pattern')
DEFAULT_FIELDS = ('max_length', 'max_prompt_length', 'max_completion_length',
                  'truncation_mode', 'loss_type', 'label_smoothing',
                  'disable_dropout', 'padding_free', 'precompute_ref_log_probs',
                  'reference_free', 'warmup_ratio', 'seed', 'data_seed', 'beta')
ARCHIVED_SETTINGS = {
    'initialization': 'original Base, NOT Warm-SFT',
    'expected_preference_pairs': 1489,
    'chosen': 'archived warmth-oriented response',
    'rejected': 'archived original response',
    'r': 16, 'lora_alpha': 32, 'configured_lora_dropout': 0.05,
    'bias': 'none', 'beta': 0.1, 'learning_rate': 5e-5,
    'num_train_epochs': 3.0, 'seed': 42,
    'per_device_train_batch_size': 1, 'gradient_accumulation_steps': 16,
    'max_length': 1024, 'max_prompt_length_override': None,
    'lr_scheduler_type': 'cosine', 'warmup_ratio': 0.1,
    'optim': 'adamw_torch_fused', 'bf16': True,
    'gradient_checkpointing': True, 'use_reentrant': False,
    'attn_implementation': 'sdpa', 'ref_model': None,
    'save_steps': 100, 'save_total_limit': 5,
    'dropout_note': 'Effective dropout also depends on the installed TRL disable_dropout behavior.',
}


def sha256(path: Path) -> str:
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for chunk in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(chunk)
    return digest.hexdigest()


def read_json(path: Path):
    with path.open(encoding='utf-8-sig') as stream:
        return json.load(stream)


def error_kind(exc: Exception) -> str:
    # Deliberately do not echo arbitrary exception messages or environment values.
    return type(exc).__name__


def inspect_pairs(path: Path, expected: int = 1489) -> dict:
    result = {'path': str(path), 'exists': path.is_file()}
    if not path.is_file():
        return result
    result['sha256'] = sha256(path)
    try:
        records = read_json(path)
    except Exception as exc:
        result['read_error'] = error_kind(exc)
        return result
    if not isinstance(records, list):
        result['validation_error'] = 'Expected a JSON list'
        return result
    issues = Counter()
    issue_rows = []
    ids = []
    for index, record in enumerate(records):
        row_issues = []
        if not isinstance(record, dict):
            issues['non_object_record'] += 1
            issue_rows.append(index)
            continue
        rid = record.get('id')
        if isinstance(rid, (str, int)) and str(rid).strip():
            ids.append(str(rid))
        else:
            row_issues.append('missing_or_invalid_id')
        prompt = record.get('prompt')
        valid_prompt = isinstance(prompt, list) and bool(prompt)
        if valid_prompt:
            valid_prompt = all(isinstance(m, dict) and
                               m.get('role') in ('system', 'user', 'assistant') and
                               isinstance(m.get('content'), str) for m in prompt)
        if not valid_prompt:
            row_issues.append('invalid_prompt_schema')
        elif prompt[-1]['role'] != 'user':
            row_issues.append('prompt_not_ending_in_user')
        chosen, rejected = record.get('chosen'), record.get('rejected')
        if not isinstance(chosen, str) or not isinstance(rejected, str):
            row_issues.append('invalid_response_schema')
        else:
            if not chosen.strip() or not rejected.strip():
                row_issues.append('blank_response')
            if chosen.strip() == rejected.strip():
                row_issues.append('identical_pair')
            if min(len(chosen.strip()), len(rejected.strip())) < 10 or max(len(chosen.strip()), len(rejected.strip())) > 6000:
                row_issues.append('outside_archived_character_limits')
        issues.update(row_issues)
        if row_issues:
            issue_rows.append(index)
    duplicates = len(ids) - len(set(ids))
    result.update(count=len(records), expected_count=expected,
                  count_matches=len(records) == expected,
                  duplicate_id_excess=duplicates, issue_counts=dict(issues),
                  first_issue_row_indices=issue_rows[:20],
                  structurally_valid=not issues and not duplicates)
    # No prompts, chosen/rejected text, or example identifiers are emitted.
    return result


def inspect_adapter_header(path: Path) -> dict:
    """Read only the safetensors JSON header, not parameter tensors."""
    if not path.is_file():
        return {'exists': False}
    try:
        with path.open('rb') as stream:
            length_bytes = stream.read(8)
            if len(length_bytes) != 8:
                raise ValueError('Missing header length')
            size = int.from_bytes(length_bytes, 'little')
            if not 0 < size <= min(64 * 1024 * 1024, path.stat().st_size - 8):
                raise ValueError('Invalid header length')
            header = json.loads(stream.read(size))
        modules = {}
        for name, spec in header.items():
            match = re.match(r'^(.*)\.lora_([AB])(?:\.[^.]+)?\.weight$', name)
            if match:
                modules.setdefault(match[1], {})[match[2]] = spec.get('shape')
        suspects = [name for name in modules if any(term in name.lower() for term in
                    ('vision', 'audio', 'multi_modal_projector', 'lm_head'))]
        shapes = Counter(tuple((f, tuple(v)) for f, v in sorted(factors.items()))
                         for factors in modules.values())
        return {'exists': True, 'module_count': len(modules),
                'complete_ab_pairs': sum(set(v) == {'A', 'B'} for v in modules.values()),
                'first_module_paths': sorted(modules)[:12],
                'nonlanguage_or_head_name_matches': sorted(suspects),
                'shape_groups': [{'shapes': dict(shape), 'count': count}
                                 for shape, count in shapes.items()]}
    except Exception as exc:
        return {'exists': True, 'read_error': error_kind(exc)}


def inspect_adapter(directory: Path) -> dict:
    cfg_path = directory / 'adapter_config.json'
    result = {'directory': str(directory), 'exists': directory.is_dir(),
              'config_exists': cfg_path.is_file()}
    if cfg_path.is_file():
        try:
            cfg = read_json(cfg_path)
            result['config'] = {key: cfg[key] for key in CONFIG_FIELDS if key in cfg}
            result['config_sha256'] = sha256(cfg_path)
        except Exception as exc:
            result['config_error'] = error_kind(exc)
    result['safetensors_header'] = inspect_adapter_header(directory/'adapter_model.safetensors')
    states = [directory/'trainer_state.json']
    if directory.is_dir():
        checkpoints = [p for p in directory.glob('checkpoint-*') if p.is_dir() and
                       re.fullmatch(r'checkpoint-\d+', p.name)]
        checkpoints.sort(key=lambda p: int(p.name.split('-')[-1]), reverse=True)
        states.extend(p/'trainer_state.json' for p in checkpoints[:1])
    for state_path in states:
        if state_path.is_file():
            try:
                state = read_json(state_path)
                result['trainer_state'] = {'path': str(state_path), **{
                    key: state.get(key) for key in
                    ('global_step', 'epoch', 'max_steps', 'num_train_epochs', 'train_batch_size')}}
            except Exception as exc:
                result['trainer_state_error'] = error_kind(exc)
            break
    return result


def inspect_imports() -> dict:
    result = {}
    try:
        import torch
        cuda = torch.cuda.is_available()
        result['torch'] = {'cuda_available': cuda, 'built_cuda_version': torch.version.cuda,
                           'gpu_devices': [{'index': i, 'name': torch.cuda.get_device_name(i),
                                            'total_memory_gib': round(torch.cuda.get_device_properties(i).total_memory / (1024**3), 2)}
                                           for i in range(torch.cuda.device_count())] if cuda else []}
    except Exception as exc:
        result['torch'] = {'import_error': error_kind(exc)}
    try:
        import transformers
        result['transformers'] = {}
        for cls in ('AutoProcessor', 'AutoModelForMultimodalLM', 'AutoModelForCausalLM'):
            try:
                result['transformers'][cls] = getattr(transformers, cls, None) is not None
            except Exception as exc:
                result['transformers'][cls] = {'import_error': error_kind(exc)}
    except Exception as exc:
        result['transformers'] = {'import_error': error_kind(exc)}
    try:
        from trl import DPOConfig, DPOTrainer
        defaults = {}
        for field in dataclasses.fields(DPOConfig):
            if field.name in DEFAULT_FIELDS:
                defaults[field.name] = field.default if field.default is not dataclasses.MISSING else {'default_factory': getattr(field.default_factory, '__name__', 'missing')}
        source_file = Path(inspect.getfile(DPOTrainer))
        source = source_file.read_text(encoding='utf-8')
        result['trl'] = {'dpo_config_defaults': defaults,
                         'processing_class_supported': 'processing_class' in inspect.signature(DPOTrainer.__init__).parameters,
                         'dpo_trainer_source_sha256': sha256(source_file),
                         'dropout_control_lines': [{'line': i, 'code': line.strip()} for i, line in enumerate(source.splitlines(), 1) if 'disable_dropout' in line][:12]}
    except Exception as exc:
        result['trl'] = {'import_or_inspection_error': error_kind(exc)}
    return result


def main() -> int:
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument('--root', type=Path, default=Path.cwd())
    ap.add_argument('--pairs', type=Path, help='Default: ROOT/data/dpo_pairs.json')
    ap.add_argument('--gemma-warm-dir', type=Path)
    ap.add_argument('--check-imports', action='store_true', help='Import installed packages and query CUDA; never load a model')
    ap.add_argument('--output', type=Path, help='Create a new JSON report; existing files are refused')
    args = ap.parse_args()
    if args.output and args.output.exists():
        ap.error('Output already exists; choose a new name or omit --output to print only.')
    root = args.root.expanduser().resolve()
    versions = {}
    for name in PACKAGES:
        try:
            versions[name] = importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            versions[name] = None
    adapters = {key: inspect_adapter(args.gemma_warm_dir.expanduser().resolve()
                                    if key == 'gemma_warm' and args.gemma_warm_dir
                                    else root/name) for key, name in ADAPTER_DIRS.items()}
    report = {'report_type': 'gemma_dpo_readiness_inventory_not_training_validation',
              'root': str(root), 'python_version': sys.version.split()[0],
              'package_versions': versions, 'archived_dpo_settings': ARCHIVED_SETTINGS,
              'pairs': inspect_pairs(args.pairs.expanduser().resolve() if args.pairs else root/'data/dpo_pairs.json'),
              'adapters': adapters,
              'runtime': inspect_imports() if args.check_imports else {'not_checked': 'Run with --check-imports in the server training environment'},
              'notes': ['No training was started and no model weights were loaded.',
                        'Package versions describe the current environment, not proof of historical run versions.',
                        'Use the archived pair file; do not silently regenerate or change its ordering.',
                        'No training_args.bin/pickle files, credentials, environment files, or response text are inspected.',
                        'An existing Gemma DPO output directory must not be overwritten.',
                        'A successful inventory does not establish DPOTrainer/Gemma compatibility or VRAM sufficiency.']}
    serialized = json.dumps(report, ensure_ascii=False, indent=2, default=str)
    if args.output:
        with args.output.open('x', encoding='utf-8') as stream:
            stream.write(serialized + '\n')
        print(f'Report saved: {args.output}')
    else:
        print(serialized)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())
