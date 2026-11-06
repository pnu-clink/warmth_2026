"""Check every release file without displaying potential credential values.

Does not read .env files, contact a service or import research entry points.
"""
from __future__ import annotations
import ast
import hashlib
import json
from pathlib import Path
import re

ROOT = Path(__file__).resolve().parents[1]
IGNORED_DIRS = {'.git', '__pycache__', '.venv', 'venv'}
BLOCKED_SUFFIXES = {'.safetensors', '.pt', '.pth', '.bin', '.pkl', '.zip', '.gz', '.7z', '.pem', '.key'}
SECRET_PATTERNS = {
    'provider credential': re.compile(r'(?<![A-Za-z0-9])(?:gh[pousr]_[A-Za-z0-9]{20,}|github_pat_[A-Za-z0-9_]{30,}|sk-(?:proj-|ant-)?[A-Za-z0-9_-]{20,}|hf_[A-Za-z0-9]{20,}|AKIA[A-Z0-9]{16}|AIza[A-Za-z0-9_-]{30,}|xox[baprs]-[A-Za-z0-9-]{20,})'),
    'private key': re.compile(r'-----BEGIN (?:RSA |EC |OPENSSH |DSA |ENCRYPTED )?PRIVATE KEY-----'),
    'JWT-shaped credential': re.compile(r'eyJ[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}\.[A-Za-z0-9_-]{12,}'),
    'URL embedded password': re.compile(r'https?://[^\s/:@]+:[^\s/@]{4,}@'),
}
ASSIGNMENT = re.compile(r'''(?ix)\b(?:openai_api_key|api_key|hf_token|huggingface_token|github_token|access_token|client_secret|aws_secret_access_key|password)\b["']?\s*[:=]\s*["']([^"'\r\n]+)["']''')
BEARER = re.compile(r'(?i)\bBearer\s+([A-Za-z0-9_.-]{16,})')
UNQUOTED = re.compile(r'(?im)^\s*(?:export\s+)?(?:OPENAI_API_KEY|API_KEY|HF_TOKEN|GITHUB_TOKEN|ACCESS_TOKEN|CLIENT_SECRET|AWS_SECRET_ACCESS_KEY|PASSWORD)\s*=\s*([A-Za-z0-9_./+-]{8,})\s*(?:#.*)?$')
PLACEHOLDER = re.compile(r'(?i)^(?:your[_ -].*|replace[_ -].*|example.*|dummy.*|test[_ -].*|redacted.*|\[redacted[^\]]*\]|<[^>]*>|none|null|false|true|\.\.\.|changeme|token|api_key)$')


def inspect_text(text):
    findings = []
    for kind, pattern in SECRET_PATTERNS.items():
        for match in pattern.finditer(text):
            findings.append((text.count('\n', 0, match.start()) + 1, kind))
    for kind, pattern in [('literal credential assignment', ASSIGNMENT), ('unquoted credential assignment', UNQUOTED), ('literal bearer credential', BEARER)]:
        for match in pattern.finditer(text):
            value = match.group(1).strip()
            if value and not PLACEHOLDER.fullmatch(value) and not value.startswith(('$', '${')):
                findings.append((text.count('\n', 0, match.start()) + 1, kind))
    return sorted(set(findings))


def inventory(root):
    files, problems = [], []
    for path in sorted(root.rglob('*')):
        rel = path.relative_to(root)
        if any(part in IGNORED_DIRS for part in rel.parts):
            continue
        if path.is_symlink():
            problems.append(f'Symlink not allowed: {rel.as_posix()}')
            continue
        if not path.is_file():
            continue
        name = path.name.lower()
        private_name = name == '.env' or (name.startswith('.env.') and name != '.env.example')
        private_name = private_name or name in {'id_rsa', 'id_ed25519', '.netrc', '.pypirc', '.npmrc', '.git-credentials'}
        if private_name or path.suffix.lower() in BLOCKED_SUFFIXES:
            problems.append(f'Private/archive/weight file must not be uploaded: {rel.as_posix()}')
            continue
        if path.suffix == '.pyc':
            continue
        files.append(path)
    return files, problems


def audit(root=ROOT, check_manifest=True):
    root = Path(root).resolve()
    files, errors = inventory(root)
    if check_manifest:
        try:
            entries = json.loads((root / 'MANIFEST.json').read_text(encoding='utf-8'))['files']
        except (OSError, ValueError, KeyError) as exc:
            raise ValueError('Unreadable MANIFEST.json') from exc
        expected = {item['path'] for item in entries}
        if len(expected) != len(entries):
            errors.append('Duplicate manifest paths')
        actual = {p.relative_to(root).as_posix() for p in files} - {'MANIFEST.json'}
        errors.extend('Unlisted file: ' + s for s in sorted(actual - expected))
        errors.extend('Missing file: ' + s for s in sorted(expected - actual))
        for item in entries:
            path = (root / item['path']).resolve()
            if not path.is_relative_to(root):
                errors.append('Unsafe manifest path')
                continue
            if path.is_file() and path in files:
                raw = path.read_bytes()
                if len(raw) != item['bytes'] or hashlib.sha256(raw).hexdigest() != item['sha256']:
                    errors.append('Changed file: ' + item['path'])
    python_count = 0
    for path in files:
        rel = path.relative_to(root).as_posix()
        try:
            text = path.read_text(encoding='utf-8-sig')
        except UnicodeError:
            errors.append('Unexpected binary file: ' + rel)
            continue
        for line, kind in inspect_text(text):
            errors.append(f'{rel}:{line}: {kind} (value withheld)')
        if path.suffix == '.py':
            try:
                ast.parse(text, filename=rel)
                python_count += 1
            except SyntaxError as exc:
                errors.append(f'{rel}:{exc.lineno}: Python syntax error')
    if errors:
        raise ValueError('\n'.join(errors))
    return {'files_scanned': len(files), 'python_files_parsed': python_count,
            'credential_pattern_findings': 0, 'unlisted_files': 0,
            'manifest_checked': check_manifest,
            'scan_limits': 'Heuristic static scan, not proof that every possible secret is absent.'}
