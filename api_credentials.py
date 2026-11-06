"""Read OpenAI credentials from a private .env, without logging key values."""
from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent


def load_openai_key(env_file: str | Path | None = None) -> str:
    """The selected .env takes precedence; a missing default permits shell auth.

    Only OPENAI_API_KEY is read. Other .env values, including API endpoints, are
    not applied to the process. Explicit missing files are always an error.
    """
    path = Path(env_file).expanduser() if env_file is not None else ROOT / '.env'
    if path.is_file():
        try:
            from dotenv import dotenv_values
        except ImportError:
            raise SystemExit('Install python-dotenv: python -m pip install python-dotenv') from None
        try:
            values = dotenv_values(path, encoding='utf-8-sig', interpolate=False)
        except Exception:
            raise SystemExit('Could not parse .env. Use UTF-8 and KEY=value syntax.') from None
        key = values.get('OPENAI_API_KEY')
    elif env_file is not None:
        raise SystemExit('The requested --env-file does not exist.')
    else:
        key = os.environ.get('OPENAI_API_KEY')
    if not isinstance(key, str) or not key.strip():
        raise SystemExit('Set OPENAI_API_KEY in the project-root .env (or the process environment).')
    key = key.strip()
    if key.lower().startswith(('replace_', 'your_', 'example', '<')) or any(c.isspace() for c in key):
        raise SystemExit('OPENAI_API_KEY is a placeholder or contains whitespace. Update the private .env.')
    return key
