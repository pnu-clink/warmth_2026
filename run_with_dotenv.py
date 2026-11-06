#!/usr/bin/env python3
"""Run a historical Python entry point with a private .env key, without editing it.

Usage: python run_with_dotenv.py [--env-file PATH] SCRIPT [SCRIPT_OPTIONS...]
Only an explicit --run or --execute in SCRIPT_OPTIONS causes key loading.
For historical scripts that always call the API, add --load-key before SCRIPT.
"""
from __future__ import annotations
import argparse
import os
from pathlib import Path
import runpy
import sys
from api_credentials import load_openai_key


def main(argv=None):
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--env-file', type=Path)
    parser.add_argument('--load-key', action='store_true')
    parser.add_argument('script', type=Path)
    parser.add_argument('script_args', nargs=argparse.REMAINDER)
    args = parser.parse_args(argv)
    script = args.script.resolve()
    if not script.is_file() or script.suffix != '.py':
        parser.error('SCRIPT must be an existing Python file.')
    active = args.load_key or any(flag in args.script_args for flag in ('--run', '--execute'))
    previous = os.environ.get('OPENAI_API_KEY')
    old_argv, old_path = sys.argv[:], sys.path[:]
    try:
        if active:
            os.environ['OPENAI_API_KEY'] = load_openai_key(args.env_file)
        sys.argv = [str(script), *args.script_args]
        sys.path.insert(0, str(script.parent))
        runpy.run_path(str(script), run_name='__main__')
    finally:
        sys.argv, sys.path = old_argv, old_path
        if active:
            if previous is None:
                os.environ.pop('OPENAI_API_KEY', None)
            else:
                os.environ['OPENAI_API_KEY'] = previous


if __name__ == '__main__':
    main()
