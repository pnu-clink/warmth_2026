"""Offline release verification. No API, GPU or network calls."""
import json
import sys
sys.dont_write_bytecode = True
from release_guard import ROOT, audit, inspect_text


def verify():
    return audit()['files_scanned'] - 1


if __name__ == '__main__':
    try:
        print(json.dumps(audit(), indent=2))
        print('PASS: no API, GPU or network calls.')
    except (ValueError, OSError) as exc:
        raise SystemExit(str(exc))
