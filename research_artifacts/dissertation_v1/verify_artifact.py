#!/usr/bin/env python3
"""Verify the dissertation artifact without third-party dependencies."""

import hashlib
from pathlib import Path
import sys


ROOT = Path(__file__).resolve().parent
CHECKPOINT_HASH = (
    'a3ba6f3a29ea70c39788ddfb9ab9aabf4d9fc07ab7e6a0a3c277bba29e443dc9')


def sha256(path):
    digest = hashlib.sha256()
    with path.open('rb') as stream:
        for block in iter(lambda: stream.read(1024 * 1024), b''):
            digest.update(block)
    return digest.hexdigest()


def main():
    checksum_path = ROOT / 'checksums.sha256'
    failures = []
    checked = 0
    for line in checksum_path.read_text(encoding='utf-8').splitlines():
        if not line.strip():
            continue
        expected, relative = line.split('  ', 1)
        path = ROOT / relative
        actual = sha256(path) if path.is_file() else 'missing'
        checked += 1
        if actual != expected:
            failures.append((relative, expected, actual))

    checkpoint = ROOT / 'checkpoint' / 'mappo_selected_validation_v1.pt'
    if sha256(checkpoint) != CHECKPOINT_HASH:
        failures.append((str(checkpoint.relative_to(ROOT)), CHECKPOINT_HASH,
                         sha256(checkpoint)))

    if failures:
        for relative, expected, actual in failures:
            print('FAILED %s\n  expected %s\n  actual   %s' % (
                relative, expected, actual), file=sys.stderr)
        return 1
    print('Verified %d files and the frozen checkpoint.' % checked)
    return 0


if __name__ == '__main__':
    raise SystemExit(main())

