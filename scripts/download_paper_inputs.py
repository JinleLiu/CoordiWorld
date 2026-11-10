#!/usr/bin/env python3
"""Acquire the approved numerical attachment and verify it before extraction.

Without --archive, the actual URL must be registered in paper_revision/ACCESS.json.
No private credentials, author directories or implicit local cache are used.
"""
import argparse
import hashlib
import json
from pathlib import Path, PurePosixPath
import shutil
import tempfile
import urllib.request
import zipfile

ROOT = Path(__file__).resolve().parents[1]


def extract_verified(archive, output, expected_sha):
    if hashlib.sha256(archive.read_bytes()).hexdigest() != expected_sha:
        raise RuntimeError('Numerical attachment SHA-256 mismatch')
    if output.exists():
        raise RuntimeError('Output directory must not already exist')
    with zipfile.ZipFile(archive) as bundle:
        names = bundle.namelist()
        prefix = 'CoordiWorld_numerics_v12a/'
        for entry in bundle.infolist():
            path = PurePosixPath(entry.filename)
            mode = entry.external_attr >> 16
            if (not entry.filename.startswith(prefix) or path.is_absolute() or '..' in path.parts
                    or '\\' in entry.filename or (mode & 0o170000) == 0o120000):
                raise RuntimeError('Unexpected or unsafe archive member')
        if sum(entry.file_size for entry in bundle.infolist()) > 32 * 1024 * 1024:
            raise RuntimeError('Unexpected numerical attachment size')
        manifest = json.loads(bundle.read(prefix + 'NUMERICAL_INPUT_MANIFEST_v12a.json'))
        required = {prefix + item['path'] for item in manifest['files']}
        if set(names) != required | {prefix + 'NUMERICAL_INPUT_MANIFEST_v12a.json'}:
            raise RuntimeError('Numerical attachment member inventory mismatch')
        for item in manifest['files']:
            if hashlib.sha256(bundle.read(prefix + item['path'])).hexdigest() != item['sha256']:
                raise RuntimeError('Numerical member SHA-256 mismatch: ' + item['path'])
        output.parent.mkdir(parents=True, exist_ok=True)
        with tempfile.TemporaryDirectory(dir=output.parent, prefix='.cw-inputs-') as directory:
            stage = Path(directory)
            bundle.extractall(stage)
            shutil.move(str(stage / prefix.rstrip('/')), str(output))
    return manifest


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--archive', type=Path, help='Already downloaded approved attachment; useful for controlled access')
    parser.add_argument('--output', type=Path, default=ROOT / 'paper_revision/numerics')
    args = parser.parse_args()
    access = json.loads((ROOT / 'paper_revision/ACCESS.json').read_text())
    asset = access['numerical_attachment']
    with tempfile.TemporaryDirectory(prefix='cw-download-') as directory:
        if args.archive:
            archive = args.archive.resolve()
        else:
            url = asset.get('download_url')
            if not url or not url.startswith('https://'):
                raise RuntimeError('No author-approved numerical download URL is configured; see ACCESS.json')
            archive = Path(directory) / asset['filename']
            request = urllib.request.Request(url, headers={'User-Agent': 'CoordiWorld-public-reproduction'})
            try:
                with urllib.request.urlopen(request, timeout=60) as response, archive.open('wb') as out:
                    shutil.copyfileobj(response, out)
            except OSError as exc:
                raise RuntimeError('Numerical attachment is not reachable at the configured Release URL. '
                                   'Check the actual Release asset and retry, or use --archive after downloading it.') from exc
        manifest = extract_verified(archive, args.output.resolve(), asset['sha256'])
    print(json.dumps({'status': 'PASS', 'files': len(manifest['files']),
                      'attachment_sha256': asset['sha256'], 'output': str(args.output)}))


if __name__ == '__main__':
    main()
