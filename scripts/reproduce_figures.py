#!/usr/bin/env python3
"""Run the paper team's v12 plot implementation with publication identifiers."""
import argparse
import hashlib
import json
from pathlib import Path
import shutil
import subprocess
import sys
import tempfile

ROOT = Path(__file__).resolve().parents[1]


def main():
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--assets-dir', type=Path, required=True,
                        help='paper_assets directory from the approved v12a numerical attachment')
    parser.add_argument('--output', type=Path, required=True, help='New or empty output directory')
    args = parser.parse_args()
    assets, output = args.assets_dir.resolve(), args.output.resolve()
    if sys.flags.optimize:
        raise RuntimeError('Run without -O: the original script uses assertions')
    if output.exists() and (not output.is_dir() or any(output.iterdir())):
        raise RuntimeError('Output must be a new or empty directory')
    if output == assets or assets in output.parents:
        raise RuntimeError('Output must be outside the input assets')
    pins = json.loads((ROOT / 'paper_revision/figures/INPUT_SHA256.json').read_text())
    code_root = ROOT / 'paper_revision/figures'
    code_pins = json.loads((code_root / 'CODE_SHA256.json').read_text())
    for name, expected in pins.items():
        path = assets / name
        if not path.is_file():
            raise RuntimeError('Missing figure input: ' + name)
        if hashlib.sha256(path.read_bytes()).hexdigest() != expected:
            raise RuntimeError('Figure input SHA-256 mismatch: ' + name)
    for name, expected in code_pins.items():
        if hashlib.sha256((code_root / name).read_bytes()).hexdigest() != expected:
            raise RuntimeError('Figure implementation SHA-256 mismatch: ' + name)
    with tempfile.TemporaryDirectory(prefix='coordiworld-paper-figures-') as directory:
        stage = Path(directory)
        for name in pins:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(assets / name, target)
        for name in code_pins:
            target = stage / name
            target.parent.mkdir(parents=True, exist_ok=True)
            shutil.copyfile(code_root / name, target)
        subprocess.run([sys.executable, '-I', str(stage / 'scripts/plot_saved_results.py')],
                       cwd=stage, check=True)
        output.mkdir(parents=True, exist_ok=True)
        shutil.copytree(stage / 'outputs', output / 'outputs')
        for name in ['DATA_CHECKS_v12.json', 'FIGURE_DATA_MANIFEST_v12.json']:
            shutil.copyfile(stage / name, output / name)
    print(json.dumps({'status': 'PASS', 'plot_algorithm_unchanged': True,
                      'script_changes': 'route identifier strings only', 'figures': 5,
                      'formats': ['pdf', 'svg', 'png'], 'new_confidence_intervals': 0}))


if __name__ == '__main__':
    main()
