"""Explicit publication resource locations; no scientific computation.

Add paper_revision/source to PYTHONPATH for historical entry-point inspection.
Full historical inference still needs its original environment and correctly
bound inputs; publication wrappers do not bypass frozen SHA assertions.
"""
import json
import os
from pathlib import Path
import re


def publication_identifier(key, config_path=None):
    """Read the designated identifier without changing any branch predicate."""
    path = config_path or os.environ.get('COORDIWORLD_IDENTITY_CONFIG')
    if path is None:
        path = Path(__file__).resolve().parents[1] / 'config/source_identifiers_v12a.json'
    values = json.loads(Path(path).read_text(encoding='utf-8'))['identifiers']
    value = values[key]
    if not isinstance(value, str) or not value:
        raise ValueError('Identifier must be a nonempty string: ' + key)
    return value


def publication_path(value, roots=None):
    """Resolve only an explicit logical-root prefix, never a machine default."""
    match = re.match(r'^(?:<(?P<angle>CW_[A-Z_]+)>|\$\{(?P<brace>CW_[A-Z_]+)\})(?P<suffix>/.*)?$', value)
    if not match:
        raise ValueError('Expected a logical CW root prefix')
    name = match.group('angle') or match.group('brace')
    source = os.environ if roots is None else roots
    root = source.get(name)
    if not root:
        raise RuntimeError('Set ' + name + ' to the explicitly restored archive location')
    base = Path(root).expanduser()
    if not base.is_absolute():
        raise ValueError(name + ' must be an absolute path')
    return str(base) + (match.group('suffix') or '')
