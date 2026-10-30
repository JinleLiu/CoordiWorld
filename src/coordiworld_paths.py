"""Path configuration for the original research modules."""
import os
from pathlib import Path

REPOSITORY = Path(__file__).resolve().parents[1]

def workspace_path(relative=''):
    """Return a string path beneath the user-configured experiment workspace."""
    root = Path(os.environ.get('COORDIWORLD_WORKSPACE', REPOSITORY/'workdir'))
    return str(root / relative)
