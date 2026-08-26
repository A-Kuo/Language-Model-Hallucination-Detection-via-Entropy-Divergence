"""Ensure the repo root is importable as `v2.<module>` regardless of how
pytest is invoked (matches the sys.path.insert convention already used in
v2/pipeline.py)."""

import os
import sys

_REPO_ROOT = os.path.dirname(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
if _REPO_ROOT not in sys.path:
    sys.path.insert(0, _REPO_ROOT)
