"""Workspace-local tmp_path fixture for environments denying OS temp access."""
import hashlib
from pathlib import Path

import pytest


@pytest.fixture
def tmp_path(request):
    root = Path(__file__).resolve().parent / "real_data_pilots" / "pytest_workspace"
    path = root / hashlib.sha256(request.node.nodeid.encode("utf-8")).hexdigest()[:16]
    path.mkdir(parents=True, exist_ok=True)
    return path
