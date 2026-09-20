"""Guards on the path layout.

The point of paths.py is that no script resolves a path from its own location.
These tests fail if that creeps back in.
"""
import os
import re
from pathlib import Path

import pytest

from pfdf_seasonality import paths


def test_repo_root_is_the_repository():
    assert (paths.REPO / "pyproject.toml").exists()
    assert paths.REPORTS == paths.REPO / "results" / "reports"


def test_data_root_honours_the_environment(monkeypatch, tmp_path):
    monkeypatch.setenv("PFDF_SEASONALITY_DATA", str(tmp_path))
    import importlib
    reloaded = importlib.reload(paths)
    try:
        assert reloaded.DATA == tmp_path
        assert reloaded.RAW == tmp_path / "data" / "raw"
    finally:
        monkeypatch.delenv("PFDF_SEASONALITY_DATA", raising=False)
        importlib.reload(paths)


def test_derived_paths_sit_under_their_roots():
    for p in (paths.INVENTORIES, paths.MTBS, paths.AUX, paths.ATLAS14,
              paths.GHCN_NORMALS, paths.CONUS404, paths.REANALYSIS_15MIN):
        assert paths.RAW in p.parents or p == paths.RAW
    assert paths.CONUS404_HOURLY.parent == paths.CONUS404
    assert paths.EVENTS.parent == paths.PROCESSED / "inventory"


def test_describe_lists_every_root():
    text = paths.describe()
    for name in ("REPO", "REPORTS", "DATA", "RAW", "PROCESSED", "FIGURES"):
        assert name in text


SCRIPT_DIRS = ["scripts", "scripts/plots"]
BANNED = re.compile(r'Path\(__file__\)')


@pytest.mark.parametrize("rel", SCRIPT_DIRS)
def test_no_script_resolves_paths_from_its_own_location(rel):
    offenders = [p.name for p in (paths.REPO / rel).glob("*.py")
                 if BANNED.search(p.read_text())]
    assert not offenders, (
        f"{offenders} compute paths from __file__; import from "
        f"pfdf_seasonality.paths instead")


def test_no_script_redefines_the_shared_constants():
    """A literal redefinition is banned; deriving a different shape is fine.

    `03_` needs lowercase keys and `11_` needs a plain list, and both build
    theirs from the shared dict, so there is still one source of truth.
    """
    literal = re.compile(
        r'^SEASONS = \{\s*["\'](?:DJF|djf)["\']\s*:\s*\['   # a month-list literal
        r'|^OKABE_ITO = \['
        r'|^MONTH_TO_SEASON = \{\s*\d', re.M)
    offenders = []
    for rel in SCRIPT_DIRS:
        for p in (paths.REPO / rel).glob("*.py"):
            if literal.search(p.read_text()):
                offenders.append(p.name)
    assert not offenders, f"{offenders} redefine shared constants; import them"
