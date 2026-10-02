#!/usr/bin/env python3
"""Every third-party import in scripts/ must be a declared dependency.

Found by building the release env the way a new user does, in a clean conda
prefix with `pip install -e ".[eval,dev]"`, and then trying to run the commands
in docs/DATA.md. They died on ModuleNotFoundError: cv2, lz4 and pandas are
imported by the preprocessing scripts and none of them was declared anywhere in
pyproject.toml. The core install carries only torch/transformers/numpy/pillow,
and the train/eval/viz extras do not cover scripts/ at all.

That is a reproducibility hole rather than a nuisance: docs/DATA.md is the
documented path from raw public downloads to the tree the paper's numbers were
measured on, and it was unrunnable after the documented install.

The `data` extra now covers it. This test keeps the declaration honest as
scripts/ grows, by walking the ASTs rather than trusting a hand-maintained list.

Run with: pytest tests/release/test_declared_dependencies.py
"""
import ast
import sys
import tomllib
from pathlib import Path

import pytest

_REPO_ROOT = Path(__file__).resolve().parents[2]
_SCRIPTS = _REPO_ROOT / "scripts"

# Import name -> distribution name, where they differ.
_IMPORT_TO_DIST = {
    "cv2": "opencv-python",
    "PIL": "pillow",
    "yaml": "pyyaml",
    "sklearn": "scikit-learn",
    "rouge_score": "rouge-score",
}

# Deliberately NOT declared. README > Install explains both: their installs are
# environment-specific and the code degrades with a clear message instead of
# crashing.
_INTENTIONALLY_UNDECLARED = {"flash_attn", "depth_anything_3"}

# Modules that live in scripts/ itself. scripts/ is not a package, but running
# `python scripts/x.py` puts scripts/ on sys.path, so these resolve.
_LOCAL_TO_SCRIPTS = {p.stem for p in _SCRIPTS.glob("*.py")}

# Top-level packages shipped by this repo.
_FIRST_PARTY = {
    "onecanvas", "model_adapters", "reprojection", "geometry", "inference",
    "utils", "training", "debug",
}


def _declared_distributions():
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    proj = cfg["project"]
    specs = list(proj.get("dependencies", []))
    for extra_specs in proj.get("optional-dependencies", {}).values():
        specs += extra_specs
    out = set()
    for spec in specs:
        name = spec.split(";")[0]
        for sep in (">=", "<=", "==", "!=", "~=", ">", "<", "["):
            name = name.split(sep)[0]
        out.add(name.strip().lower().replace("_", "-"))
    return out


def _third_party_imports(path):
    tree = ast.parse(path.read_text())
    mods = set()
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            for a in node.names:
                mods.add(a.name.split(".")[0])
        elif isinstance(node, ast.ImportFrom) and node.level == 0 and node.module:
            mods.add(node.module.split(".")[0])
    return {
        m for m in mods
        if m not in sys.stdlib_module_names
        and m not in _FIRST_PARTY
        and m not in _LOCAL_TO_SCRIPTS
    }


def _script_files():
    return sorted(_SCRIPTS.glob("*.py"))


def test_scripts_directory_is_not_empty():
    assert _script_files(), "no scripts found; this test would pass vacuously"


@pytest.mark.parametrize("path", _script_files(), ids=lambda p: p.name)
def test_script_imports_are_declared(path):
    declared = _declared_distributions()
    undeclared = []
    for mod in sorted(_third_party_imports(path)):
        if mod in _INTENTIONALLY_UNDECLARED:
            continue
        dist = _IMPORT_TO_DIST.get(mod, mod).lower().replace("_", "-")
        if dist not in declared:
            undeclared.append(f"{mod} (would need '{dist}')")
    assert not undeclared, (
        f"{path.name} imports undeclared dependencies: {undeclared}. "
        f"Add them to an extra in pyproject.toml, or to "
        f"_INTENTIONALLY_UNDECLARED here with the reason documented in "
        f"README > Install."
    )


def test_data_extra_covers_the_preprocessing_stack():
    """Pin the specific gap that shipped, so it cannot silently come back."""
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    data = cfg["project"]["optional-dependencies"]["data"]
    names = {s.split(">=")[0].split("==")[0].strip().lower() for s in data}
    for need in ("opencv-python", "lz4", "pandas", "pyarrow"):
        assert need in names, f"the data extra must carry {need}"


def test_all_extra_includes_data():
    with open(_REPO_ROOT / "pyproject.toml", "rb") as f:
        cfg = tomllib.load(f)
    all_spec = " ".join(cfg["project"]["optional-dependencies"]["all"])
    assert "data" in all_spec, "'all' must pull in the data extra"


if __name__ == "__main__":
    raise SystemExit(pytest.main([__file__, "-v"]))
