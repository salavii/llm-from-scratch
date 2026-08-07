"""Guards against notebook/src divergence -- the failure that hid the loss bug.

The notebooks deliberately define the components they teach: reading ch4 should
show you a LayerNorm, not an import. That means several definitions exist twice,
once in a notebook and once in src/Attention.py.

That duplication is fine as long as the copies agree. It stopped being fine when
calc_loss_loader existed three times -- broken in src and ch5, quietly corrected
in ch6 -- with nothing marking which one was canonical.

This test makes that impossible to repeat: any notebook definition sharing a name
with src/Attention.py must be structurally identical to it. Genuinely different
implementations get their own name (calc_loss_batch_classifier, SelfAttention_V1,
DummyGPTModel), which is what a reader wants anyway.
"""
import ast
import json
from pathlib import Path

import pytest

REPO_ROOT = Path(__file__).resolve().parents[1]
SRC_FILE = REPO_ROOT / "src" / "Attention.py"
NOTEBOOKS = sorted((REPO_ROOT / "notebooks").glob("*.ipynb"))


def _top_level_defs(code):
    """Map {name: normalised source} for top-level defs, ignoring formatting."""
    try:
        tree = ast.parse(code)
    except SyntaxError:
        return {}
    return {
        node.name: ast.unparse(node)
        for node in tree.body
        if isinstance(node, (ast.FunctionDef, ast.ClassDef))
    }


def _notebook_defs(path):
    nb = json.loads(path.read_text(encoding="utf-8"))
    found = {}
    for cell in nb["cells"]:
        if cell["cell_type"] != "code":
            continue
        for name, source in _top_level_defs("".join(cell["source"])).items():
            found.setdefault(name, []).append(source)
    return found


SRC_DEFS = _top_level_defs(SRC_FILE.read_text(encoding="utf-8"))


def test_src_module_parses():
    assert SRC_DEFS, "src/Attention.py defined nothing -- did it fail to parse?"


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem[:24])
def test_notebook_defs_match_src(notebook):
    """Any notebook definition named like a src one must match it exactly."""
    mismatches = []
    for name, copies in _notebook_defs(notebook).items():
        if name not in SRC_DEFS:
            continue
        for copy in copies:
            if copy != SRC_DEFS[name]:
                mismatches.append(name)
                break

    assert not mismatches, (
        f"{notebook.name} has definitions that drifted from src/Attention.py: "
        f"{sorted(set(mismatches))}. Either sync them, or rename the notebook's "
        f"version if it is meant to be a different implementation."
    )


@pytest.mark.parametrize("notebook", NOTEBOOKS, ids=lambda p: p.stem[:24])
def test_no_duplicate_defs_within_a_notebook(notebook):
    """A notebook redefining the same name twice is drift waiting to happen."""
    repeated = {
        name: len(copies)
        for name, copies in _notebook_defs(notebook).items()
        if len(copies) > 1
    }

    assert not repeated, (
        f"{notebook.name} defines the same name more than once: {repeated}. "
        f"Define it once, or import it from src/Attention.py."
    )
