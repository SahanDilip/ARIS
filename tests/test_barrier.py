"""The information barrier, enforced rather than asserted.

Nothing in the population or investigator tiers may import the platform. Three
lines of test code turn "we treated the platform as a black box" from a claim a
reviewer must take on trust into a property of the artefact. Say so in the paper.
"""
from __future__ import annotations

import ast
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent / "aris"
GUARDED = ["population", "investigator"]
FORBIDDEN = "platform"


def _imports(path: Path) -> list[str]:
    tree = ast.parse(path.read_text())
    out = []
    for node in ast.walk(tree):
        if isinstance(node, ast.Import):
            out += [a.name for a in node.names]
        elif isinstance(node, ast.ImportFrom):
            # level>0 means relative: "from ..platform.env import X" -> "..platform.env"
            out.append("." * (node.level or 0) + (node.module or ""))
    return out


def test_guarded_tiers_do_not_import_platform():
    violations = []
    for tier in GUARDED:
        for py in (ROOT / tier).rglob("*.py"):
            for imp in _imports(py):
                parts = imp.lstrip(".").split(".")
                if FORBIDDEN in parts:
                    violations.append(f"{py.relative_to(ROOT)} imports {imp}")
    assert not violations, "INFORMATION BARRIER BREACHED:\n" + "\n".join(violations)


def test_guarded_tiers_exist():
    for tier in GUARDED:
        assert (ROOT / tier).is_dir(), f"missing tier {tier}"
