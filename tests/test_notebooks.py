"""Notebooks must be valid and executable as Jupyter actually reads them.

Jupyter concatenates a cell's `source` list *verbatim* -- `"".join(source)`, not
`"\\n".join(source)`. Every entry except the last therefore has to end with a
newline. A notebook that omits them is still valid JSON and still parses if you
join with newlines yourself, so it looks fine to a naive check while rendering
as one giant run-on paragraph and failing to execute.

That is not hypothetical: it shipped. These tests check the notebook the way the
consumer does.
"""

import ast
import json
from pathlib import Path

import pytest

NOTEBOOKS = sorted((Path(__file__).parent.parent / "notebooks").glob("*.ipynb"))

pytestmark = pytest.mark.skipif(not NOTEBOOKS, reason="no notebooks")


@pytest.mark.parametrize("path", NOTEBOOKS, ids=lambda p: p.name)
class TestNotebook:
    def test_is_valid_nbformat(self, path):
        nbformat = pytest.importorskip("nbformat")
        nb = nbformat.read(str(path), as_version=4)
        nbformat.validate(nb)

    def test_source_lines_end_with_newline(self, path):
        """Without trailing newlines every cell collapses to a single line."""
        nb = json.loads(path.read_text())
        offenders = []
        for i, cell in enumerate(nb["cells"]):
            for line in cell["source"][:-1]:
                if not line.endswith("\n"):
                    offenders.append((i, line[:60]))
        assert not offenders, (
            f"{len(offenders)} source lines lack a trailing newline, so Jupyter "
            f"will run them together; first few: {offenders[:3]}"
        )

    def test_code_cells_parse_as_joined(self, path):
        """Parse exactly what Jupyter executes: "".join(source)."""
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "code":
                continue
            src = "".join(cell["source"])
            # Drop IPython shell escapes, which are not Python.
            src = "\n".join(ln for ln in src.split("\n")
                            if not ln.lstrip().startswith(("!", "%")))
            try:
                ast.parse(src)
            except SyntaxError as e:
                pytest.fail(f"cell {i} of {path.name} does not parse: {e}")

    def test_long_markdown_cells_have_paragraph_breaks(self, path):
        """A long cell with no blank line is the visible signature of the bug.

        Only applies to cells long enough that a wall of text would be wrong --
        a notebook of short section headings is perfectly fine.
        """
        nb = json.loads(path.read_text())
        for i, cell in enumerate(nb["cells"]):
            if cell["cell_type"] != "markdown":
                continue
            text = "".join(cell["source"])
            if len(text) <= 400:
                continue
            assert "\n\n" in text, (
                f"cell {i} of {path.name} is {len(text)} chars with no paragraph "
                f"break; it will render as one block: {text[:80]!r}"
            )

    def test_no_unmerged_branch_references(self, path):
        """Install cells must not pin a feature branch after it is merged."""
        text = path.read_text()
        assert "claude/helico-residue-contacts-redesign" not in text, (
            "notebook still installs from the merged feature branch"
        )
