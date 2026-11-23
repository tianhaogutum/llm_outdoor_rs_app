"""Generated blocks inside hand-written Markdown (README.md, docs/FAILURES.md, docs/REVIEW.md).

A block is delimited by two HTML comments, which GitHub does not render::

    <!-- BEGIN GENERATED name (make report; do not edit) -->
    ...generated Markdown...
    <!-- END GENERATED name -->

``make report`` replaces the body of every block with what the renderers produce from
``eval/results/*.json``; text outside blocks is prose and is never touched. Every block named in a file
must have a renderer and every renderer must find its block, so a renamed or deleted block fails loudly
instead of going stale.
"""

from __future__ import annotations

import re
from pathlib import Path

BEGIN = "<!-- BEGIN GENERATED {name} (make report; do not edit) -->"
END = "<!-- END GENERATED {name} -->"
_RE = re.compile(
    r"<!-- BEGIN GENERATED (?P<name>[a-z0-9-]+)[^>]*-->\n(?P<body>.*?)<!-- END GENERATED (?P=name) -->",
    re.S,
)


def names(text: str) -> list[str]:
    return [m.group("name") for m in _RE.finditer(text)]


def fill(text: str, blocks: dict[str, str], *, where: str = "document") -> str:
    """Return ``text`` with every block body replaced by ``blocks[name]`` (one trailing newline)."""
    found = names(text)
    dup = sorted({n for n in found if found.count(n) > 1})
    if dup:
        raise ValueError(f"{where}: duplicate generated blocks {dup}")
    missing = sorted(set(blocks) - set(found))
    unknown = sorted(set(found) - set(blocks))
    if missing or unknown:
        raise ValueError(f"{where}: blocks without a marker {missing}; markers without a renderer {unknown}")

    def sub(m: re.Match) -> str:
        n = m.group("name")
        body = blocks[n].strip("\n") + "\n"
        return f"{BEGIN.format(name=n)}\n{body}{END.format(name=n)}"

    return _RE.sub(sub, text)


def update_file(path: Path, blocks: dict[str, str]) -> bool:
    """Rewrite the blocks of ``path`` in place; returns True if the file changed."""
    old = path.read_text(encoding="utf-8")
    new = fill(old, blocks, where=path.name)
    if new != old:
        path.write_text(new, encoding="utf-8")
        return True
    return False


def body(text: str, name: str) -> str:
    """Body of one block (for tests and for renderers that read hand-written text)."""
    for m in _RE.finditer(text):
        if m.group("name") == name:
            return m.group("body")
    raise KeyError(name)


def strip_blocks(text: str) -> str:
    """``text`` with every block body removed (the prose only)."""
    return _RE.sub(lambda m: "", text)
