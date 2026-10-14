#!/usr/bin/env python3
"""
sync_tracker.py — verify that every claim in track.html is still true.

A rubric checklist written by hand rots the moment someone renames a function.
This script treats each `<span class="where">` in track.html as a machine-
checkable assertion of the form "file — symbol, symbol, symbol", and goes and
looks.

    python sync_tracker.py              # verify, print a report
    python sync_tracker.py --json       # machine-readable
    python sync_tracker.py --stats      # add a codebase inventory

Exit status is 0 when everything resolves and 1 when something has drifted, so
it drops straight into a pre-submission check or a CI step.
"""

from __future__ import annotations

import argparse
import json
import re
import sys
from dataclasses import dataclass, field
from pathlib import Path

ROOT = Path(__file__).resolve().parent
TRACKER = ROOT / "track.html"

# ANSI, disabled when stdout is not a terminal.
_TTY = sys.stdout.isatty()
def _c(code: str, text: str) -> str:
    return f"\033[{code}m{text}\033[0m" if _TTY else text

GREEN = lambda s: _c("32", s)
RED = lambda s: _c("31", s)
YELLOW = lambda s: _c("33", s)
DIM = lambda s: _c("2", s)
BOLD = lambda s: _c("1", s)
CYAN = lambda s: _c("36", s)


# ─────────────────────────────────────────────────────────────────────────────
#  Claim extraction
# ─────────────────────────────────────────────────────────────────────────────


@dataclass
class Claim:
    """One checklist row: a human claim plus the evidence it points at."""

    module: str
    text: str
    raw_where: str
    paths: list[str] = field(default_factory=list)
    symbols: list[str] = field(default_factory=list)

    # Resolved during verification
    missing_paths: list[str] = field(default_factory=list)
    missing_symbols: list[str] = field(default_factory=list)

    @property
    def ok(self) -> bool:
        return not self.missing_paths and not self.missing_symbols


_CLAIM_RE = re.compile(
    r'<div class="claim"><span>(?P<text>.*?)</span>'
    r'<span class="where">(?P<where>.*?)</span>',
    re.DOTALL,
)
_MODULE_RE = re.compile(r"<h3>(?P<title>.*?)</h3>", re.DOTALL)
_TAG_RE = re.compile(r"<[^>]+>")

_ENTITIES = {
    "&amp;": "&", "&lt;": "<", "&gt;": ">", "&quot;": '"',
    "&#39;": "'", "&nbsp;": " ", "&times;": "×", "&rarr;": "→",
}


def _clean(text: str) -> str:
    text = _TAG_RE.sub("", text)
    for entity, char in _ENTITIES.items():
        text = text.replace(entity, char)
    return " ".join(text.split())


# A path looks like a/b/c.ext. A symbol is an identifier, optionally dotted or
# with a trailing (), that is not a path.
_PATH_RE = re.compile(r"[\w./-]+\.(?:py|tsx|ts|astro|mjs|json|css|html)\b")
_SYMBOL_RE = re.compile(r"\b([A-Za-z_][\w]*(?:\.[A-Za-z_][\w]*)*)\s*(?:\(\))?")

# Words that appear in the "where" prose but are not symbols to look for.
_STOPWORDS = {
    "and", "the", "its", "in", "for", "with", "via", "or", "of", "a", "an",
    "UI", "frontend", "backend", "docstring", "rationale", "default", "header",
    "box", "their", "to", "on", "is", "as", "by", "from", "not", "but", "that",
    "term", "trick", "flag", "real", "time", "exact",
}


def parse_claims(html: str) -> list[Claim]:
    """Walk track.html, pairing each claim with the module heading above it."""
    claims: list[Claim] = []

    # Index module headings by position so each claim can find its section.
    headings = [(m.start(), _clean(m.group("title"))) for m in _MODULE_RE.finditer(html)]

    for match in _CLAIM_RE.finditer(html):
        pos = match.start()
        module = "General"
        for h_pos, title in headings:
            if h_pos < pos:
                module = title
            else:
                break

        where = _clean(match.group("where"))
        paths = _PATH_RE.findall(where)

        # Strip the paths out before hunting for symbols, so "main.py" does not
        # contribute a bogus "main" / "py" symbol pair.
        residue = where
        for p in paths:
            residue = residue.replace(p, " ")

        symbols: list[str] = []
        for token in _SYMBOL_RE.findall(residue):
            token = token.strip()
            if not token or token in _STOPWORDS or token in symbols:
                continue
            if len(token) < 3:
                continue
            # Keep things that look like identifiers rather than English:
            # CamelCase, snake_case, dotted, or a leading underscore.
            if ("_" in token) or ("." in token) or any(c.isupper() for c in token[1:]):
                symbols.append(token)

        claims.append(
            Claim(
                module=module,
                text=_clean(match.group("text")),
                raw_where=where,
                paths=paths,
                symbols=symbols,
            )
        )

    return claims


# ─────────────────────────────────────────────────────────────────────────────
#  Verification
# ─────────────────────────────────────────────────────────────────────────────


def _resolve(path_str: str) -> Path | None:
    """Resolve a claim's path against the repo root, tolerating prefixes."""
    ignored_parts = {".venv", "venv", "node_modules", ".git", ".astro", "dist", "__pycache__"}
    candidates = [ROOT / path_str]
    # Claims sometimes name a file without its directory ("generator.py").
    if "/" not in path_str:
        candidates.extend(
            p for p in ROOT.rglob(path_str)
            if not any(part in ignored_parts for part in p.parts)
        )
    for candidate in candidates:
        if candidate.is_file():
            return candidate
    return None


def verify(claims: list[Claim]) -> list[Claim]:
    """Check each claim's files exist and each symbol appears in one of them."""
    source_cache: dict[Path, str] = {}

    for claim in claims:
        resolved: list[Path] = []
        for path_str in claim.paths:
            found = _resolve(path_str)
            if found is None:
                claim.missing_paths.append(path_str)
            else:
                resolved.append(found)

        if not resolved:
            # Nothing to search symbols in; the path failure is the finding.
            claim.missing_symbols.extend(claim.symbols)
            continue

        haystack = ""
        for path in resolved:
            if path not in source_cache:
                try:
                    source_cache[path] = path.read_text(encoding="utf-8", errors="replace")
                except OSError:
                    source_cache[path] = ""
            haystack += source_cache[path]

        for symbol in claim.symbols:
            # Match the last dotted component too: a claim may say
            # "EdgeModelZoo.infer" where the source defines `def infer`.
            needle = symbol.split(".")[-1]
            if symbol not in haystack and needle not in haystack:
                claim.missing_symbols.append(symbol)

    return claims


# ─────────────────────────────────────────────────────────────────────────────
#  Codebase inventory
# ─────────────────────────────────────────────────────────────────────────────

_COUNT_GLOBS = {
    "Python (backend)": "backend/**/*.py",
    "TypeScript / TSX": "frontend/src/**/*.ts*",
    "Astro": "frontend/src/**/*.astro",
    "CSS": "frontend/src/**/*.css",
}


def inventory() -> dict[str, dict[str, int]]:
    """Count files, lines and comment lines per language."""
    out: dict[str, dict[str, int]] = {}
    for label, pattern in _COUNT_GLOBS.items():
        files = [p for p in ROOT.glob(pattern) if p.is_file() and "node_modules" not in p.parts]
        lines = 0
        comments = 0
        blank = 0
        for path in files:
            try:
                text = path.read_text(encoding="utf-8", errors="replace")
            except OSError:
                continue
            for line in text.splitlines():
                lines += 1
                stripped = line.strip()
                if not stripped:
                    blank += 1
                elif stripped.startswith(("#", "//", "/*", "*", '"""', "'''", "<!--")):
                    comments += 1
        out[label] = {
            "files": len(files),
            "lines": lines,
            "comment_lines": comments,
            "blank_lines": blank,
        }
    return out


# ─────────────────────────────────────────────────────────────────────────────
#  Reporting
# ─────────────────────────────────────────────────────────────────────────────


def report(claims: list[Claim], show_stats: bool) -> int:
    by_module: dict[str, list[Claim]] = {}
    for claim in claims:
        by_module.setdefault(claim.module, []).append(claim)

    print()
    print(BOLD("  Data Center Edge-Vision Commander — evidence check"))
    print(DIM(f"  {TRACKER.name} → source tree at {ROOT}"))
    print()

    total = len(claims)
    passed = sum(1 for c in claims if c.ok)

    for module, group in by_module.items():
        ok_count = sum(1 for c in group if c.ok)
        header = f"  {module}"
        status = f"{ok_count}/{len(group)}"
        colour = GREEN if ok_count == len(group) else RED
        print(f"{CYAN(header)}  {colour(status)}")

        for claim in group:
            if claim.ok:
                print(f"    {GREEN('✓')} {DIM(claim.text[:84])}")
            else:
                print(f"    {RED('✗')} {claim.text[:84]}")
                for path in claim.missing_paths:
                    print(f"        {RED('missing file')}   {path}")
                for symbol in claim.missing_symbols:
                    print(f"        {YELLOW('missing symbol')} {symbol}")
        print()

    pct = (passed / total * 100) if total else 0.0
    summary = f"  {passed}/{total} claims verified ({pct:.0f}%)"
    print(BOLD(GREEN(summary)) if passed == total else BOLD(RED(summary)))

    if show_stats:
        print()
        print(BOLD("  Codebase inventory"))
        stats = inventory()
        width = max(len(k) for k in stats)
        for label, counts in stats.items():
            density = (
                counts["comment_lines"] / max(counts["lines"] - counts["blank_lines"], 1) * 100
            )
            print(
                f"    {label.ljust(width)}  "
                f"{str(counts['files']).rjust(3)} files  "
                f"{str(counts['lines']).rjust(6)} lines  "
                f"{DIM(f'{density:.0f}% commented')}"
            )

    print()
    return 0 if passed == total else 1


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__.strip().splitlines()[0])
    parser.add_argument("--json", action="store_true", help="emit machine-readable JSON")
    parser.add_argument("--stats", action="store_true", help="include a codebase inventory")
    args = parser.parse_args()

    if not TRACKER.exists():
        print(RED(f"track.html not found at {TRACKER}"), file=sys.stderr)
        return 2

    claims = verify(parse_claims(TRACKER.read_text(encoding="utf-8")))

    if args.json:
        payload = {
            "total": len(claims),
            "passed": sum(1 for c in claims if c.ok),
            "claims": [
                {
                    "module": c.module,
                    "text": c.text,
                    "where": c.raw_where,
                    "ok": c.ok,
                    "missing_paths": c.missing_paths,
                    "missing_symbols": c.missing_symbols,
                }
                for c in claims
            ],
        }
        if args.stats:
            payload["inventory"] = inventory()
        print(json.dumps(payload, indent=2))
        return 0 if payload["passed"] == payload["total"] else 1

    return report(claims, args.stats)


if __name__ == "__main__":
    raise SystemExit(main())
