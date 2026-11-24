# ArtAgent/core/dataset/caption_io.py
"""
Reading, writing and combining caption files (`image.png` -> `image.txt`).

Trainers disagree about multi-line caption files: SimpleTuner splits them into one
caption per line and OneTrainer samples one line per epoch, so text that was
"appended" on a new line silently becomes an *alternative* caption (and a
"---" separator line becomes a caption of its own). kohya only reads multiple
lines with enable_wildcard / enable_multiline_caption and then skips tag shuffling.

So ArtAgents keeps one caption per line on purpose:
  - Append / Prepend join text on the SAME line with an explicit joiner (", " or " ").
  - "Add as caption variant" writes a new line deliberately, one clean caption per line.
"""
from __future__ import annotations

import os
import re
import tempfile

MODE_OVERWRITE = "Overwrite"
MODE_SKIP = "Skip"
MODE_APPEND = "Append"
MODE_PREPEND = "Prepend"
MODE_VARIANT = "Add as caption variant"
FILE_MODES = [MODE_OVERWRITE, MODE_SKIP, MODE_APPEND, MODE_PREPEND, MODE_VARIANT]

JOINER_CHOICES = {
    "comma  ( , )": ", ",
    "space  (   )": " ",
    "period ( . )": ". ",
}
DEFAULT_JOINER = ", "

# Separators the old Append/Prepend wrote ("\n\n---\n\n") and similar artefacts
_SEPARATOR_LINE = re.compile(r"^\s*(-{3,}|={3,}|_{3,}|\*{3,})\s*$")
_KOHYA_MULTILINE_SEPARATOR = "---===---"


def caption_path(image_path: str, extension: str = ".txt") -> str:
    """Sidecar caption path for an image (same folder, same stem)."""
    return os.path.splitext(image_path)[0] + extension


def read_caption(image_path: str, extension: str = ".txt") -> str:
    path = caption_path(image_path, extension)
    if not os.path.exists(path):
        return ""
    with open(path, "r", encoding="utf-8", errors="replace") as f:
        return f.read()


def write_text_atomic(path: str, text: str) -> None:
    """Writes via a temp file + os.replace, so a crash never leaves half a caption."""
    folder = os.path.dirname(path) or "."
    fd, tmp = tempfile.mkstemp(prefix=".tmp-", suffix=".txt", dir=folder)
    try:
        with os.fdopen(fd, "w", encoding="utf-8", newline="\n") as f:
            f.write(text)
        os.replace(tmp, path)
    except BaseException:
        try:
            os.remove(tmp)
        except OSError:
            pass
        raise


def write_caption(image_path: str, text: str, extension: str = ".txt") -> str:
    path = caption_path(image_path, extension)
    write_text_atomic(path, text)
    return path


def single_line(text: str, joiner: str = " ") -> str:
    """
    Collapses model output to one clean caption line: strips markdown bullets/headings,
    joins lines with `joiner` (", " for tags, " " for prose) and normalises whitespace.
    """
    if not text:
        return ""
    parts = []
    for line in str(text).replace("\r", "\n").split("\n"):
        line = line.strip()
        if not line or _SEPARATOR_LINE.match(line):
            continue
        line = re.sub(r"^(#{1,6}\s+|[-*•]\s+|\d+[.)]\s+)", "", line)  # markdown heading/bullet/number
        line = line.replace("**", "").replace("__", "")
        parts.append(line.strip())
    if not parts:
        return ""
    sep = joiner if joiner.strip() else " "
    joined = parts[0]
    for part in parts[1:]:
        if sep.strip() and joined.rstrip().endswith(sep.strip()):
            joined = joined.rstrip() + " " + part
        else:
            joined = joined.rstrip() + sep + part
    return re.sub(r"[ \t]+", " ", joined).strip()


def split_variants(text: str) -> list[str]:
    """Caption variants in a file: one per non-empty line, ignoring separator lines."""
    if not text:
        return []
    lines = []
    for line in text.replace("\r", "\n").split("\n"):
        stripped = line.strip()
        if not stripped or _SEPARATOR_LINE.match(stripped) or stripped == _KOHYA_MULTILINE_SEPARATOR:
            continue
        lines.append(stripped)
    return lines


def variant_count(text: str) -> int:
    return len(split_variants(text))


def join_on_line(first: str, second: str, joiner: str) -> str:
    """Joins two caption fragments on one line without doubling the separator."""
    first, second = (first or "").strip(), (second or "").strip()
    if not first:
        return second
    if not second:
        return first
    sep_core = joiner.strip()
    if sep_core and second.startswith(sep_core):  # 'a' + ', b' -> 'a, b'
        return first.rstrip() + second
    if sep_core and first.endswith(sep_core):  # 'a,' + 'b' -> 'a, b'
        return f"{first} {second}"
    return f"{first}{joiner}{second}"


def combine(existing: str, new: str, mode: str, joiner: str = DEFAULT_JOINER) -> str:
    """
    Combines an existing caption file's text with a new caption.

    Append/Prepend apply to every existing variant line (normally just one), so a
    file never gains accidental extra lines. MODE_VARIANT adds `new` as an extra line.
    """
    new = single_line(new, joiner if joiner.strip() else " ")
    variants = split_variants(existing)
    if mode == MODE_OVERWRITE or not variants:
        return new
    if mode == MODE_SKIP:
        return existing
    if mode == MODE_APPEND:
        return "\n".join(join_on_line(v, new, joiner) for v in variants)
    if mode == MODE_PREPEND:
        return "\n".join(join_on_line(new, v, joiner) for v in variants)
    if mode == MODE_VARIANT:
        if new in variants:
            return "\n".join(variants)
        return "\n".join(variants + [new])
    raise ValueError(f"Unknown caption file mode: {mode}")


def needs_normalizing(text: str) -> bool:
    """True for files with separator lines, blank lines between captions or stray whitespace lines."""
    if not text:
        return False
    raw_lines = text.replace("\r", "\n").split("\n")
    non_empty = [l for l in raw_lines if l.strip()]
    if any(_SEPARATOR_LINE.match(l.strip()) or l.strip() == _KOHYA_MULTILINE_SEPARATOR for l in non_empty):
        return True
    return len(non_empty) > 1 and any(not l.strip() for l in raw_lines[:-1])


def normalize(text: str, how: str, joiner: str = DEFAULT_JOINER) -> str:
    """
    how = "merge":    all variants merged into one line with `joiner`
    how = "variants": one clean caption per line (separators and blank lines removed)
    """
    variants = split_variants(text)
    if how == "merge":
        if joiner.strip() == ",":  # tag captions: keep each tag once
            seen, tags = set(), []
            for v in variants:
                for tag in (t.strip() for t in v.split(",")):
                    if tag and tag.lower() not in seen:
                        seen.add(tag.lower())
                        tags.append(tag)
            return ", ".join(tags)
        merged = ""
        for v in variants:
            merged = join_on_line(merged, v, joiner)
        return merged
    if how == "variants":
        seen, unique = set(), []
        for v in variants:
            if v not in seen:
                seen.add(v)
                unique.append(v)
        return "\n".join(unique)
    raise ValueError(f"Unknown normalize mode: {how}")
