# ArtAgent/core/dataset/clip_tokenizer.py
"""
Offline CLIP token counter (the text encoder of SD 1.5 / SDXL / Pony / Illustrious / NoobAI).

Implements CLIP's byte-pair encoding with the original vocabulary
(data/bpe_simple_vocab_16e6.txt.gz, MIT, see data/LICENSE-CLIP-BPE.txt), so
captions can be checked against the 75-token window without downloading
anything. Only token *counts* are needed, so ids/decoding are not implemented.
"""
from __future__ import annotations

import gzip
import html
import os
import re
import threading
from functools import lru_cache

_VOCAB_PATH = os.path.join(os.path.dirname(__file__), "data", "bpe_simple_vocab_16e6.txt.gz")

# CLIP's pattern uses \p{L} / \p{N}; this is the equivalent with Python's `re`:
# contractions, runs of letters, single digits, runs of other symbols (incl. "_").
_PATTERN = re.compile(r"'s|'t|'re|'ve|'m|'ll|'d|[^\W\d_]+|\d|(?:[^\s\w]|_)+", re.IGNORECASE)
_UNCURL = str.maketrans({"‘": "'", "’": "'", "‚": "'", "‛": "'",
                         "“": '"', "”": '"', "„": '"', "‟": '"'})


@lru_cache()
def _bytes_to_unicode() -> dict[int, str]:
    bs = list(range(ord("!"), ord("~") + 1)) + list(range(ord("¡"), ord("¬") + 1)) + list(range(ord("®"), ord("ÿ") + 1))
    cs = bs[:]
    n = 0
    for b in range(256):
        if b not in bs:
            bs.append(b)
            cs.append(256 + n)
            n += 1
    return dict(zip(bs, (chr(c) for c in cs)))


def _get_pairs(word: tuple[str, ...]) -> set[tuple[str, str]]:
    return {(word[i], word[i + 1]) for i in range(len(word) - 1)}


class ClipTokenizer:
    """Byte-pair encoder equivalent to OpenAI CLIP's SimpleTokenizer (counting only)."""

    def __init__(self, bpe_path: str = _VOCAB_PATH):
        merges = gzip.open(bpe_path).read().decode("utf-8").split("\n")
        merges = merges[1:49152 - 256 - 2 + 1]
        self.bpe_ranks = {tuple(m.split()): i for i, m in enumerate(merges)}
        self.byte_encoder = _bytes_to_unicode()
        self._cache: dict[str, int] = {}

    def _bpe_len(self, token: str) -> int:
        """Number of BPE pieces for one pre-tokenized word."""
        cached = self._cache.get(token)
        if cached is not None:
            return cached
        word = tuple(token[:-1]) + (token[-1] + "</w>",)
        pairs = _get_pairs(word)
        if not pairs:
            self._cache[token] = 1
            return 1
        while True:
            bigram = min(pairs, key=lambda pair: self.bpe_ranks.get(pair, float("inf")))
            if bigram not in self.bpe_ranks:
                break
            first, second = bigram
            new_word: list[str] = []
            i = 0
            while i < len(word):
                try:
                    j = word.index(first, i)
                except ValueError:
                    new_word.extend(word[i:])
                    break
                new_word.extend(word[i:j])
                i = j
                if word[i] == first and i < len(word) - 1 and word[i + 1] == second:
                    new_word.append(first + second)
                    i += 2
                else:
                    new_word.append(word[i])
                    i += 1
            word = tuple(new_word)
            if len(word) == 1:
                break
            pairs = _get_pairs(word)
        if len(self._cache) < 50_000:
            self._cache[token] = len(word)
        return len(word)

    @staticmethod
    def clean(text: str) -> str:
        text = html.unescape(html.unescape(text or ""))
        text = text.translate(_UNCURL)  # ftfy.fix_text (used by CLIP) straightens curly quotes
        return re.sub(r"\s+", " ", text).strip().lower()

    def count(self, text: str) -> int:
        """CLIP tokens in `text`, excluding the start/end tokens (the 75-token budget)."""
        total = 0
        for piece in _PATTERN.findall(self.clean(text)):
            encoded = "".join(self.byte_encoder[b] for b in piece.encode("utf-8"))
            total += self._bpe_len(encoded)
        return total


_instance: ClipTokenizer | None = None
_lock = threading.Lock()


def clip_token_count(text: str) -> int:
    """Counts CLIP tokens (without BOS/EOS). Loads the vocabulary on first use (~0.5 s)."""
    global _instance
    if _instance is None:
        with _lock:
            if _instance is None:
                _instance = ClipTokenizer()
    return _instance.count(text)
