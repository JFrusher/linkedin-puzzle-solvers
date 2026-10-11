"""Word-play category checks for Pinpoint, built only from WordNet.

About half of real Pinpoint categories are word play rather than meaning:

* **Join word** — one word forms a compound with every clue ("words before *breaker*": circuit breaker,
  icebreaker, tiebreaker). Compounds that are merely a subtype of the clue (red oak is an oak) are not
  word play and are ignored.
* **Clue + class word** — clues that name a class only with the class word added (blue → blue whale).
  :func:`compound_hypernyms` adds the parents of such compounds to a clue's meaning.
* **Many senses** — every clue matches a different meaning of one word (score, tip, location → *point*),
  through a shared synonym or the clue appearing in that meaning's definition.
"""

from __future__ import annotations

import bisect
import re
from collections import defaultdict
from dataclasses import dataclass
from functools import lru_cache
from typing import Any

from core.nlp import wordnet

MIN_SENSES = 5
"""Only words with at least this many meanings can be a "many senses" answer."""
SUBTYPE_DEPTH = 2
"""A compound whose parents, this many levels up, include the clue is a subtype, not word play."""


@dataclass(frozen=True)
class Lexicon:
    names: list[str]
    """Every WordNet lemma name, sorted, for prefix scans."""
    words: frozenset[str]
    """Single alphabetic lemma names."""
    before: dict[str, frozenset[str]]
    """``before[a]`` = every ``b`` with an ``a_b`` lemma."""
    after: dict[str, frozenset[str]]
    """``after[b]`` = every ``a`` with an ``a_b`` lemma."""
    prefixes: dict[str, frozenset[str]]
    """``prefixes[tail]`` = every ``head`` where ``head + tail`` is a single word and both are words."""
    mentions: dict[str, frozenset[tuple[str, str]]]
    """``mentions[token]`` = (word, synset) pairs of polysemous words whose sense names or defines ``token``."""


@lru_cache(maxsize=1)
def lexicon() -> Lexicon:
    """Build the WordNet indexes once per process."""
    wn = wordnet()
    names = sorted(set(wn.all_lemma_names()))
    words = frozenset(name for name in names if name.isalpha())
    before: dict[str, set[str]] = defaultdict(set)
    after: dict[str, set[str]] = defaultdict(set)
    for name in names:
        parts = name.split("_")
        if len(parts) == 2 and all(part.isalpha() for part in parts):
            before[parts[0]].add(parts[1])
            after[parts[1]].add(parts[0])
    prefixes: dict[str, set[str]] = defaultdict(set)
    for word in words:
        for cut in range(3, len(word) - 2):
            if word[:cut] in words and word[cut:] in words:
                prefixes[word[cut:]].add(word[:cut])
    # Count meanings from the lemma index (no parsing), then parse only polysemous words: parsing all
    # 117k synsets takes ~16 s, these ~5k words under 2 s.
    sense_counts = {word: sum(map(len, by_pos.values())) for word, by_pos in wn._lemma_pos_offset_map.items()}
    mentions: dict[str, set[tuple[str, str]]] = defaultdict(set)
    for word, count in sense_counts.items():
        if count < MIN_SENSES or not word.isalpha():
            continue
        for synset in wn.synsets(word):
            tokens = set(re.findall(r"[a-z]+", synset.definition().lower()))
            tokens |= {name.lower() for name in synset.lemma_names() if name.isalpha()}
            for token in tokens - {word}:
                mentions[token].add((word, synset.name()))

    def freeze(index: dict[str, set[Any]]) -> dict[str, frozenset[Any]]:
        return {key: frozenset(values) for key, values in index.items()}

    return Lexicon(names, words, freeze(before), freeze(after), freeze(prefixes), freeze(mentions))


def _key(clue: str) -> str:
    return clue.strip().lower().replace(" ", "_")


def _is_subtype(compound: str, clue: str) -> bool:
    """True when ``compound`` is just a kind of ``clue`` (red_oak is an oak)."""
    for synset in wordnet().synsets(compound):
        for parent in synset.closure(lambda s: s.hypernyms() + s.instance_hypernyms(), depth=SUBTYPE_DEPTH):
            if clue in parent.lemma_names():
                return True
    return False


def joins(clue: str) -> set[str]:
    """Category labels of every join word that forms a (non-subtype) compound with ``clue``."""
    lex = lexicon()
    key = _key(clue)
    labels: set[str] = set()
    for tail in lex.before.get(key, ()):
        if not _is_subtype(f"{key}_{tail}", key):
            labels.add(f'words before "{tail}"')
    for head in lex.after.get(key, ()):
        if not _is_subtype(f"{head}_{key}", key):
            labels.add(f'words after "{head}"')
    start = bisect.bisect_left(lex.names, key)
    for name in lex.names[start:]:
        if not name.startswith(key):
            break
        tail = name[len(key) :]
        if len(tail) >= 3 and tail in lex.words and not _is_subtype(name, key):
            labels.add(f'words before "{tail}"')
    for head in lex.prefixes.get(key, ()):
        if not _is_subtype(head + key, key):
            labels.add(f'words after "{head}"')
    return labels


def compound_hypernyms(clue: str, depth: int) -> list[Any]:
    """Synsets of ``clue_X`` compounds that are a kind of ``X`` (blue_whale is a whale)."""
    wn = wordnet()
    key = _key(clue)
    found = []
    for tail in lexicon().before.get(key, ()):
        for synset in wn.synsets(f"{key}_{tail}", pos=wn.NOUN):
            if any(tail in parent.lemma_names() for parent in synset.closure(lambda s: s.hypernyms(), depth=depth)):
                found.append(synset)
    return found


def senses(clues: list[str]) -> dict[str, int]:
    """Words for which the clues match the most distinct meanings, as ``meanings of "W"`` → count."""
    hits: dict[str, dict[int, set[str]]] = defaultdict(dict)
    for index, clue in enumerate(clues):
        for word, synset in lexicon().mentions.get(_key(clue), ()):
            hits[word].setdefault(index, set()).add(synset)
    counts: dict[str, int] = {}
    for word, per_clue in hits.items():
        used: set[str] = set()
        # Match clues with the fewest options first, so each clue gets a meaning of its own.
        for options in sorted(per_clue.values(), key=len):
            free = sorted(options - used)
            if free:
                used.add(free[0])
        counts[f'meanings of "{word}"'] = len(used)
    return counts
