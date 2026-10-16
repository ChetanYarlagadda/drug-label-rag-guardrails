"""Section-aware chunking: each SPL section is split into sentence-aligned chunks that keep their metadata."""

from __future__ import annotations

import re
from dataclasses import asdict, dataclass

from labelrag.data import SECTION_TITLE, SECTIONS, Drug

MIN_WORDS = 120  # pack sentences until a chunk has at least this many words ...
MAX_WORDS = 220  # ... without going over this many (longer sentences are split on words)

_SENT_SPLIT = re.compile(r"(?<=[.!?:;])\s+(?=[A-Z0-9•(\[])|\s+(?=•)")


@dataclass(frozen=True)
class Chunk:
    """One retrievable unit. ``section_id`` (drug:section) is the unit of section-level citation credit."""

    chunk_id: str  # e.g. "metformin:description:00"
    drug_id: str
    drug_name: str
    section: str  # openFDA field name
    section_title: str
    order: int  # position within the section
    text: str  # the label text of this chunk

    @property
    def section_id(self) -> str:
        return f"{self.drug_id}:{self.section}"

    def header(self) -> str:
        """Contextual header: the drug and section the text comes from."""
        return f"{self.drug_name} | {self.section_title}"

    def indexed_text(self, with_header: bool = True) -> str:
        return f"{self.header()}\n{self.text}" if with_header else self.text

    def to_row(self) -> dict:
        return {**asdict(self), "section_id": self.section_id, "n_words": len(self.text.split())}


def split_sentences(text: str) -> list[str]:
    """Split label text into sentences and bullet items (simple rules tuned for SPL prose)."""
    return [s.strip() for s in _SENT_SPLIT.split(text) if s and s.strip()]


def pack(sentences: list[str], min_words: int = MIN_WORDS, max_words: int = MAX_WORDS) -> list[str]:
    """Greedily pack sentences into chunks of min..max words; a sentence longer than max_words is split on words."""
    pieces: list[str] = []
    for s in sentences:
        words = s.split()
        if len(words) <= max_words:
            pieces.append(s)
        else:
            pieces.extend(" ".join(words[i:i + max_words]) for i in range(0, len(words), max_words))
    chunks: list[str] = []
    cur: list[str] = []
    n = 0
    for p in pieces:
        k = len(p.split())
        if cur and n + k > max_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
        cur.append(p)
        n += k
        if n >= min_words:
            chunks.append(" ".join(cur))
            cur, n = [], 0
    if cur:
        if chunks and n < min_words // 3 and len(chunks[-1].split()) + n <= max_words + min_words // 3:
            chunks[-1] = chunks[-1] + " " + " ".join(cur)  # fold a short tail into the previous chunk
        else:
            chunks.append(" ".join(cur))
    return chunks


def chunk_drug(drug: Drug) -> list[Chunk]:
    """All chunks of one drug's label, in label order."""
    out: list[Chunk] = []
    order = [k for k, _, _ in SECTIONS]
    for section in sorted(drug.sections, key=order.index):
        for i, text in enumerate(pack(split_sentences(drug.sections[section]))):
            out.append(Chunk(f"{drug.drug_id}:{section}:{i:02d}", drug.drug_id, drug.name, section,
                             SECTION_TITLE[section], i, text))
    return out


def chunk_corpus(drugs: list[Drug]) -> list[Chunk]:
    return [c for d in drugs for c in chunk_drug(d)]
