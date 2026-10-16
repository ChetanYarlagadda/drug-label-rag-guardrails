"""Drug-name detection: which indexed drugs does a question name? (longest match first, word boundaries)."""

from __future__ import annotations

import re

from labelrag.data import Drug


def _norm(text: str) -> str:
    return " " + re.sub(r"[^a-z0-9]+", " ", text.lower()).strip() + " "


class DrugLexicon:
    """Maps surface names (moiety, openFDA generic name, brand names) to drug ids."""

    def __init__(self, names_by_drug: dict[str, list[str]]) -> None:
        self.names: dict[str, set[str]] = {}
        for drug_id, names in names_by_drug.items():
            for name in names:
                key = _norm(name).strip()
                if len(key) >= 4:
                    self.names.setdefault(key, set()).add(drug_id)
        self._ordered = sorted(self.names, key=lambda n: (-len(n), n))

    @classmethod
    def from_drugs(cls, drugs: list[Drug]) -> "DrugLexicon":
        return cls({d.drug_id: sorted({d.name, d.generic_name, *d.brand_names}) for d in drugs})

    def detect(self, question: str) -> list[str]:
        """Drug ids named in the question, in order of first mention; a matched span is not matched again."""
        text = _norm(question)
        found: list[tuple[int, str]] = []
        for name in self._ordered:
            pat = f" {name} "
            start = text.find(pat)
            while start >= 0:
                for d in sorted(self.names[name]):
                    found.append((start, d))
                text = text[:start] + " " + "#" * (len(pat) - 2) + " " + text[start + len(pat):]
                start = text.find(pat)
        out: list[str] = []
        for _, d in sorted(found):
            if d not in out:
                out.append(d)
        return out
