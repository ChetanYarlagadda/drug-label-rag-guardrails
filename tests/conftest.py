"""Shared fixtures: two tiny synthetic drug labels and their chunks."""

from __future__ import annotations

import pytest

from labelrag.chunking import Chunk, chunk_drug
from labelrag.data import Drug


def make_drug(drug_id: str, name: str, sections: dict[str, str], pharm_class: list[str] | None = None) -> Drug:
    return Drug(drug_id=drug_id, name=name, generic_name=name.upper(), rank=0, n_labels=1, set_id="x", version="1",
                effective_time="20260101", manufacturer="Test", brand_names=[], route=["ORAL"],
                pharm_class=pharm_class or [], sections=sections, full_text=" ".join(sections.values()))


@pytest.fixture
def drugs() -> list[Drug]:
    alpha = make_drug("alphazol", "alphazol", {
        "boxed_warning": "WARNING: LACTIC ACIDOSIS Postmarketing cases of lactic acidosis have occurred.",
        "indications_and_usage": "1 INDICATIONS AND USAGE Alphazol is indicated for type 2 diabetes mellitus.",
        "dosage_forms_and_strengths": "3 DOSAGE FORMS AND STRENGTHS Tablets: 500 mg, 850 mg, and 1,000 mg.",
        "description": "11 DESCRIPTION Alphazol hydrochloride is a white powder with a molecular weight of 165.62.",
        "clinical_pharmacology": "12 CLINICAL PHARMACOLOGY The plasma elimination half-life is approximately 6.2 hours.",
        "how_supplied": "16 HOW SUPPLIED Bottles of 100. Store at 20° to 25°C (68° to 77°F).",
    }, ["Biguanide [EPC]"])
    beta = make_drug("betamab", "betamab", {
        "indications_and_usage": "1 INDICATIONS AND USAGE Betamab is indicated for hypertension.",
        "dosage_forms_and_strengths": "3 DOSAGE FORMS AND STRENGTHS Tablets: 10 mg and 20 mg.",
        "description": "11 DESCRIPTION Betamab has a molecular weight of 441.53.",
        "how_supplied": "16 HOW SUPPLIED Store at 2°C to 8°C (36°F to 46°F) in a refrigerator.",
    }, ["Biguanide [EPC]"])
    return [alpha, beta]


@pytest.fixture
def chunks(drugs) -> list[Chunk]:
    return [c for d in drugs for c in chunk_drug(d)]
