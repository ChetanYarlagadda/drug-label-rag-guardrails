import numpy as np
import pytest

from labelrag.evalset import (EvalItem, answerable_gold, boxed_heading, canon, half_life, heading_topics,
                              molecular_weight, numbers_in, storage_temperature, strengths)
from labelrag.scoring import answer_correct, cohen_kappa, mcnemar_exact, refusal_scores, retrieval_metrics, wilson


@pytest.mark.parametrize("text,celsius,fahrenheit", [
    ("Store at 20° to 25°C (68° to 77°F) [see USP].", ["20", "25"], ["68", "77"]),
    ("Store at 20°C to 25°C (68°F to 77°F).", ["20", "25"], ["68", "77"]),
    ("Store at: 20-25°C (68-77°F).", ["20", "25"], ["68", "77"]),
    ("Store at 25°C (77°F); excursions permitted to 15-30°C.", ["25"], ["77"]),
    ("Vials should be stored under refrigeration at 2ºC–8ºC.", ["2", "8"], []),
])
def test_storage_temperature(text, celsius, fahrenheit):
    g = storage_temperature(text)
    assert g["celsius"] == celsius and g["fahrenheit"] == fahrenheit


def test_strengths_reads_lists_and_thousands():
    assert strengths("Tablets: 2.5 mg, 5 mg, 10 mg, 20 mg") == ["2.5 mg", "5 mg", "10 mg", "20 mg"]
    assert strengths("Capsules: 10, 18, 25, and 40 mg; tablets 1,000 mg") == ["10 mg", "18 mg", "25 mg", "40 mg", "1000 mg"]
    assert strengths("Each tablet contains 25 mcg") == ["25 mcg"]


def test_molecular_weight_half_life_and_heading():
    assert molecular_weight("C4H11N5•HCl and a molecular weight of 165.62.") == "165.62"
    assert molecular_weight("The molecular weight of ketorolac tromethamine is 376.40.") == "376.4"
    nums, sents = half_life("Absorption is fast. The elimination half-life is 5 to 7 hours in adults. Done.")
    assert nums == ["5", "7"] and len(sents) == 1
    assert boxed_heading("BOXED WARNING WARNING: SUICIDAL THOUGHTS AND BEHAVIORS Antidepressants increased") == \
        "SUICIDAL THOUGHTS AND BEHAVIORS"
    assert boxed_heading("Cardiovascular Thrombotic Events NSAIDs cause risk") is None
    assert heading_topics("RISKS FROM USE OF ZETADOL TABLETS; ADDICTION", "zetadol") == [["addict"]]


def test_numbers_ignore_citation_ids():
    assert numbers_in("It is 1,000 mg [C2] and 2.50 mL (C3).") == {"1000", "2.5"}
    assert canon("068") == "68"


def test_gold_chunks_point_at_the_evidence(drugs, chunks):
    alpha_chunks = [c for c in chunks if c.drug_id == "alphazol"]
    g, ids = answerable_gold(drugs[0], alpha_chunks, "storage")
    assert ids == ["alphazol:how_supplied:00"] and g["celsius"] == ["20", "25"]
    g, ids = answerable_gold(drugs[0], alpha_chunks, "strengths")
    assert g["strengths"] == ["500 mg", "850 mg", "1000 mg"] and ids == ["alphazol:dosage_forms_and_strengths:00"]


def _item(subtype: str, gold: dict) -> EvalItem:
    return EvalItem("q", "test", "answerable", subtype, "?", ["d"], "answer", gold)


def test_answer_correct_rules():
    st = _item("storage", {"celsius": ["20", "25"], "fahrenheit": ["68", "77"]})
    assert answer_correct(st, "Store at 20°C to 25°C (C1).")
    assert answer_correct(st, "Store between 68 and 77 °F.")
    assert not answer_correct(st, "Store at 25°C.")
    s = _item("strengths", {"strengths": ["10 mg", "20 mg"]})
    assert answer_correct(s, "10 mg and 20 mg tablets") and not answer_correct(s, "10 mg tablets")
    mw = _item("molecular_weight", {"molecular_weight": "165.62"})
    assert answer_correct(mw, "It is 165.6 g/mol") and not answer_correct(mw, "It is 129.16")
    bw = _item("boxed_warning", {"topics": [["behavi", "suicid", "though"]]})
    assert answer_correct(bw, "It warns about suicidal thoughts.") and not answer_correct(bw, "It warns about bleeding.")
    hl = _item("half_life", {"numbers": ["5", "7"]})
    assert answer_correct(hl, "About 5 to 7 hours.") and not answer_correct(hl, "About 12 hours.")


def test_metrics_and_statistics():
    assert retrieval_metrics(2)["hit@1"] == 0.0 and retrieval_metrics(2)["hit@3"] == 1.0
    assert retrieval_metrics(None)["mrr@10"] == 0.0
    r = refusal_scores(np.array([True, True, False, False]), np.array([True, False, True, False]))
    assert r["refusal_precision"] == 0.5 and r["refusal_recall"] == 0.5 and r["behaviour_accuracy"] == 0.5
    lo, hi = wilson(43, 50)
    assert lo < 0.86 < hi
    assert mcnemar_exact(0, 0) == 1.0 and mcnemar_exact(0, 10) < 0.01
    assert cohen_kappa([True, False, True, False], [True, False, True, False]) == 1.0
