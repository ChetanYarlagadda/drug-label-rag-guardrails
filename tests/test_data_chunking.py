from labelrag.chunking import chunk_drug, pack, split_sentences
from labelrag.data import extract_sections, moiety
from labelrag.lexicon import DrugLexicon


def test_moiety_strips_salts_but_keeps_first_word():
    assert moiety("METFORMIN HYDROCHLORIDE") == "metformin"
    assert moiety("AMOXICILLIN AND CLAVULANATE POTASSIUM") == "amoxicillin and clavulanate"
    assert moiety("POTASSIUM CHLORIDE") == "potassium chloride"
    assert moiety("NITROFURANTOIN (MONOHYDRATE/MACROCRYSTALS)") == "nitrofurantoin"
    assert moiety("BUTALBITAL, ACETAMINOPHEN AND CAFFEINE") == "butalbital, acetaminophen and caffeine"


def test_extract_sections_drops_children_of_present_parent_and_duplicate_storage():
    label = {
        "clinical_pharmacology": ["12 CLINICAL PHARMACOLOGY 12.3 Pharmacokinetics The half-life is 6 hours."],
        "pharmacokinetics": ["12.3 Pharmacokinetics The half-life is 6 hours."],
        "pregnancy": ["8.1 Pregnancy Limited data are available in pregnant women."],
        "how_supplied": ["16 HOW SUPPLIED/STORAGE AND HANDLING Bottles of 30. Store at 20° to 25°C."],
        "storage_and_handling": ["Storage and Handling Store at 20° to 25°C."],
    }
    s = extract_sections(label)
    assert "clinical_pharmacology" in s and "pharmacokinetics" not in s  # parent already holds the child
    assert "pregnancy" in s  # the parent (use_in_specific_populations) is missing, so the child is kept
    assert "how_supplied" in s and "storage_and_handling" not in s  # same storage text indexed once


def test_pack_respects_word_limits_and_keeps_all_words():
    sents = [" ".join(["w"] * 50) + "." for _ in range(9)] + [" ".join(["x"] * 500) + "."]
    chunks = pack(sents, min_words=120, max_words=220)
    assert all(len(c.split()) <= 220 + 40 for c in chunks)
    assert sum(len(c.split()) for c in chunks) == sum(len(s.split()) for s in sents)


def test_chunks_carry_drug_and_section_metadata(drugs):
    chunks = chunk_drug(drugs[0])
    assert {c.section for c in chunks} == set(drugs[0].sections)
    c = next(c for c in chunks if c.section == "how_supplied")
    assert c.chunk_id == "alphazol:how_supplied:00" and c.section_id == "alphazol:how_supplied"
    assert c.indexed_text(True).startswith("alphazol | How supplied\n") and c.indexed_text(False) == c.text


def test_split_sentences_splits_bullets():
    assert split_sentences("Do this. • Item one • Item two") == ["Do this.", "• Item one", "• Item two"]


def test_lexicon_prefers_the_longest_name():
    lex = DrugLexicon({"losartan": ["losartan"], "losartan_hctz": ["losartan and hydrochlorothiazide"],
                       "hctz": ["hydrochlorothiazide"]})
    assert lex.detect("Is losartan and hydrochlorothiazide stored cold?") == ["losartan_hctz"]
    assert lex.detect("Compare LOSARTAN with hydrochlorothiazide.") == ["losartan", "hctz"]
    assert lex.detect("What is the capital of Australia?") == []
