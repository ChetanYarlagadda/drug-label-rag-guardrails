"""openFDA drug labels: pick ~200 common prescription drugs, fetch one label each (cached), extract SPL sections.

The drug list is data-driven: openFDA's ``count`` endpoint ranks the generic names of human prescription labels by
the number of labels (a proxy for widely dispensed multi-source drugs). Names are grouped by active moiety (salt
words removed, so METOPROLOL TARTRATE and METOPROLOL SUCCINATE are one drug), and for each moiety the most recent
label that has the core sections is kept.
"""

from __future__ import annotations

import hashlib
import json
import re
import time
from dataclasses import dataclass, field
from pathlib import Path

import requests

API = "https://api.fda.gov/drug/label.json"
RX_FILTER = 'openfda.product_type:"HUMAN PRESCRIPTION DRUG"'

# Salt / ester / hydrate words dropped to get the active moiety (never the first word of a component).
SALT_WORDS = {
    "hydrochloride", "dihydrochloride", "hcl", "sodium", "potassium", "calcium", "magnesium", "besylate", "maleate",
    "tartrate", "bitartrate", "succinate", "fumarate", "mesylate", "hyclate", "citrate", "sulfate", "phosphate",
    "acetate", "bromide", "hydrobromide", "monohydrate", "dihydrate", "trihydrate", "tromethamine", "medoxomil",
    "propionate", "dipropionate", "acetonide", "pamoate", "benzoate", "bisulfate", "carbonate", "cypionate",
    "furoate", "valerate", "decanoate", "enanthate", "mofetil", "disodium", "anhydrous", "hemihydrate", "lysine",
    "dimesylate", "napsylate", "xinafoate", "trometamol", "hydrate", "sesquihydrate",
}
# Products that are prescription "drugs" in SPL but not medicines a drug-label QA system is about.
EXCLUDE = {"oxygen", "nitrogen", "nitrous oxide", "carbon dioxide", "air", "helium", "sodium chloride",
           "sodium fluoride", "water", "dextrose", "medical air"}
REQUIRED_SECTIONS = ("indications_and_usage", "dosage_and_administration", "contraindications", "adverse_reactions",
                     "description")

# Indexed SPL sections in label order: (openFDA field, display title, parent field or None).
# A parent section (e.g. clinical_pharmacology) already contains its subsections, so children are only used when the
# parent is missing; this avoids indexing the same text twice.
SECTIONS: list[tuple[str, str, str | None]] = [
    ("boxed_warning", "Boxed warning", None),
    ("indications_and_usage", "Indications and usage", None),
    ("dosage_and_administration", "Dosage and administration", None),
    ("dosage_forms_and_strengths", "Dosage forms and strengths", None),
    ("contraindications", "Contraindications", None),
    ("warnings_and_cautions", "Warnings and precautions", None),
    ("warnings", "Warnings", None),
    ("precautions", "Precautions", None),
    ("adverse_reactions", "Adverse reactions", None),
    ("drug_interactions", "Drug interactions", None),
    ("use_in_specific_populations", "Use in specific populations", None),
    ("pregnancy", "Pregnancy", "use_in_specific_populations"),
    ("lactation", "Lactation", "use_in_specific_populations"),
    ("nursing_mothers", "Nursing mothers", "use_in_specific_populations"),
    ("pediatric_use", "Pediatric use", "use_in_specific_populations"),
    ("geriatric_use", "Geriatric use", "use_in_specific_populations"),
    ("overdosage", "Overdosage", None),
    ("description", "Description", None),
    ("clinical_pharmacology", "Clinical pharmacology", None),
    ("mechanism_of_action", "Mechanism of action", "clinical_pharmacology"),
    ("pharmacodynamics", "Pharmacodynamics", "clinical_pharmacology"),
    ("pharmacokinetics", "Pharmacokinetics", "clinical_pharmacology"),
    ("how_supplied", "How supplied", None),
    ("storage_and_handling", "Storage and handling", None),
]
SECTION_TITLE = {k: t for k, t, _ in SECTIONS}


@dataclass
class Drug:
    """One indexed drug: the chosen label's identifiers and its cleaned sections."""

    drug_id: str  # slug of the moiety, e.g. "metformin"
    name: str  # display name used in questions, e.g. "metformin"
    generic_name: str  # openFDA generic name of the chosen label, e.g. "METFORMIN HYDROCHLORIDE"
    rank: int  # rank of the generic name in the openFDA label count
    n_labels: int
    set_id: str
    version: str
    effective_time: str
    manufacturer: str
    brand_names: list[str]
    route: list[str]
    pharm_class: list[str]
    sections: dict[str, str] = field(default_factory=dict)  # field -> cleaned text
    full_text: str = ""  # every text field of the label (indexed or not), for absence checks


def slug(s: str) -> str:
    return re.sub(r"[^a-z0-9]+", "_", s.lower()).strip("_")


def moiety(generic_name: str) -> str:
    """Active-moiety name: lowercase, parenthetical notes and salt words removed (first word of a part always kept)."""
    s = re.sub(r"\(.*?\)", " ", generic_name.lower())
    parts = [p.strip() for p in re.split(r",\s*|\s+and\s+", s) if p.strip()]
    out = []
    for p in parts:
        toks = p.split()
        kept = toks[:1] + [t for t in toks[1:] if t not in SALT_WORDS]
        out.append(" ".join(kept))
    # de-duplicate parts while keeping order ("x sodium and x" -> "x")
    seen: list[str] = []
    for p in out:
        if p not in seen:
            seen.append(p)
    return " and ".join(seen) if len(seen) <= 2 else ", ".join(seen[:-1]) + " and " + seen[-1]


def clean(text: str) -> str:
    """Collapse whitespace and drop control characters."""
    text = re.sub(r"[\u0000-\u0008\u000b-\u001f]", " ", text)
    return re.sub(r"\s+", " ", text).strip()


def _norm(text: str) -> str:
    return re.sub(r"[^a-z0-9]+", " ", text.lower()).strip()


def _get(url_params: dict, session: requests.Session, retries: int = 5) -> dict:
    for attempt in range(retries):
        r = session.get(API, params=url_params, timeout=60)
        if r.status_code == 429 or r.status_code >= 500:
            time.sleep(2 ** attempt * 2)
            continue
        if r.status_code == 404:
            return {"results": []}
        r.raise_for_status()
        return r.json()
    r.raise_for_status()
    return {}


def fetch_counts(raw_dir: Path, session: requests.Session | None = None) -> list[dict]:
    """Generic names of human prescription labels ranked by label count (cached)."""
    path = raw_dir / "openfda_rx_generic_name_counts.json"
    if not path.exists():
        session = session or requests.Session()
        data = _get({"search": RX_FILTER, "count": "openfda.generic_name.exact", "limit": 1000}, session)
        path.parent.mkdir(parents=True, exist_ok=True)
        path.write_text(json.dumps(data), encoding="utf-8")
    return json.loads(path.read_text(encoding="utf-8"))["results"]


def candidate_moieties(counts: list[dict]) -> list[dict]:
    """One entry per active moiety, in count order, keeping the most-labelled generic name."""
    seen: dict[str, dict] = {}
    for rank, row in enumerate(counts):
        m = moiety(row["term"])
        if m in EXCLUDE or m in seen:
            continue
        seen[m] = {"moiety": m, "generic_name": row["term"], "rank": rank, "n_labels": row["count"]}
    return list(seen.values())


def fetch_labels(generic_name: str, raw_dir: Path, session: requests.Session, limit: int = 3) -> list[dict]:
    """The ``limit`` most recent prescription labels with this exact generic name (cached as the raw API response)."""
    path = raw_dir / "labels" / f"{slug(generic_name)}.json"
    if path.exists():
        return json.loads(path.read_text(encoding="utf-8"))["results"]
    search = f'openfda.generic_name.exact:"{generic_name}" AND {RX_FILTER}'
    data = _get({"search": search, "sort": "effective_time:desc", "limit": limit}, session)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(data), encoding="utf-8")
    time.sleep(0.3)  # stay well under openFDA's 240 requests/minute without a key
    return data.get("results", [])


def _text(label: dict, key: str) -> str:
    v = label.get(key)
    if not v:
        return ""
    return clean(" ".join(v) if isinstance(v, list) else str(v))


def label_ok(label: dict) -> bool:
    """A label is usable if it has the core sections and some warnings section."""
    has_core = all(len(_text(label, k).split()) >= 5 for k in REQUIRED_SECTIONS)
    has_warn = any(_text(label, k) for k in ("warnings_and_cautions", "warnings"))
    return has_core and has_warn


def extract_sections(label: dict) -> dict[str, str]:
    """Cleaned text of the indexed sections, without duplicated parent/child or storage text."""
    texts = {k: _text(label, k) for k, _, _ in SECTIONS}
    out: dict[str, str] = {}
    for key, _, parent in SECTIONS:
        t = texts[key]
        if len(t.split()) < 3:
            continue
        if parent is not None and len(texts.get(parent, "").split()) >= 3:
            continue  # the parent section already contains this text
        out[key] = t
    if "storage_and_handling" in out and "how_supplied" in out:
        body = _norm(re.sub(r"^\S*\s*storage and handling", "", out["storage_and_handling"], flags=re.I))
        if body and body in _norm(out["how_supplied"]):
            del out["storage_and_handling"]
    return out


def full_label_text(label: dict) -> str:
    """Every text field of a label, tables included (tags stripped), used to verify that a fact is absent."""
    parts = []
    for k in label:
        if k in ("openfda", "set_id", "id", "effective_time", "version"):
            continue
        t = _text(label, k)
        parts.append(clean(re.sub(r"<[^>]+>", " ", t)) if k.endswith("_table") else t)
    return " ".join(parts)


def build_drug(entry: dict, label: dict) -> Drug:
    of = label.get("openfda", {})
    brands = sorted({b.strip() for b in of.get("brand_name", []) if b and _norm(b) != _norm(entry["generic_name"])})
    return Drug(
        drug_id=slug(entry["moiety"]), name=entry["moiety"], generic_name=entry["generic_name"], rank=entry["rank"],
        n_labels=entry["n_labels"], set_id=label.get("set_id", ""), version=str(label.get("version", "")),
        effective_time=label.get("effective_time", ""), manufacturer=(of.get("manufacturer_name") or [""])[0],
        brand_names=brands, route=of.get("route", []), pharm_class=of.get("pharm_class_epc", []),
        sections=extract_sections(label), full_text=full_label_text(label),
    )


def select_drugs(raw_dir: Path, n_drugs: int = 200, log=print) -> tuple[list[Drug], list[dict]]:
    """Walk the ranked moieties, fetch labels, and keep the first ``n_drugs`` with a usable label.

    Returns the drugs and the moieties that were not indexed (skipped or beyond the cut), whose names serve as
    out-of-scope drugs in the evaluation.
    """
    session = requests.Session()
    cands = candidate_moieties(fetch_counts(raw_dir, session))
    drugs: list[Drug] = []
    skipped: list[dict] = []
    for i, entry in enumerate(cands):
        if len(drugs) >= n_drugs:
            return drugs, skipped + cands[i:]
        labels = fetch_labels(entry["generic_name"], raw_dir, session)
        label = next((lb for lb in labels if label_ok(lb)), None)
        if label is None:
            skipped.append({**entry, "reason": "no label with the core sections"})
            continue
        drugs.append(build_drug(entry, label))
        if len(drugs) % 25 == 0:
            log(f"  {len(drugs)} drugs selected (rank {entry['rank']})")
    return drugs, skipped


def manifest(raw_dir: Path) -> dict:
    """sha256 of every cached raw file, so results can be tied to the exact downloads."""
    files = sorted(p for p in raw_dir.rglob("*.json"))
    h = hashlib.sha256()
    for p in files:
        h.update(p.name.encode())
        h.update(p.read_bytes())
    return {"n_files": len(files), "bytes": sum(p.stat().st_size for p in files), "sha256": h.hexdigest()}
