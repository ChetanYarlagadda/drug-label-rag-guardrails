"""In-process check of the FastAPI service with the real engine (CPU encoders, cached LLM replies).

    python scripts/api_smoke.py      # writes reports/results/api_smoke.json

The questions are test questions whose LLM replies are already in data/cache/llm_cache.sqlite, so the check makes
no new model calls unless the cache is empty.
"""

from __future__ import annotations

import json
import sys
import time
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pandas as pd  # noqa: E402
from fastapi.testclient import TestClient  # noqa: E402

from app.main import create_app, load_engine  # noqa: E402

QIDS = ["test_ans_half_life_06", "test_adv_off_label_00", "test_oos_general_00"]


def main() -> None:
    ev = pd.read_csv(ROOT / "reports" / "results" / "eval_set.csv").set_index("qid")
    out = {"requests": []}
    with TestClient(create_app(load_engine)) as client:
        out["health"] = client.get("/health").json()
        for qid in QIDS:
            if qid not in ev.index or isinstance(ev.loc[qid, "context_drug"], str):
                continue  # wrong-drug items need an injected context, which the public API does not allow
            q = ev.loc[qid, "question"]
            t0 = time.perf_counter()
            body = client.post("/ask", json={"question": q}).json()
            body["latency_ms"] = round((time.perf_counter() - t0) * 1000)
            for e in body["excerpts"]:
                e["text"] = e["text"][:160] + "..."
            out["requests"].append({"qid": qid, "response": body})
        t0 = time.perf_counter()
        s = client.post("/search", json={"question": "Can warfarin be used during pregnancy?", "k": 3}).json()
        out["search"] = {"question": s["question"], "detected_drugs": s["detected_drugs"],
                         "hits": [(e["chunk_id"], e["score"]) for e in s["excerpts"]],
                         "latency_ms": round((time.perf_counter() - t0) * 1000)}
    path = ROOT / "reports" / "results" / "api_smoke.json"
    path.write_text(json.dumps(out, indent=2))
    print(json.dumps(out, indent=2)[:3000])


if __name__ == "__main__":
    main()
