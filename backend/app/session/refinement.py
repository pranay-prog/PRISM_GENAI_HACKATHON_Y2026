"""Late-arriving constraints.

apply_facts()          merges facts from a new chunk into the session and
                       returns only real changes (new key or changed value).
analyse_constraint()   explains a REFINE: which constraint arrived, which
                       earlier assumption it invalidates, which answered
                       intents depend on it, and what the new focus query is.
"""
from __future__ import annotations

from ..intents.decomposer import IntentDecomposer
from .state import SessionState


def apply_facts(state: SessionState, extracted: list[dict], chunk_index: int) -> list[dict]:
    changes = []
    for f in extracted:
        old = state.facts.get(f["key"])
        if old == f["value"]:
            continue
        state.facts[f["key"]] = f["value"]
        change = {"key": f["key"], "old": old, "new": f["value"], "chunk_index": chunk_index,
                  "evidence": f["evidence"]}
        state.fact_history.append(change)
        changes.append(change)
    return changes


def analyse_constraint(state: SessionState, changes: list[dict], affected_labels: list[str],
                       decomposer: IntentDecomposer, chunk_text: str) -> dict:
    pack = decomposer.pack
    invalidated = [
        {"fact": c["key"], "previous": c["old"], "now": c["new"]} for c in changes if c["old"] is not None
    ]
    affected = []
    for label in affected_labels:
        rec = state.intents[label]
        deps = set(pack.intents[label].depends_on)
        affected.append({
            "intent_id": rec.intent_id, "label": label,
            "because_of": [c["key"] for c in changes if c["key"] in deps],
            "previous_query": rec.query,
            "new_query": decomposer.build_query(label, state.facts),
        })
    return {
        "constraint_text": chunk_text,
        "new_constraints": [{"fact": c["key"], "value": c["new"], "evidence": c["evidence"]} for c in changes],
        "invalidated_assumptions": invalidated,
        "correction_marker": decomposer.has_correction_marker(chunk_text),
        "affected_intents": affected,
    }
