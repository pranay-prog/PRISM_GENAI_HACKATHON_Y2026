"""Data models for intents and the domain pack that configures them."""
from __future__ import annotations

import json
import re
from dataclasses import asdict, dataclass, field
from functools import lru_cache
from pathlib import Path


@dataclass
class IntentDef:
    label: str
    display: str
    triggers: list[tuple[re.Pattern, float]]
    slots: list[str]
    depends_on: list[str]
    base_query: str
    requires_any_fact: list[str] = field(default_factory=list)
    fact_terms: dict = field(default_factory=dict)  # intent-specific query terms per fact value


@dataclass
class FactDef:
    key: str
    value: object
    patterns: list[re.Pattern]


@dataclass
class DomainPack:
    domain: str
    intents: dict[str, IntentDef]
    facts: list[FactDef]
    fact_terms: dict[str, dict[str, str]]
    correction_markers: list[re.Pattern]
    presentation_patterns: list[re.Pattern]

    def terms_for(self, key: str, value) -> str:
        return self.fact_terms.get(key, {}).get(str(value).lower() if isinstance(value, bool) else str(value), "")


def _rx(p: str) -> re.Pattern:
    return re.compile(p, re.IGNORECASE)


@lru_cache(maxsize=8)
def load_domain_pack(path: Path) -> DomainPack:
    raw = json.loads(Path(path).read_text())
    intents = {
        i["label"]: IntentDef(
            label=i["label"], display=i["display"],
            triggers=[(_rx(p), float(w)) for p, w in i["triggers"]],
            slots=list(i.get("slots", [])), depends_on=list(i.get("depends_on", [])),
            base_query=i["base_query"], requires_any_fact=list(i.get("requires_any_fact", [])),
            fact_terms=i.get("fact_terms", {}),
        )
        for i in raw["intents"]
    }
    facts = [FactDef(f["key"], f["value"], [_rx(p) for p in f["patterns"]]) for f in raw["facts"]]
    return DomainPack(
        domain=raw["domain"], intents=intents, facts=facts, fact_terms=raw.get("fact_terms", {}),
        correction_markers=[_rx(p) for p in raw.get("correction_markers", [])],
        presentation_patterns=[_rx(p) for p in raw.get("presentation_patterns", [])],
    )


@dataclass
class DetectedIntent:
    label: str
    display: str
    confidence: float
    query: str
    slot_fill: float                # entity completeness for this intent, 0..1
    matched_triggers: list[str]
    intent_id: str = ""             # assigned by the session (stable across chunks)

    def to_dict(self) -> dict:
        return asdict(self)
