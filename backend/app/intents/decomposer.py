"""Multi-intent decomposition.

Deterministic mode (default, no LLM needed):
  1. every intent trigger in the domain pack is matched against the transcript;
  2. overlapping spans are claimed by the highest-weight trigger only, so one
     phrase cannot create two intents (over-fragmentation guard);
  3. intent confidence = min(1, sum of its trigger weights);
  4. intents below `min_confidence` are reported as candidates, not intents;
  5. each intent gets a focused sub-query = base query + terms from the facts
     it depends on.

LLM mode: the provider proposes intent labels/queries as JSON. Output is
validated against the domain pack; malformed or unknown output falls back to
the deterministic result.
"""
from __future__ import annotations

import json
import logging
import re

from .intent_models import DetectedIntent, DomainPack

log = logging.getLogger(__name__)


class IntentDecomposer:
    def __init__(self, pack: DomainPack, min_confidence: float = 0.5, candidate_confidence: float = 0.3,
                 provider=None):
        self.pack = pack
        self.min_confidence = min_confidence
        self.candidate_confidence = candidate_confidence
        self.provider = provider

    # ------------------------------------------------------------------ facts
    def extract_facts(self, text: str) -> list[dict]:
        """Facts stated in `text`, in order of appearance. Later mentions of the
        same key override earlier ones when applied to the session."""
        found = []
        for fdef in self.pack.facts:
            for pat in fdef.patterns:
                m = pat.search(text)
                if m:
                    found.append({"key": fdef.key, "value": fdef.value, "evidence": m.group(0), "pos": m.start()})
                    break
        found.sort(key=lambda f: f["pos"])
        return found

    # ---------------------------------------------------------------- queries
    def build_query(self, label: str, facts: dict) -> str:
        idef = self.pack.intents[label]
        parts = [idef.base_query]
        for key in dict.fromkeys(idef.slots + idef.depends_on):
            if key in facts:
                val = facts[key]
                sval = str(val).lower() if isinstance(val, bool) else str(val)
                override = idef.fact_terms.get(key)
                t = override.get(sval, "") if override is not None else self.pack.terms_for(key, val)
                if t:
                    parts.append(t)
        words, seen = [], set()
        for w in " ".join(parts).split():
            if w.lower() not in seen:
                seen.add(w.lower())
                words.append(w)
        return " ".join(words)

    def slot_fill(self, label: str, facts: dict) -> float:
        slots = self.pack.intents[label].slots
        return 1.0 if not slots else sum(1 for s in slots if s in facts) / len(slots)

    # -------------------------------------------------------------- decompose
    def _match_spans(self, text: str) -> dict[str, list[tuple[str, float]]]:
        matches = []
        for label, idef in self.pack.intents.items():
            for pat, w in idef.triggers:
                for m in pat.finditer(text):
                    matches.append((w, m.start(), m.end(), label, m.group(0)))
        matches.sort(key=lambda x: (-x[0], x[1]))
        claimed: list[tuple[int, int, str]] = []
        per_label: dict[str, list[tuple[str, float]]] = {}
        for w, s, e, label, txt in matches:
            if any(s < ce and cs < e and cl != label for cs, ce, cl in claimed):
                continue  # span already owned by a stronger trigger of another intent
            claimed.append((s, e, label))
            per_label.setdefault(label, []).append((txt, w))
        return per_label

    def decompose(self, text: str, facts: dict) -> tuple[list[DetectedIntent], list[DetectedIntent]]:
        """Returns (intents, candidates)."""
        intents, candidates = [], []
        for label, hits in self._match_spans(text).items():
            idef = self.pack.intents[label]
            if idef.requires_any_fact and not any(k in facts for k in idef.requires_any_fact):
                continue
            # the same trigger phrase repeated does not add confidence
            by_trigger: dict[str, float] = {}
            for txt, w in hits:
                by_trigger[txt.lower()] = max(by_trigger.get(txt.lower(), 0), w)
            conf = round(min(1.0, sum(by_trigger.values())), 3)
            di = DetectedIntent(label, idef.display, conf, self.build_query(label, facts),
                                round(self.slot_fill(label, facts), 3), sorted(by_trigger))
            if conf >= self.min_confidence:
                intents.append(di)
            elif conf >= self.candidate_confidence:
                candidates.append(di)
        order = {lbl: i for i, lbl in enumerate(self.pack.intents)}
        intents.sort(key=lambda d: order[d.label])
        if self.provider is not None and getattr(self.provider, "is_llm", False):
            intents = self._llm_refine(text, facts, intents)
        return intents, candidates

    def presentation_only(self, text: str) -> bool:
        """True when the utterance only asks to re-present an existing answer."""
        if not any(p.search(text) for p in self.pack.presentation_patterns):
            return False
        intents, _ = self.decompose(text, {})
        return not intents and not self.extract_facts(text)

    def has_correction_marker(self, text: str) -> bool:
        return any(p.search(text) for p in self.pack.correction_markers)

    # ----------------------------------------------------------------- LLM
    def _llm_refine(self, text: str, facts: dict, fallback: list[DetectedIntent]) -> list[DetectedIntent]:
        from ..generation.prompts import decomposition_prompt

        labels = {l: d.display for l, d in self.pack.intents.items()}
        try:
            raw = self.provider.complete(decomposition_prompt(text, labels), json_mode=True)
            data = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip()))
            out = []
            for item in data.get("intents", []):
                label = item.get("label")
                if label not in self.pack.intents:
                    continue
                det = next((d for d in fallback if d.label == label), None)
                query = str(item.get("query") or self.build_query(label, facts))[:300]
                out.append(DetectedIntent(label, self.pack.intents[label].display,
                                          det.confidence if det else 0.6, query,
                                          self.slot_fill(label, facts), det.matched_triggers if det else ["llm"]))
            return out or fallback
        except Exception as exc:
            log.warning("LLM decomposition failed (%s); using deterministic intents", exc)
            return fallback
