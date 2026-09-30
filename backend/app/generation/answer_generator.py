"""AnswerGenerator - builds citation-grounded answer versions.

Each answer is organised in sections, one per intent. On refinement only the
sections of affected / new intents are regenerated; every other section is
carried over unchanged from the previous version, so established context is
preserved and the diff between versions is explicit.

Deterministic mode (no LLM) is extractive: claims are sentences selected from
the retrieved chunks, so every claim is verbatim grounded in the corpus. An
intent whose best evidence does not cover the query terms well enough is
reported as an uncertainty instead of being answered.

LLM mode asks the provider for JSON claims with chunk_ids, then validates
every citation against the evidence actually supplied; claims without a valid
citation are removed and reported as uncertainties.
"""
from __future__ import annotations

import json
import logging
import re

from ..retrieval.reranker import term_coverage
from ..retrieval.text import sentences
from .prompts import answer_prompt, restructure_prompt

log = logging.getLogger(__name__)

_NUM_WORDS = {"one": 1, "two": 2, "three": 3, "four": 4, "five": 5, "six": 6}


def _cite(e: dict) -> dict:
    return {"chunk_id": e["chunk_id"], "document_id": e["document_id"], "section": e["section"]}


class AnswerGenerator:
    def __init__(self, provider, idf: dict[str, float], min_coverage: float = 0.34,
                 claims_per_intent: int = 3):
        self.provider = provider
        self.idf = idf
        self.min_coverage = min_coverage
        self.claims_per_intent = claims_per_intent

    # ================================================================ public
    def generate(self, *, intents: list[dict], evidence_by_intent: dict[str, list[dict]], facts: dict,
                 transcript: str, previous: dict | None, regenerate_ids: set[str], trigger: str,
                 timestamp: float) -> dict:
        prev_sections = {s["intent_id"]: s for s in (previous or {}).get("sections", [])}
        to_build = [i for i in intents if i["intent_id"] in regenerate_ids or i["intent_id"] not in prev_sections]
        carried_claims = {c["claim"] for iid, sec in prev_sections.items()
                          if iid not in {i["intent_id"] for i in to_build} for c in sec.get("claims", [])}
        built, generator_used, usage = self._build_sections(to_build, evidence_by_intent, facts, transcript,
                                                            carried_claims)
        sections = []
        for i in intents:
            if i["intent_id"] in built:
                sections.append(built[i["intent_id"]])
            else:
                carried = dict(prev_sections[i["intent_id"]])
                carried["regenerated"] = False
                sections.append(carried)
        version = self._assemble(sections, previous, trigger, timestamp,
                                 affected=[i["intent_id"] for i in to_build])
        version["generator"] = generator_used
        version["token_usage"] = usage
        return version

    def restructure(self, previous: dict, instruction: str, timestamp: float) -> dict:
        """Presentation-only change. No retrieval: reuses the grounded claims."""
        claims = [c for s in previous["sections"] if s["supported"] for c in s["claims"]]
        style, n = self._parse_style(instruction)
        new_claims, generator_used = None, "deterministic-restructure"
        if getattr(self.provider, "is_llm", False):
            new_claims = self._llm_restructure(instruction, claims)
            generator_used = self.provider.name if new_claims else generator_used
        if not new_claims:
            new_claims = self._deterministic_restructure(previous["sections"], style, n)
        section = {"intent_id": "ALL", "label": "restructured", "display": f"Restructured ({style})",
                   "supported": True, "regenerated": True, "claims": new_claims}
        version = self._assemble([section], previous, "presentation_restructure", timestamp, affected=[])
        version["format"] = style
        version["source_version"] = previous["version"]
        version["uncertainties"] = list(previous.get("uncertainties", []))
        version["generator"] = generator_used
        return version

    # ======================================================= section building
    def _build_sections(self, intents, evidence_by_intent, facts, transcript, carried_claims=frozenset()):
        built: dict[str, dict] = {}
        usage: dict = {}
        generator_used = "deterministic-extractive"
        if intents and getattr(self.provider, "is_llm", False):
            try:
                built = self._llm_sections(intents, evidence_by_intent, facts, transcript)
                usage = getattr(self.provider, "last_usage", {})
                generator_used = self.provider.name
            except Exception as exc:
                log.warning("LLM generation failed (%s); using extractive generator", exc)
                built = {}
        used_sentences: set[str] = set(carried_claims)
        for s in built.values():
            used_sentences.update(c["claim"] for c in s["claims"])
        for i in intents:
            if i["intent_id"] not in built:
                built[i["intent_id"]] = self._extractive_section(i, evidence_by_intent.get(i["intent_id"], []),
                                                                 used_sentences)
        return built, generator_used, usage

    def grounding_score(self, query: str, evidence: list[dict]) -> float:
        return max((term_coverage(query, f"{e.get('title', '')} {e['text']}", self.idf) for e in evidence[:3]),
                   default=0.0)

    def _extractive_section(self, intent: dict, evidence: list[dict], used: set[str]) -> dict:
        base = {"intent_id": intent["intent_id"], "label": intent["label"], "display": intent["display"],
                "regenerated": True}
        g = self.grounding_score(intent["query"], evidence)
        if not evidence or g < self.min_coverage:
            return {**base, "supported": False, "grounding": round(g, 3), "claims": [],
                    "uncertainty": f"The knowledge base does not contain enough evidence to answer: "
                                   f"{intent['display'].lower()} (best evidence coverage {g:.2f}).",
                    "evidence_considered": [e["chunk_id"] for e in evidence[:3]]}
        scored = []
        for rank, e in enumerate(evidence[:3]):
            for pos, sent in enumerate(sentences(e["text"])):
                if sent in used or len(sent.split()) < 5:
                    continue
                cov = term_coverage(intent["query"], sent, self.idf)
                # a claim must speak to the intent itself, not only to context facts
                if cov <= 0 or term_coverage(intent.get("core_query", intent["query"]), sent, self.idf) <= 0:
                    continue
                scored.append((cov + 0.15 * float(e.get("rerank_score") or 0) - 0.05 * rank, rank, pos, sent, e))
        top = sorted(scored, key=lambda x: -x[0])[: self.claims_per_intent]
        top.sort(key=lambda x: (x[1], x[2]))  # read in document order
        claims = []
        for _, _, _, sent, e in top:
            used.add(sent)
            claims.append({"claim": sent, "citations": [_cite(e)]})
        if not claims:
            return {**base, "supported": False, "grounding": round(g, 3), "claims": [],
                    "uncertainty": f"Retrieved passages do not contain a direct statement about "
                                   f"{intent['display'].lower()}."}
        return {**base, "supported": True, "grounding": round(g, 3), "claims": claims}

    def _llm_sections(self, intents, evidence_by_intent, facts, transcript) -> dict[str, dict]:
        payload = [{**i, "evidence": evidence_by_intent.get(i["intent_id"], [])[:4]} for i in intents]
        raw = self.provider.complete(answer_prompt(transcript, facts, payload), json_mode=True)
        data = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip()))
        out = {}
        for sec in data.get("sections", []):
            iid = sec.get("intent_id")
            intent = next((i for i in intents if i["intent_id"] == iid), None)
            if intent is None:
                continue
            allowed = {e["chunk_id"]: e for e in evidence_by_intent.get(iid, [])}
            claims, dropped = [], []
            for c in sec.get("claims", []):
                cites = [_cite(allowed[cid]) for cid in c.get("chunk_ids", []) if cid in allowed]
                (claims if cites else dropped).append({"claim": str(c.get("text", "")).strip(), "citations": cites})
            if claims:
                out[iid] = {"intent_id": iid, "label": intent["label"], "display": intent["display"],
                            "regenerated": True, "supported": True, "claims": claims,
                            "grounding": round(self.grounding_score(intent["query"], list(allowed.values())), 3),
                            "dropped_unsupported_claims": [d["claim"] for d in dropped]}
        return out

    # ========================================================== restructure
    @staticmethod
    def _parse_style(text: str) -> tuple[str, int]:
        t = text.lower()
        m = re.search(r"\b(\d+|one|two|three|four|five|six)\b", t)
        n = (int(m.group(1)) if m.group(1).isdigit() else _NUM_WORDS[m.group(1)]) if m else 3
        if "bullet" in t or "point" in t:
            return "bullets", max(1, min(n, 10))
        if "simpl" in t:
            return "simpler", n
        return "summary", n

    @staticmethod
    def _deterministic_restructure(sections: list[dict], style: str, n: int) -> list[dict]:
        supported = [s for s in sections if s["supported"] and s["claims"]]
        if style == "bullets":
            out, depth = [], 0
            while len(out) < n and any(depth < len(s["claims"]) for s in supported):
                for s in supported:
                    if depth < len(s["claims"]) and len(out) < n:
                        out.append(dict(s["claims"][depth]))
                depth += 1
            return out
        # summary / simpler: the leading claim of every supported section
        return [dict(s["claims"][0]) for s in supported]

    def _llm_restructure(self, instruction: str, claims: list[dict]) -> list[dict] | None:
        try:
            simple = [{"text": c["claim"], "chunk_ids": [x["chunk_id"] for x in c["citations"]]} for c in claims]
            raw = self.provider.complete(restructure_prompt(instruction, simple), json_mode=True)
            data = json.loads(re.sub(r"^```(json)?|```$", "", raw.strip()))
            allowed = {x["chunk_id"]: x for c in claims for x in c["citations"]}
            out = []
            for c in data.get("claims", []):
                cites = [allowed[cid] for cid in c.get("chunk_ids", []) if cid in allowed]
                if cites:
                    out.append({"claim": str(c["text"]).strip(), "citations": cites})
            return out or None
        except Exception as exc:
            log.warning("LLM restructure failed (%s); using deterministic restructure", exc)
            return None

    # ============================================================ assembly
    @staticmethod
    def _assemble(sections, previous, trigger, timestamp, affected) -> dict:
        prev_claims = {c["claim"] for s in (previous or {}).get("sections", []) for c in s.get("claims", [])}
        claims = [c for s in sections for c in s.get("claims", [])]
        cur_claims = {c["claim"] for c in claims}
        lines = []
        for s in sections:
            lines.append(s["display"] + ":")
            if s["supported"]:
                for c in s["claims"]:
                    refs = ", ".join(x["chunk_id"] for x in c["citations"])
                    lines.append(f"- {c['claim']} [{refs}]")
            else:
                lines.append(f"- {s['uncertainty']}")
        citations = [{"claim": c["claim"], **x} for c in claims for x in c["citations"]]
        prev_ids = {s["intent_id"] for s in (previous or {}).get("sections", [])}
        return {
            "version": (previous["version"] + 1) if previous else 1,
            "timestamp": timestamp,
            "trigger": trigger,
            "answer_text": "\n".join(lines),
            "sections": sections,
            "citations": citations,
            "uncertainties": [s["uncertainty"] for s in sections if not s["supported"]],
            "affected_intents": affected,
            "citation_coverage": round(sum(1 for c in claims if c["citations"]) / len(claims), 4) if claims else None,
            "changes": {
                "added_intents": [s["intent_id"] for s in sections if s["intent_id"] not in prev_ids],
                "regenerated_intents": [s["intent_id"] for s in sections if s.get("regenerated")],
                "carried_over_intents": [s["intent_id"] for s in sections if not s.get("regenerated")],
                "added_claims": sorted(cur_claims - prev_claims),
                "removed_claims": sorted(prev_claims - cur_claims),
            },
        }
