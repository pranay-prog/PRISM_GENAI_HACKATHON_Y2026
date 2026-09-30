"""Prompt templates used when an LLM provider is configured."""
from __future__ import annotations

import json

GROUNDING_RULES = """Rules:
- Use ONLY the evidence passages below. Do not use outside knowledge.
- Every claim must cite one or more chunk_ids from the evidence it is based on.
- Do not invent facts, numbers, times or policies.
- If the evidence does not answer the question, add it to "uncertainties" instead of guessing.
- Keep each claim to one sentence and keep citations attached to the claim they support."""


def decomposition_prompt(transcript: str, labels: dict[str, str]) -> str:
    return f"""You split a customer support utterance into independent retrieval intents.
Allowed intent labels (use only these): {json.dumps(labels)}
Do not over-fragment: one question is one intent. Only include intents the customer actually expressed.
Return JSON: {{"intents": [{{"label": "<allowed label>", "query": "<short search query>"}}]}}

Utterance: {transcript}"""


def answer_prompt(transcript: str, facts: dict, sections: list[dict]) -> str:
    blocks = []
    for s in sections:
        ev = "\n".join(f"[{e['chunk_id']}] ({e['title']}) {e['text']}" for e in s["evidence"])
        blocks.append(f"INTENT {s['intent_id']} - {s['display']}\nEvidence:\n{ev}")
    return f"""You are an agent-assist system answering a telecom support conversation.
{GROUNDING_RULES}

Conversation so far: {transcript}
Known facts: {json.dumps(facts)}

{chr(10).join(blocks)}

Return JSON: {{"sections": [{{"intent_id": "I1", "claims": [{{"text": "...", "chunk_ids": ["..."]}}]}}], "uncertainties": ["..."]}}
Write 1-3 claims per intent, in the order given."""


def restructure_prompt(instruction: str, claims: list[dict]) -> str:
    return f"""Rewrite the grounded answer below according to the instruction.
Do not add information. Keep each rewritten claim's chunk_ids from the claim(s) it came from.
Instruction: {instruction}
Claims: {json.dumps(claims)}
Return JSON: {{"claims": [{{"text": "...", "chunk_ids": ["..."]}}]}}"""
