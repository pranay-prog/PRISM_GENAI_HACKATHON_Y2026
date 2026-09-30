"""Tokenisation shared by BM25, the LSA fallback embedder, the lexical reranker
and the grounding check. Deliberately simple and deterministic."""
from __future__ import annotations

import re

STOPWORDS = frozenset(
    """a an the and or but if then else of to in on at by for with from into onto over under
    is are was were be been being am do does did doing have has had having i me my mine we our
    you your he she it its they them their this that these those there here what which who whom
    whose when where why how can could would should will shall may might must not no nor so too
    very just also only than as about up down out off again once all any both each few more most
    other some such own same s t don doesn didn isn aren wasn weren won wouldn shouldn couldn
    im ive id youre dont cant wont cannot please get got getting also still even really actually""".split()
)

_NORMALISE = [
    (re.compile(r"\bwi[\s-]?fi\b", re.I), "wifi"),
    (re.compile(r"\be[\s-]?mail\b", re.I), "email"),
    (re.compile(r"\blog[\s-]?in\b", re.I), "login"),
    (re.compile(r"\bsign[\s-]?in\b", re.I), "signin"),
    (re.compile(r"\b2fa\b", re.I), "mfa"),
    (re.compile(r"\btwo[\s-]factor\b", re.I), "mfa"),
    (re.compile(r"\bmulti[\s-]factor( authentication)?\b", re.I), "mfa"),
    (re.compile(r"\bcan'?t\b", re.I), "cannot"),
    (re.compile(r"\bisn'?t\b", re.I), "not"),
    (re.compile(r"\bdoesn'?t\b|\bdon'?t\b|\bwon'?t\b", re.I), "not"),
]
_TOKEN = re.compile(r"[a-z0-9]+")


def normalise(text: str) -> str:
    out = text
    for pat, rep in _NORMALISE:
        out = pat.sub(rep, out)
    return out.lower()


def stem(tok: str) -> str:
    if len(tok) <= 4 or tok.isdigit():
        return tok
    for suf, rep in (("ies", "y"), ("ing", ""), ("ed", ""), ("es", "e"), ("s", "")):
        if tok.endswith(suf) and len(tok) - len(suf) >= 3 and not tok.endswith("ss"):
            return tok[: len(tok) - len(suf)] + rep
    return tok


def tokenize(text: str, drop_stopwords: bool = True) -> list[str]:
    toks = _TOKEN.findall(normalise(text))
    if drop_stopwords:
        toks = [t for t in toks if t not in STOPWORDS]
    return [stem(t) for t in toks]


def sentences(text: str) -> list[str]:
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", text.strip())
    return [p.strip() for p in parts if p.strip()]
