"""Interpretable retrieval score.

    score = 0.30 * semantic_stability
          + 0.25 * intent_confidence
          + 0.20 * entity_completeness
          + 0.15 * information_gain
          + 0.10 * novelty

Weights and thresholds come from config (env-overridable) so they can be
tuned from the benchmark.
"""
from __future__ import annotations

from dataclasses import asdict, dataclass

from ..config import ControllerWeights


@dataclass
class Features:
    semantic_stability: float
    intent_confidence: float
    entity_completeness: float
    information_gain: float
    novelty: float

    def to_dict(self) -> dict:
        return {k: round(v, 4) for k, v in asdict(self).items()}


def score(features: Features, weights: ControllerWeights) -> tuple[float, dict[str, float]]:
    contributions = {name: round(getattr(features, name) * w, 4) for name, w in asdict(weights).items()}
    return round(sum(contributions.values()), 4), contributions


def band(value: float, retrieve_threshold: float, monitor_threshold: float) -> str:
    if value >= retrieve_threshold:
        return "retrieve"
    if value >= monitor_threshold:
        return "monitor"
    return "wait"


def weakest_feature(features: Features) -> str:
    d = asdict(features)
    return min(d, key=d.get)
