from app.telemetry.events import EVENT_TYPES
from app.telemetry.metrics import compute_metrics
from helpers import feed, new_session


def test_all_major_events_recorded():
    sess = new_session()
    feed(sess, [(0, "My internet isn't working."), (0.8, "The router has a red LOS light."),
                (1.6, "Actually mobile data works, only Wi-Fi doesn't."),
                (2.4, "Give me the previous answer in three bullet points.", True)])
    types = {e["type"] for e in sess.bus.dicts()}
    for t in ("transcript_chunk", "controller_decision", "intent_detected", "subquery_created",
              "retrieval_started", "retrieval_completed", "fusion_completed", "rerank_completed",
              "session_updated", "late_constraint", "answer_generated", "retrieval_suppressed"):
        assert t in types, t
    assert types <= set(EVENT_TYPES)
    ev = sess.bus.dicts()[0]
    assert {"event_id", "session_id", "timestamp", "type", "payload"} == set(ev)
    ts = [e["timestamp"] for e in sess.bus.dicts()]
    assert ts == sorted(ts)


def test_metrics_derived_from_events():
    sess = new_session()
    feed(sess, [(0, "My internet isn't working."), (0.8, "The router has a red LOS light.", True)])
    m = compute_metrics(sess.bus.dicts())
    assert m["retrieval_calls"] == 2 and m["decision_counts"] == {"WAIT": 1, "RETRIEVE": 1}
    assert 0.8 <= m["first_retrieval_latency_s"] < 2.0
    assert m["answer_versions"] == 1 and m["citation_coverage"] == 1.0
    assert all("total" in l for l in m["retrieval_latency_ms"])
