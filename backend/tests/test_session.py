import tempfile
from pathlib import Path

from app.db.database import Database
from app.session.manager import SessionManager, SessionNotFound
from app.session.refinement import apply_facts
from app.session.state import SessionState
from helpers import feed, new_session


def test_fact_updates_record_changes():
    st = SessionState("S")
    ch = apply_facts(st, [{"key": "failure_scope", "value": "all_devices", "evidence": "x"}], 0)
    assert ch[0]["old"] is None
    ch = apply_facts(st, [{"key": "failure_scope", "value": "wifi_only", "evidence": "y"}], 1)
    assert ch[0]["old"] == "all_devices" and st.facts["failure_scope"] == "wifi_only"
    assert apply_facts(st, [{"key": "failure_scope", "value": "wifi_only", "evidence": "y"}], 2) == []
    assert len(st.fact_history) == 2


def test_state_persists_in_sqlite():
    with tempfile.TemporaryDirectory() as d:
        db = Database(Path(d) / "t.sqlite3")
        mgr = SessionManager(db)
        sess = new_session(sid="PERSIST")
        feed(sess, [(0, "I can't connect to Wi-Fi.", True)])
        mgr.save(sess.state)
        loaded = SessionManager(db).get("PERSIST")
        assert loaded.facts == sess.state.facts
        assert loaded.intents["wifi_connectivity"].status == "answered"
        assert loaded.answer_versions[0]["citations"]
        try:
            mgr.get("missing")
            raise AssertionError
        except SessionNotFound:
            pass
        db.close()


def test_late_constraint_event_and_answer_versions():
    sess = new_session()
    feed(sess, [(0, "My internet isn't working."), (0.8, "The router has a red LOS light."),
                (1.6, "My phone also can't connect."), (2.4, "Actually mobile data works, only Wi-Fi doesn't."),
                (3.2, "Also, will I get compensation for the outage?", True)])
    lc = [e for e in sess.bus.dicts() if e["type"] == "late_constraint"][0]["payload"]
    assert {"fact": "failure_scope", "previous": "all_devices", "now": "wifi_only"} in lc["invalidated_assumptions"]
    vs = sess.state.answer_versions
    assert [v["version"] for v in vs] == [1, 2, 3, 4]
    assert [v["trigger"] for v in vs] == ["initial_retrieval", "new_intent", "late_constraint", "new_intent"]
    v3 = vs[2]
    assert set(v3["changes"]["regenerated_intents"]) == {"I1", "I3"}
    assert v3["changes"]["carried_over_intents"] == ["I2"]  # unaffected section preserved
    for v in vs:
        assert {"version", "timestamp", "answer_text", "citations", "affected_intents", "trigger"} <= set(v)
