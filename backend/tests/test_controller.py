from helpers import decisions, feed, new_session, services_with


def test_wait_then_early_retrieve():
    r = feed(new_session(), [(0.0, "My internet isn't working."), (0.8, "The router has a red LOS light.")])
    assert decisions(r) == ["WAIT", "RETRIEVE"]
    d = r[1]["decision"]
    assert d["score"] >= 0.70 and set(d["features"]) == {"semantic_stability", "intent_confidence",
                                                          "entity_completeness", "information_gain", "novelty"}
    assert len(d["retrieve_intent_ids"]) == 2  # multi-intent retrieved in one parallel group


def test_final_chunk_forces_retrieval():
    r = feed(new_session(), [(0.0, "I can't connect to Wi-Fi.", True)])
    assert decisions(r) == ["RETRIEVE"] and r[0]["decision"]["reason"] == "utterance_complete"


def test_refine_on_late_constraint():
    r = feed(new_session(), [(0, "My internet isn't working."), (0.8, "The router has a red LOS light."),
                             (1.6, "My phone also can't connect."),
                             (2.4, "Actually mobile data works, only Wi-Fi doesn't.")])
    assert decisions(r)[-1] == "REFINE"
    assert r[-1]["decision"]["reason"] == "late_constraint"
    assert set(r[-1]["decision"]["refine_intent_ids"]) == {"I1", "I3"}


def test_suppress_without_retrieval():
    sess = new_session()
    r = feed(sess, [(0, "I can't connect to Wi-Fi.", True), (2, "Give me the previous answer in three bullet points.", True)])
    assert decisions(r) == ["RETRIEVE", "SUPPRESS"]
    t_suppress = [e for e in sess.bus.dicts() if e["type"] == "retrieval_suppressed"][0]["timestamp"]
    assert not [e for e in sess.bus.dicts() if e["type"] == "retrieval_started" and e["timestamp"] >= t_suppress]


def test_presentation_request_without_answer_waits():
    r = feed(new_session(), [(0, "Summarize the previous answer.", True)])
    assert decisions(r) == ["WAIT"] and r[0]["decision"]["reason"] == "no_grounded_answer_to_restructure"


def test_no_new_information_waits():
    r = feed(new_session(), [(0, "I can't connect to Wi-Fi.", True), (1, "ok thanks")])
    assert decisions(r)[-1] == "WAIT"


def test_threshold_is_configurable():
    strict = new_session(services_with(retrieval_threshold=0.99))
    r = feed(strict, [(0.0, "My internet isn't working."), (0.8, "The router has a red LOS light.")])
    assert decisions(r) == ["WAIT", "WAIT"]


def test_every_chunk_ablation_retrieves_each_chunk():
    r = feed(new_session(services_with(controller_mode="every_chunk")),
             [(0, "My internet isn't working."), (0.8, "The router has a red LOS light.")])
    assert decisions(r) == ["RETRIEVE", "RETRIEVE"]
