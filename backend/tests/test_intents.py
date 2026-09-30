from helpers import services


def dec(text, facts=None):
    d = services().decomposer
    facts = facts if facts is not None else {f["key"]: f["value"] for f in d.extract_facts(text)}
    return d.decompose(text, facts)[0]


def test_multi_intent_decomposition():
    labels = [i.label for i in dec("My internet is down, the router has a red LOS light, and I want to know "
                                   "whether I qualify for outage compensation.")]
    assert set(labels) == {"connectivity_outage", "router_indicator", "outage_compensation"}


def test_single_intent_stays_single():
    assert [i.label for i in dec("How do I reset my password?")] == ["account_login"]


def test_no_over_fragmentation_from_shared_phrase():
    # "outage" inside a compensation question must not create a connectivity intent
    assert [i.label for i in dec("Will I get compensation for the outage?")] == ["outage_compensation"]


def test_context_gated_intent():
    # working from home in an IT-access context is a fact, not a telecom remote-work intent
    assert "remote_work_support" not in [i.label for i in dec("I'm working from home and my VPN is connected")]


def test_fact_extraction_and_query_building():
    d = services().decomposer
    facts = {f["key"]: f["value"] for f in d.extract_facts("Actually mobile data works, only Wi-Fi doesn't.")}
    assert facts["mobile_data_working"] is True and facts["failure_scope"] == "wifi_only"
    q = d.build_query("outage_compensation", facts)
    assert "exclusions" in q and "compensation" in q


def test_presentation_only_detection():
    d = services().decomposer
    for t in ("Summarize the previous answer.", "Give that in 3 bullets.", "Make the previous answer shorter.",
              "Explain the same answer more simply."):
        assert d.presentation_only(t), t
    assert not d.presentation_only("Summarize the billing policy for cancellations")
