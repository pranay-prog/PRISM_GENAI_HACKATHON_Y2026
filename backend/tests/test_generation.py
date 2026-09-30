from helpers import feed, new_session, services


def test_citations_point_to_real_retrieved_chunks():
    sess = new_session()
    feed(sess, [(0, "My router has a red LOS light and the internet is down", True)])
    ans = sess.state.current_answer
    kb = services().kb
    assert ans["citations"] and ans["citation_coverage"] == 1.0
    for c in ans["citations"]:
        chunk = kb.chunk_by_id[c["chunk_id"]]
        assert chunk.document_id == c["document_id"] and chunk.section == c["section"]
        assert c["chunk_id"] in sess.state.evidence
        assert c["claim"] in chunk.text  # extractive: claim is verbatim grounded


def test_unsupported_question_marked_uncertain():
    sess = new_session()
    feed(sess, [(0, "Can you recommend a good pizza place near me?", True)])
    ans = sess.state.current_answer
    assert ans["uncertainties"] and not ans["citations"]
    assert ans["citation_coverage"] is None


def test_restructure_reuses_citations_only():
    sess = new_session()
    feed(sess, [(0, "I can't connect to Wi-Fi.", True), (1, "Summarize that in two bullet points", True)])
    v1, v2 = sess.state.answer_versions
    assert v2["trigger"] == "presentation_restructure" and v2["format"] == "bullets"
    assert len(v2["sections"][0]["claims"]) == 2
    assert {c["chunk_id"] for c in v2["citations"]} <= {c["chunk_id"] for c in v1["citations"]}
