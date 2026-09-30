"""Scenarios.

The three demo scenarios drive Demo Mode. All scenarios carry gold labels used
ONLY by the evaluator (intent labels, relevant documents per intent, expected
controller decisions). The engine never sees these labels.
"""
from __future__ import annotations

SCENARIOS: list[dict] = [
    {
        "id": "outage_wifi_compensation",
        "title": "Outage + Wi-Fi + Compensation",
        "demo": True,
        "description": "Evolving broadband fault with a late correction and a late compensation question.",
        "chunks": [
            {"time": 0.0, "text": "My internet isn't working."},
            {"time": 0.8, "text": "The router has a red LOS light."},
            {"time": 1.6, "text": "My phone also can't connect."},
            {"time": 2.4, "text": "Actually mobile data works, only Wi-Fi doesn't."},
            {"time": 3.2, "text": "Also, will I get compensation for the outage?", "final": True},
        ],
        "gold": {
            "intents": {
                "connectivity_outage": ["NET-001", "NET-002", "NET-016", "NET-010", "DEV-006"],
                "router_indicator": ["NET-014", "NET-004"],
                "wifi_connectivity": ["NET-010", "NET-009", "DEV-006"],
                "outage_compensation": ["POL-001", "POL-006"],
            },
            "expected_decisions": ["WAIT", "RETRIEVE", "RETRIEVE", "REFINE", "RETRIEVE"],
            "refinement": {"chunk_index": 3, "focus_docs": ["NET-010"]},
        },
    },
    {
        "id": "vpn_mfa_finance",
        "title": "VPN + Authentication + Finance Access",
        "demo": True,
        "description": "Business remote-access problem: VPN up, MFA failing, then the real target emerges.",
        "chunks": [
            {"time": 0.0, "text": "I can't access the company dashboard."},
            {"time": 0.9, "text": "My VPN is connected."},
            {"time": 1.8, "text": "I'm getting an MFA error."},
            {"time": 2.7, "text": "I'm working from home."},
            {"time": 3.6, "text": "I need access to the finance dashboard.", "final": True},
        ],
        "gold": {
            "intents": {
                "application_access": ["BIZ-004", "BIZ-006"],
                "vpn_access": ["BIZ-002", "BIZ-001", "BIZ-005"],
                "mfa_error": ["BIZ-003", "ACC-004"],
            },
            "expected_decisions": ["WAIT", "RETRIEVE", "RETRIEVE", "REFINE", "REFINE"],
            "refinement": {"chunk_index": 4, "focus_docs": ["BIZ-004"]},
        },
    },
    {
        "id": "query_suppression",
        "title": "Query Suppression",
        "demo": True,
        "description": "A grounded answer, then a presentation-only follow-up that must not trigger retrieval.",
        "chunks": [
            {"time": 0.0, "text": "I can't connect to Wi-Fi.", "final": True},
            {"time": 2.0, "text": "Give me the previous answer in three bullet points.", "final": True},
        ],
        "gold": {
            "intents": {"wifi_connectivity": ["NET-009", "NET-010", "DEV-006"]},
            "expected_decisions": ["RETRIEVE", "SUPPRESS"],
        },
    },
    {
        "id": "password_single_intent",
        "title": "Single intent (password)",
        "demo": False,
        "chunks": [
            {"time": 0.0, "text": "I forgot my password"},
            {"time": 0.9, "text": "and I can't sign in to my account.", "final": True},
        ],
        "gold": {"intents": {"account_login": ["ACC-002", "ACC-001", "ACC-006"]}},
    },
    {
        "id": "slow_speed_upgrade",
        "title": "Slow speed + upgrade question",
        "demo": False,
        "chunks": [
            {"time": 0.0, "text": "My internet is really slow in the evenings."},
            {"time": 1.0, "text": "I'm thinking about upgrading to a faster plan."},
            {"time": 1.9, "text": "Would that fix the problem?", "final": True},
        ],
        "gold": {"intents": {"slow_speed": ["NET-005", "NET-006"], "plan_change": ["PLN-003", "PLN-002"]}},
    },
    {
        "id": "remote_worker_outage",
        "title": "Remote worker outage + technician",
        "demo": False,
        "chunks": [
            {"time": 0.0, "text": "The internet is down at home."},
            {"time": 0.9, "text": "I work from home so I need it fixed fast."},
            {"time": 1.8, "text": "Can you send a technician?", "final": True},
        ],
        "gold": {
            "intents": {
                "connectivity_outage": ["NET-001", "NET-002", "NET-016"],
                "remote_work_support": ["POL-004"],
                "technician_visit": ["SVC-001"],
            }
        },
    },
    {
        "id": "out_of_domain",
        "title": "Out-of-corpus question",
        "demo": False,
        "chunks": [{"time": 0.0, "text": "Can you recommend a good pizza place near me?", "final": True}],
        "gold": {"intents": {}, "expect_uncertainty": True},
    },
]

# Query -> relevant documents, for the dense vs BM25 vs hybrid ablation.
RETRIEVAL_EVAL: list[tuple[str, list[str]]] = [
    ("red LOS light on the ONT", ["NET-014", "NET-004"]),
    ("the fiber box shows loss of signal", ["NET-014", "NET-004"]),
    ("no internet on any device at home", ["NET-001", "NET-006"]),
    ("is there an outage in my area", ["NET-002", "NET-016"]),
    ("wifi broken but 4G on my phone is fine", ["NET-010"]),
    ("phone won't join the wireless network", ["NET-009"]),
    ("old smart plug cannot connect to wireless", ["NET-011", "DEV-004"]),
    ("websites say server address could not be found", ["NET-008"]),
    ("connection keeps dropping every hour", ["NET-007"]),
    ("speeds are much lower than my plan", ["NET-005", "NET-006"]),
    ("how many gadgets can connect to the hub", ["DEV-005"]),
    ("can I use my own router", ["DEV-003"]),
    ("reset the router to factory settings", ["DEV-007"]),
    ("am I owed money for the days without internet", ["POL-001", "POL-003"]),
    ("what does not qualify for outage compensation", ["POL-006"]),
    ("priority help for people working from home", ["POL-004", "SVC-002"]),
    ("how long until an engineer visits", ["SVC-001"]),
    ("when will the service be restored", ["SVC-003"]),
    ("MFA-429 too many attempts", ["ACC-004"]),
    ("forgot my password reset link", ["ACC-002"]),
    ("VPN says connected but the dashboard won't load", ["BIZ-002"]),
    ("access denied on the finance dashboard", ["BIZ-004"]),
    ("who approves my access request", ["BIZ-006", "BIZ-004"]),
    ("vpn keeps disconnecting on home broadband", ["BIZ-005"]),
    ("notice period to cancel my contract", ["PLN-006"]),
    ("switch to a cheaper package", ["PLN-003", "PLN-002"]),
    ("when is billing support open", ["SVC-005"]),
    ("moving to a new flat next month", ["PLN-007"]),
]


def get_scenario(scenario_id: str) -> dict:
    for s in SCENARIOS:
        if s["id"] == scenario_id:
            return s
    raise KeyError(scenario_id)


def public_view(s: dict) -> dict:
    """Scenario without gold labels (what the UI receives)."""
    return {k: v for k, v in s.items() if k != "gold"}
