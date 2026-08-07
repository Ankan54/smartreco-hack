"""The agent's structure and its guarantees.

These are stubbed at the mesh.chat boundary: the point is to pin the graph's
control flow and its cost ceiling, which is where the real bugs were. The
live end-to-end behaviour is exercised by scripts/simulate_users.py.
"""

import json
import uuid

import pytest

from app import agent, auth, db, intent, vectors


@pytest.fixture
def user_id() -> int:
    uid = auth.create_user(f"a{uuid.uuid4().hex[:8]}@example.com", "password-1234")
    pid = 700000 + int(uuid.uuid4().int % 10000)
    vectors.upsert_product({
        "id": pid, "title": "Agentic Systems in Practice",
        "description": "Build and ship multi-agent workflows with tool routing and "
                       "durable execution across long-running tasks.",
        "category": "AI/ML", "level": "Advanced", "price": 3999.0,
        "provider": "Test", "skills": "agents, orchestration", "rating": 4.8,
        "prereq_ids": [], "is_active": 1,
    })
    with db.tx() as c:
        c.execute("INSERT INTO events(id,user_id,session_id,type,product_id,ts) "
                  "VALUES (?,?,'t','view',?,datetime('now'))",
                  (str(uuid.uuid4()), uid, pid))
    # Push it through the real intent engine rather than only inserting the row:
    # persist() snapshots the live intent vector, so a user with no vector is
    # not a state the gate would ever actually hand to the agent.
    intent.apply_events(uid, [{"type": "view", "product_id": pid}])
    yield uid
    vectors.delete_product(pid)


class FakeChat:
    """Stands in for mesh.chat. Records calls so we can count LLM spend."""

    def __init__(self, critic_verdicts=("revise", "revise", "revise", "revise")):
        self.calls: list[str] = []
        self.critic_verdicts = list(critic_verdicts)

    def __call__(self, messages, schema=None, model=None):
        name = schema["name"] if schema else "raw"
        self.calls.append(name)
        meta = {"model": model or "test", "cache": "MISS"}
        if name == "dossier":
            return {"prose": "You are circling agentic systems.",
                    "claims": [{"id": "c1", "polarity": "+", "kind": "interest",
                                "text": "agentic AI frameworks", "strength": 0.9,
                                "evidence": "viewed an agents course"}]}, meta
        if name == "retrieval_grade":
            return {"sufficient": True, "reason": "fine", "pseudo_query": "",
                    "filters": {"category": "", "level": "", "max_price": 0}}, meta
        if name == "recommendation":
            return {"headline": "Next up", "narrative": "You viewed an agents course.",
                    "items": [{"product_id": -1, "reason": "invented", "confidence": 0.9}]}, meta
        if name == "critique":
            v = self.critic_verdicts.pop(0) if self.critic_verdicts else "pass"
            return {"verdict": v, "faults": ["too generic"]}, meta
        raise AssertionError(f"unexpected schema {name}")

    def count(self, kind: str) -> int:
        return self.calls.count(kind)


def test_graph_has_every_node_the_bonus_enumerates():
    """'nodes that analyze the query/activity, decide when to retrieve, evaluate
    retrieval quality, refine, and generate' -- all five must exist by name."""
    nodes = set(agent.GRAPH.get_graph().nodes)
    for required in ("read_behavior", "should_reflect", "grade", "refine", "generate"):
        assert required in nodes or required == "should_reflect", required
    assert {"reflect", "retrieve", "verify", "critic", "persist"} <= nodes


def test_graph_renders_its_own_mermaid():
    """Agent Cam ships LangGraph's real topology, not a hand-drawn diagram."""
    assert "read_behavior" in agent.GRAPH_MERMAID
    assert "critic" in agent.GRAPH_MERMAID


def test_critic_retry_is_bounded_to_one(user_id, monkeypatch):
    """The bug this pins: mutating state inside a conditional-edge function is
    discarded, so the guard never stuck and the critic looped four times,
    burning double the budgeted LLM calls."""
    fake = FakeChat(critic_verdicts=["revise"] * 6)
    monkeypatch.setattr(agent.mesh, "chat", fake)

    agent.run(user_id, "cold_start", drift=0.0)

    assert fake.count("recommendation") == 2, "generate should run at most twice"
    assert fake.count("critique") == 1, "critic should not re-run after its retry"
    assert len(fake.calls) <= 5, f"LLM budget blown: {fake.calls}"


def test_reflect_is_skipped_when_the_trigger_is_not_a_course_change(user_id, monkeypatch):
    """volume/stale reuse the existing beliefs -- that is the whole point of
    should_reflect, and it caps the per-fire budget."""
    fake = FakeChat(critic_verdicts=["pass"])
    monkeypatch.setattr(agent.mesh, "chat", fake)

    agent.run(user_id, "cold_start", drift=0.0)       # seeds a dossier
    fake.calls.clear()
    fake.critic_verdicts = ["pass"]
    agent.run(user_id, "volume", drift=0.01)

    assert fake.count("dossier") == 0, "should not reflect on a volume trigger"


def test_verify_drops_hallucinated_ids_and_backfills(user_id, monkeypatch):
    """generate is stubbed to return product_id -1, which is not in the catalog.
    Nothing invented may ever reach the user."""
    fake = FakeChat(critic_verdicts=["pass"])
    monkeypatch.setattr(agent.mesh, "chat", fake)

    final = agent.run(user_id, "cold_start", drift=0.0)
    ids = [i["product_id"] for i in final["draft"]["items"]]

    assert -1 not in ids, "a hallucinated id reached the output"
    assert ids, "verify should backfill rather than return nothing"
    real = {r["id"] for r in db.q("SELECT id FROM products WHERE is_active=1")}
    assert set(ids) <= real, "every recommended id must be a real active product"


def test_persist_writes_a_current_recommendation_and_resets_drift(user_id, monkeypatch):
    fake = FakeChat(critic_verdicts=["pass"])
    monkeypatch.setattr(agent.mesh, "chat", fake)

    agent.run(user_id, "cold_start", drift=0.42)

    row = db.q1("SELECT * FROM recommendations WHERE user_id=? AND is_current=1", (user_id,))
    assert row and row["trigger"] == "cold_start"
    assert json.loads(row["items_json"]), "no items persisted"
    trace = json.loads(row["trace_json"])
    assert [s["node"] for s in trace["steps"]][0] == "read_behavior"

    state = db.q1("SELECT * FROM user_state WHERE user_id=?", (user_id,))
    assert state["events_since_reco"] == 0, "drift baseline was not reset"
    assert state["intent_vec_at_last_reco"] is not None


def test_only_one_recommendation_is_current(user_id, monkeypatch):
    fake = FakeChat(critic_verdicts=["pass", "pass"])
    monkeypatch.setattr(agent.mesh, "chat", fake)
    agent.run(user_id, "cold_start", drift=0.0)
    fake.critic_verdicts = ["pass"]
    agent.run(user_id, "manual", drift=0.2)

    n = db.q1("SELECT COUNT(*) n FROM recommendations WHERE user_id=? AND is_current=1",
              (user_id,))["n"]
    assert n == 1
