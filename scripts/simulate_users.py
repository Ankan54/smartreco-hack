"""Replay demo personas through the REAL endpoints, end to end.

This is the honest end-to-end check: it posts to /api/events exactly as the
browser does, so the whole chain runs -- dedup, the intent vector, the drift
gate, and the agent when the gate opens. Nothing is stubbed.

It doubles as the calibration harness for DRIFT_THRESHOLD: --report prints the
drift trace for each persona so you can see whether a genuine change of course
actually crosses the line, rather than guessing at the constant.

    uv run python scripts/simulate_users.py                 # against a running server
    uv run python scripts/simulate_users.py --report        # drift trace, no agent
    uv run python scripts/simulate_users.py --persona switcher
"""

import argparse
import sys
import time
import uuid
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, db, dossier, intent  # noqa: E402

BASE = "http://127.0.0.1:8000"
PASSWORD = "simulated-persona"

# Each persona is a story about a change of course, because that is what the
# drift gate exists to detect.
PERSONAS = {
    "switcher": {
        "blurb": "an ML engineer who pivots to finance halfway through",
        "seed": ["AI/ML"],
        "acts": [
            ("AI/ML", 4, "deep learning model training"),
            ("Finance", 4, "financial modelling and valuation"),
        ],
    },
    "deepener": {
        "blurb": "stays in one field and goes deeper -- must NOT keep waking the agent",
        "seed": ["Data Engineering"],
        "acts": [
            ("Data Engineering", 4, "batch data pipelines"),
            ("Data Engineering", 4, "streaming data pipelines"),
        ],
    },
    "drifter": {
        "blurb": "wanders across three unrelated fields",
        "seed": ["Design"],
        "acts": [
            ("Design", 3, "user experience research"),
            ("Cybersecurity", 3, "network security fundamentals"),
            ("Leadership", 3, "leading engineering teams"),
        ],
    },
}


def products_in(category: str, n: int) -> list[int]:
    rows = db.q(
        "SELECT id FROM products WHERE category = ? AND is_active = 1 "
        "ORDER BY rating DESC NULLS LAST LIMIT ?", (category, n))
    return [r["id"] for r in rows]


def event(kind: str, **fields) -> dict:
    return {"id": str(uuid.uuid4()), "type": kind, "session_id": "sim",
            "ts": time.strftime("%Y-%m-%dT%H:%M:%S"), **fields}


class Persona:
    def __init__(self, name: str, spec: dict, client: httpx.Client):
        self.name = name
        self.spec = spec
        self.client = client
        self.email = f"sim-{name}-{uuid.uuid4().hex[:6]}@example.com"
        self.user_id: int | None = None
        self.drift_trace: list[tuple[str, float, str]] = []
        self._counted = 0

    def sign_up(self) -> None:
        self.client.post(f"{BASE}/signup",
                         data={"email": self.email, "password": PASSWORD})
        row = db.q1("SELECT id FROM users WHERE email = ?", (self.email,))
        self.user_id = row["id"]
        self.client.post(f"{BASE}/welcome", data={"picks": self.spec["seed"]})
        # cold_start fires the agent in the background; wait for the baseline so
        # the first act measures drift against something real.
        deadline = time.time() + 90
        while time.time() < deadline:
            if intent.get_state(self.user_id)["intent_vec_at_last_reco"] is not None:
                break
            time.sleep(0.5)

    def send(self, events: list[dict]) -> int:
        """Returns the events_since_reco we expect once the server catches up."""
        r = self.client.post(f"{BASE}/api/events", json={"events": events})
        r.raise_for_status()
        return r.json().get("accepted", 0)

    def settle(self, expected_new: int, timeout: float = 30.0) -> None:
        """/api/events answers 202 BEFORE the intent update runs -- that is the
        whole point of the non-blocking design. So a simulator that reads state
        straight after posting sees the previous state and reports drift 0.0000
        for everything. Wait for the background write instead of guessing."""
        target = self._counted + expected_new
        was = intent.get_state(self.user_id)["last_reco_at"]
        deadline = time.time() + timeout
        while time.time() < deadline:
            s = intent.get_state(self.user_id)
            # The counter resets to 0 when the agent fires, so a CHANGED
            # last_reco_at is also proof the batch landed. Comparing against a
            # snapshot matters: any non-null value would otherwise satisfy this
            # on the first poll and defeat the wait entirely.
            if s["events_since_reco"] >= target or s["last_reco_at"] != was:
                break
            time.sleep(0.4)
        self._counted = intent.get_state(self.user_id)["events_since_reco"]

    def observe(self, label: str) -> None:
        s = intent.get_state(self.user_id)
        fire, reason, d = intent.should_fire(s)
        self.drift_trace.append((label, d, reason))

    def browse(self, category: str, n: int, query: str) -> None:
        """One 'act': a search, then a few product views with real dwell."""
        sent = self.send([event("search", query=query)])
        for pid in products_in(category, n):
            sent += self.send([event("view", product_id=pid)])
            sent += self.send([event("dwell", product_id=pid, value=45.0)])
        self.settle(sent)
        self.observe(category)


def run(names: list[str], report_only: bool) -> None:
    with httpx.Client(timeout=120, follow_redirects=True) as client:
        for name in names:
            spec = PERSONAS[name]
            print(f"\n=== {name}: {spec['blurb']}")
            p = Persona(name, spec, client)
            p.sign_up()
            print(f"    signed up {p.email} (user {p.user_id}), seeded {spec['seed']}")

            for i, (category, n, query) in enumerate(spec["acts"]):
                if i and not report_only:
                    # MIN_RECO_INTERVAL_SEC is a real guard, not an inconvenience:
                    # respect it so the trace shows a genuine drift fire.
                    wait = config.MIN_RECO_INTERVAL_SEC + 3
                    print(f"    (waiting {wait}s for the rate limit to clear)")
                    time.sleep(wait)
                p.browse(category, n, query)

            print(f"    {'act':<20} {'drift':>7}  {'threshold':>9}  gate")
            for label, d, reason in p.drift_trace:
                crossed = "FIRE" if reason in ("drift", "cold_start", "volume",
                                               "stale") else reason
                print(f"    {label:<20} {d:7.4f}  {config.DRIFT_THRESHOLD:9.2f}  {crossed}")

            if report_only:
                continue

            # Give the background agent a moment, then show what it produced.
            time.sleep(8)
            reco = db.q1("SELECT * FROM recommendations WHERE user_id = ? "
                         "AND is_current = 1", (p.user_id,))
            d = dossier.current(p.user_id)
            if reco:
                print(f"\n    -> \"{reco['headline']}\"  (woke on: {reco['trigger']})")
                print(f"       {reco['narrative'][:180]}")
            else:
                print("    -> no recommendation yet")
            if d:
                print(f"       dossier v{d['version']}: "
                      + " | ".join(f"{c['polarity']}{c['text']}" for c in d["claims"][:4]))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--persona", choices=sorted(PERSONAS), action="append")
    ap.add_argument("--report", action="store_true",
                    help="drift trace only; skip waiting for the agent")
    ap.add_argument("--base", default=BASE)
    args = ap.parse_args()

    globals()["BASE"] = args.base
    try:
        httpx.get(f"{BASE}/healthz", timeout=5).raise_for_status()
    except Exception:
        sys.exit(f"no server at {BASE} -- start it with "
                 "`uv run uvicorn app.main:app --port 8000`")

    run(args.persona or sorted(PERSONAS), args.report)


if __name__ == "__main__":
    main()
