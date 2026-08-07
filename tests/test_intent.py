"""The intent engine's maths and the drift gate.

These are pure-numpy assertions with no network: if the decay, the drift metric
or the trigger rules break, the whole "smart about when we call the AI" claim
breaks with them, silently and invisibly.
"""

import uuid
from datetime import datetime, timedelta, timezone

import numpy as np
import pytest

from app import auth, config, db, intent


@pytest.fixture
def user_id() -> int:
    return auth.create_user(f"i{uuid.uuid4().hex[:8]}@example.com", "password-1234")


def vec(*parts) -> np.ndarray:
    v = np.zeros(config.EMBED_DIM, dtype=np.float32)
    for i, x in parts:
        v[i] = x
    return v


def set_state(user_id, *, now=None, ref=None, n=0, last_reco=None, last_event=None):
    intent.get_state(user_id)
    with db.tx() as c:
        c.execute(
            "UPDATE user_state SET intent_vec=?, intent_vec_at_last_reco=?,"
            " events_since_reco=?, last_reco_at=?, last_event_at=? WHERE user_id=?",
            (db.to_blob(now) if now is not None else None,
             db.to_blob(ref) if ref is not None else None,
             n, last_reco, last_event, user_id),
        )
    return intent.get_state(user_id)


# --- decay -----------------------------------------------------------------

def test_decay_halves_at_the_half_life():
    v = vec((0, 1.0))
    assert intent.decay(v, config.HALF_LIFE_MIN)[0] == pytest.approx(0.5, abs=1e-6)
    assert intent.decay(v, 2 * config.HALF_LIFE_MIN)[0] == pytest.approx(0.25, abs=1e-6)
    assert intent.decay(v, 0)[0] == pytest.approx(1.0)


def test_decay_preserves_direction():
    """Decay must scale, never rotate -- otherwise it would invent interest drift
    out of nothing but the passage of time."""
    v = vec((0, 3.0), (7, 4.0))
    assert intent.cosine(v, intent.decay(v, 137.0)) == pytest.approx(1.0, abs=1e-6)


# --- drift -----------------------------------------------------------------

def test_drift_is_zero_for_the_same_bearing_and_one_for_orthogonal():
    a, b = vec((0, 1.0)), vec((1, 1.0))
    assert intent.drift_of({"intent_vec": a, "intent_vec_at_last_reco": a}) == pytest.approx(0.0, abs=1e-6)
    assert intent.drift_of({"intent_vec": b, "intent_vec_at_last_reco": a}) == pytest.approx(1.0, abs=1e-6)


def test_drift_ignores_magnitude():
    """Engagement getting stronger is not a change of course."""
    a = vec((0, 1.0))
    assert intent.drift_of({"intent_vec": a * 9.0, "intent_vec_at_last_reco": a}) == pytest.approx(0.0, abs=1e-6)


# --- the gate --------------------------------------------------------------

def test_first_ever_recommendation_is_cold_start(user_id):
    s = set_state(user_id, now=vec((0, 1.0)), ref=None)
    fire, reason, _ = intent.should_fire(s)
    assert fire and reason == "cold_start"


def test_no_signal_never_fires(user_id):
    fire, reason, _ = intent.should_fire(set_state(user_id))
    assert not fire and reason == "no_signal"


def test_small_drift_does_not_wake_the_agent(user_id):
    """The whole point: browsing more of the same must cost zero LLM calls."""
    a = vec((0, 1.0))
    nudged = a + 0.02 * vec((1, 1.0))
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    s = set_state(user_id, now=nudged, ref=a, n=9, last_reco=old)
    fire, reason, d = intent.should_fire(s)
    assert not fire and reason == "steady"
    assert d < config.DRIFT_THRESHOLD


def test_large_drift_wakes_the_agent(user_id):
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    s = set_state(user_id, now=vec((1, 1.0)), ref=vec((0, 1.0)),
                  n=config.MIN_EVENTS_FOR_DRIFT, last_reco=old)
    fire, reason, d = intent.should_fire(s)
    assert fire and reason == "drift"
    assert d >= config.DRIFT_THRESHOLD


def test_drift_needs_enough_events_behind_it(user_id):
    """One stray click should not be able to swing the vector and fire."""
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    s = set_state(user_id, now=vec((1, 1.0)), ref=vec((0, 1.0)),
                  n=config.MIN_EVENTS_FOR_DRIFT - 1, last_reco=old)
    fire, _, _ = intent.should_fire(s)
    assert not fire


def test_rate_limit_overrides_every_other_trigger(user_id):
    """A burst of activity must never become a burst of LLM calls."""
    just_now = datetime.now(timezone.utc).isoformat()
    s = set_state(user_id, now=vec((1, 1.0)), ref=vec((0, 1.0)),
                  n=config.VOLUME_TRIGGER * 5, last_reco=just_now)
    fire, reason, _ = intent.should_fire(s)
    assert not fire and reason == "rate_limited"


def test_volume_fires_even_without_drift(user_id):
    a = vec((0, 1.0))
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    s = set_state(user_id, now=a, ref=a, n=config.VOLUME_TRIGGER, last_reco=old)
    fire, reason, _ = intent.should_fire(s)
    assert fire and reason == "volume"


def test_stale_fires_after_a_long_quiet_gap(user_id):
    a = vec((0, 1.0))
    ancient = (datetime.now(timezone.utc)
               - timedelta(hours=config.STALE_HOURS + 1)).isoformat()
    s = set_state(user_id, now=a, ref=a, n=1, last_reco=ancient)
    fire, reason, _ = intent.should_fire(s)
    assert fire and reason == "stale"


def test_mark_reasoned_resets_drift_to_zero(user_id):
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()
    s = set_state(user_id, now=vec((1, 1.0)), ref=vec((0, 1.0)),
                  n=7, last_reco=old)
    assert intent.drift_of(s) > config.DRIFT_THRESHOLD

    intent.mark_reasoned(user_id)
    s2 = intent.get_state(user_id)
    assert intent.drift_of(s2) == pytest.approx(0.0, abs=1e-6)
    assert s2["events_since_reco"] == 0


# --- events feed the vector ------------------------------------------------

def test_bounce_pushes_the_vector_away(user_id, monkeypatch):
    """A negative weight is what stops this being a popularity vector."""
    liked, rejected = vec((0, 1.0)), vec((1, 1.0))
    monkeypatch.setattr(intent, "signal_vector",
                        lambda ev: liked if ev["product_id"] == 1 else rejected)

    set_state(user_id, now=None)
    intent.apply_events(user_id, [{"type": "view", "product_id": 1}])
    after_view = intent.get_state(user_id)["intent_vec"]
    sim_before = intent.cosine(after_view, rejected)

    intent.apply_events(user_id, [{"type": "bounce", "product_id": 2}])
    after_bounce = intent.get_state(user_id)["intent_vec"]
    assert intent.cosine(after_bounce, rejected) < sim_before


def test_unknown_event_types_are_ignored(user_id, monkeypatch):
    monkeypatch.setattr(intent, "signal_vector", lambda ev: vec((0, 1.0)))
    set_state(user_id, now=None)
    intent.apply_events(user_id, [{"type": "telepathy", "product_id": 1}])
    assert intent.get_state(user_id)["events_since_reco"] == 0


def test_horizon_tilt_is_bounded_and_tracks_drift(user_id):
    a = vec((0, 1.0))
    old = (datetime.now(timezone.utc) - timedelta(hours=1)).isoformat()

    set_state(user_id, now=a, ref=a, n=1, last_reco=old)
    level = intent.horizon_state(user_id)
    assert level["tilt_deg"] == pytest.approx(0.0) and level["state"] == "level"

    set_state(user_id, now=vec((1, 1.0)), ref=a, n=9, last_reco=old)
    crossed = intent.horizon_state(user_id)
    assert crossed["ratio"] == 1.0            # clamped, never past the tick
    assert 0 < crossed["tilt_deg"] <= 4.0
    assert crossed["state"] == "crossed"


# --- the thundering herd ---------------------------------------------------

def test_only_one_concurrent_claim_wins(user_id):
    """The bug this pins: should_fire() is a read, and a burst of events spawns
    a background task each. All of them passed the gate before any of them
    wrote, so nine events produced EIGHT concurrent cold-start agent runs --
    roughly forty LLM calls where there should have been five."""
    set_state(user_id, now=vec((0, 1.0)), ref=None)

    wins = [intent.try_claim(user_id) for _ in range(10)]
    assert sum(wins) == 1, f"expected exactly one winner, got {sum(wins)}"
    assert wins[0] is True, "the first caller should win"


def test_claim_is_released_after_the_cooldown(user_id):
    old = (datetime.now(timezone.utc)
           - timedelta(seconds=config.MIN_RECO_INTERVAL_SEC + 5)).isoformat()
    set_state(user_id, now=vec((0, 1.0)), ref=None, last_reco=old)
    assert intent.try_claim(user_id) is True
    assert intent.try_claim(user_id) is False, "cooldown not re-armed after a win"


def test_claim_works_under_real_threads(user_id):
    """Same assertion, but through actual concurrency rather than a loop --
    this is the shape the bug arrived in."""
    import threading

    set_state(user_id, now=vec((0, 1.0)), ref=None)
    results, lock = [], threading.Lock()

    def go():
        won = intent.try_claim(user_id)
        with lock:
            results.append(won)

    threads = [threading.Thread(target=go) for _ in range(8)]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    assert sum(results) == 1, f"race not prevented: {sum(results)} winners"


def test_concurrent_batches_do_not_lose_signal(user_id, monkeypatch):
    """Updating the vector is a read-modify-write in Python. Without the per-user
    lock, parallel batches overwrite each other: measured, ten concurrent
    single-event batches produced a vector of norm 1.0 instead of ~8, silently
    discarding nine events. The counter hid it, because `x = x + ?` is atomic
    inside SQLite while the vector is not."""
    import threading

    directions = [vec((i, 1.0)) for i in range(10)]
    monkeypatch.setattr(intent, "signal_vector",
                        lambda ev: directions[ev["product_id"]])
    set_state(user_id, now=None)

    threads = [
        threading.Thread(target=lambda i=i: intent.apply_events(
            user_id, [{"type": "view", "product_id": i}]))
        for i in range(10)
    ]
    for t in threads:
        t.start()
    for t in threads:
        t.join()

    s = intent.get_state(user_id)
    assert s["events_since_reco"] == 10
    # Ten orthogonal unit vectors summed give norm sqrt(10) ~= 3.16.
    # The race collapsed this to 1.0.
    assert float(np.linalg.norm(s["intent_vec"])) > 3.0
