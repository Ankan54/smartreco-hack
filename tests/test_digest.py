"""Scheduled proactive delivery.

The assertions that matter are the ones about NOT sending. Signup addresses are
unverified, so a scheduler that mails everyone who registered is sending
unsolicited email to strangers -- a bug with a real-world victim, not just a
failing test.
"""

import json
import uuid

import pytest

from app import auth, db, notify, scheduler, vectors


@pytest.fixture
def user_id() -> int:
    return auth.create_user(f"g{uuid.uuid4().hex[:8]}@example.com", "password-1234")


@pytest.fixture
def product_id():
    pid = 500000 + int(uuid.uuid4().int % 10000)
    vectors.upsert_product({
        "id": pid, "title": "Digest Fixture Course",
        "description": "A course that exists so the digest has something to link to.",
        "category": "AI/ML", "level": "Beginner", "price": 1499.0,
        "provider": "Test", "skills": "", "rating": 4.4,
        "prereq_ids": [], "is_active": 1})
    yield pid
    vectors.delete_product(pid)


def give_reco(user_id: int, product_id: int) -> int:
    with db.tx() as c:
        cur = c.execute(
            "INSERT INTO recommendations(user_id,headline,narrative,items_json,trigger)"
            " VALUES (?,?,?,?,?)",
            (user_id, "Where to go next", "You kept circling agentic AI.",
             json.dumps([{"product_id": product_id, "reason": "Follows what you read",
                          "confidence": 0.8}]), "scheduled"))
        return cur.lastrowid


class Outbox:
    def __init__(self, ok=True):
        self.sent = []
        self.ok = ok

    def __call__(self, to, subject, html_body, text_body):
        self.sent.append({"to": to, "subject": subject,
                          "html": html_body, "text": text_body})
        return self.ok


def test_never_emails_a_user_who_did_not_opt_in(user_id, product_id, monkeypatch):
    """The important one. Default is off, and off means silent."""
    give_reco(user_id, product_id)
    box = Outbox()
    monkeypatch.setattr(notify, "send", box)
    monkeypatch.setattr(notify, "enabled", lambda: True)

    assert notify.deliver_digest(user_id) is False
    assert box.sent == [], "emailed a user who never opted in"


def test_emails_an_opted_in_user(user_id, product_id, monkeypatch):
    give_reco(user_id, product_id)
    with db.tx() as c:
        c.execute("UPDATE users SET digest_opt_in = 1 WHERE id = ?", (user_id,))
    box = Outbox()
    monkeypatch.setattr(notify, "send", box)
    monkeypatch.setattr(notify, "enabled", lambda: True)

    assert notify.deliver_digest(user_id) is True
    assert len(box.sent) == 1
    msg = box.sent[0]
    email = db.q1("SELECT email FROM users WHERE id=?", (user_id,))["email"]
    assert msg["to"] == email
    assert "Digest Fixture Course" in msg["html"]
    assert "Digest Fixture Course" in msg["text"], "no plain-text alternative"
    assert "You kept circling agentic AI." in msg["text"]
    assert "Turn it off" in msg["html"], "no way to unsubscribe"


def test_the_same_recommendation_is_not_mailed_twice(user_id, product_id, monkeypatch):
    give_reco(user_id, product_id)
    with db.tx() as c:
        c.execute("UPDATE users SET digest_opt_in = 1 WHERE id = ?", (user_id,))
    box = Outbox()
    monkeypatch.setattr(notify, "send", box)
    monkeypatch.setattr(notify, "enabled", lambda: True)

    assert notify.deliver_digest(user_id) is True
    assert notify.deliver_digest(user_id) is False, "sent a duplicate digest"
    assert len(box.sent) == 1


def test_delivery_is_a_no_op_when_smtp_is_unconfigured(user_id, product_id):
    """A clean clone with only a Mesh key must still boot and still pass checks."""
    give_reco(user_id, product_id)
    with db.tx() as c:
        c.execute("UPDATE users SET digest_opt_in = 1 WHERE id = ?", (user_id,))
    assert notify.send("x@example.com", "s", "<p>h</p>", "t") is False
    assert notify.deliver_digest(user_id) is False


def test_html_escapes_untrusted_content(user_id, product_id, monkeypatch):
    """The narrative is LLM-written from the user's own search queries, so it
    crosses a trust boundary before it reaches an inbox."""
    with db.tx() as c:
        c.execute("UPDATE users SET digest_opt_in = 1 WHERE id = ?", (user_id,))
        c.execute("INSERT INTO recommendations(user_id,headline,narrative,items_json,"
                  "trigger) VALUES (?,?,?,?,?)",
                  (user_id, "<script>alert(1)</script>",
                   "you searched <img src=x onerror=alert(1)>",
                   json.dumps([{"product_id": product_id, "reason": "<b>bold</b>",
                                "confidence": 0.5}]), "scheduled"))
    box = Outbox()
    monkeypatch.setattr(notify, "send", box)
    monkeypatch.setattr(notify, "enabled", lambda: True)

    notify.deliver_digest(user_id)
    html_body = box.sent[0]["html"]

    # What matters is that no injected TAG survives. The literal text
    # "onerror=" is harmless once its angle brackets are entities -- asserting
    # on the substring alone would be testing the wrong thing.
    assert "<script>" not in html_body
    assert "<img" not in html_body
    assert "<b>bold</b>" not in html_body, "a reason injected raw markup"
    assert "&lt;script&gt;" in html_body
    assert "&lt;img src=x onerror=alert(1)&gt;" in html_body


def test_scheduler_registers_both_jobs(monkeypatch):
    monkeypatch.setattr("app.config.ENABLE_SCHEDULER", True)
    s = scheduler.start()
    try:
        assert s is not None
        ids = {j.id for j in s.get_jobs()}
        assert ids == {"drift_sweep", "daily_digest"}
    finally:
        scheduler.shutdown()


def test_scheduler_can_be_switched_off(monkeypatch):
    monkeypatch.setattr("app.config.ENABLE_SCHEDULER", False)
    assert scheduler.start() is None


def test_digest_job_sends_nothing_without_smtp(monkeypatch):
    monkeypatch.setattr(notify, "enabled", lambda: False)
    assert scheduler.digest() == 0
