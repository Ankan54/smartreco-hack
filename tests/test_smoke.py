"""Phase 1 smoke: the auth + browse loop actually works end to end."""

import uuid

from fastapi.testclient import TestClient

from app import db, vectors
from app.main import app


def _fresh_product() -> int:
    """Goes through the same write path the admin UI uses. Inserting straight
    into `products` would leave the vector and text indexes behind, which is
    precisely the bug test_dualwrite exists to catch."""
    pid = 800000 + int(uuid.uuid4().int % 10000)
    vectors.upsert_product({
        "id": pid,
        "title": "Test Course on LangGraph",
        "description": ("A long enough description for the card blurb to render "
                        "sensibly across three lines of text."),
        "category": "AI/ML", "level": "Advanced", "price": 2999.0,
        "provider": "Reckon Originals", "skills": "langgraph,agents",
        "rating": 4.7, "prereq_ids": [], "is_active": 1,
    })
    return pid


def test_browse_signup_and_enroll():
    db.init()
    pid = _fresh_product()
    email = f"t{uuid.uuid4().hex[:8]}@example.com"

    with TestClient(app) as client:
        assert client.get("/healthz").json() == {"ok": True}

        # anonymous browsing works. Filter by the unique title rather than
        # expecting it on page 1 -- the catalog is 280+ courses ranked by
        # rating, so an unfiltered assertion here is order-dependent.
        assert client.get("/").status_code == 200
        r = client.get("/", params={"q": "Test Course on LangGraph"})
        assert r.status_code == 200
        assert "Test Course on LangGraph" in r.text
        assert client.get(f"/p/{pid}").status_code == 200
        assert client.get("/p/999999").status_code == 404

        # short password is rejected, not crashed on
        r = client.post("/signup", data={"email": email, "password": "short"},
                        follow_redirects=False)
        assert r.status_code == 303 and "error" in r.headers["location"]

        # signup lands on onboarding, not the dashboard
        r = client.post("/signup", data={"email": email, "password": "correct-horse"},
                        follow_redirects=False)
        assert r.headers["location"] == "/welcome"
        assert client.get("/me", follow_redirects=False).headers["location"] == "/welcome"

        # duplicate email is handled
        r = client.post("/signup", data={"email": email, "password": "correct-horse"},
                        follow_redirects=False)
        assert "already+registered" in r.headers["location"]

        # onboarding picks are recorded, then the dashboard opens
        client.post("/welcome", data={"picks": ["AI/ML", "Cloud"]}, follow_redirects=False)
        assert client.get("/me").status_code == 200

        # enroll writes exactly one event
        client.post(f"/enroll/{pid}", follow_redirects=False)
        uid = db.q1("SELECT id FROM users WHERE email=?", (email,))["id"]
        rows = db.q("SELECT type FROM events WHERE user_id=?", (uid,))
        assert sorted(r["type"] for r in rows) == ["enroll", "search", "search"]

        # sign out, then wrong password stays out
        client.get("/logout")
        assert client.get("/me", follow_redirects=False).headers["location"] == "/login"
        r = client.post("/login", data={"email": email, "password": "nope"},
                        follow_redirects=False)
        assert "error" in r.headers["location"]

        r = client.post("/login", data={"email": email, "password": "correct-horse"},
                        follow_redirects=False)
        assert r.headers["location"] == "/me"
