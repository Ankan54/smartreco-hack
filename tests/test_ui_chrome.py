"""UI chrome contracts for chart-room polish — markup only, no visual asserts."""

import uuid

from fastapi.testclient import TestClient

from app import auth, db
from app.main import app


def _signed_in_client(*, role: str = "user"):
    db.init()
    email = f"u{uuid.uuid4().hex[:8]}@example.com"
    if role == "admin":
        auth.create_user(email, "correct-horse", role="admin")
        client = TestClient(app)
        client.post("/login", data={"email": email, "password": "correct-horse"},
                    follow_redirects=False)
        r = client.get("/me", follow_redirects=False)
        if r.status_code == 303 and "/welcome" in r.headers.get("location", ""):
            client.post("/welcome", data={"picks": ["AI/ML"]}, follow_redirects=False)
        return client
    client = TestClient(app)
    client.post("/signup", data={"email": email, "password": "correct-horse"},
                follow_redirects=False)
    client.post("/welcome", data={"picks": ["AI/ML"]}, follow_redirects=False)
    return client


def test_signed_in_header_has_profile_menu_not_loose_account_links():
    with _signed_in_client() as client:
        html = client.get("/").text
        assert 'id="profile-menu"' in html
        assert 'aria-haspopup="menu"' in html
        assert 'href="/me"' in html
        assert 'href="/logout"' in html
        # account actions live in the menu, not as top-level nav text links
        assert ">For you<" not in html.split('id="profile-menu"')[0]
        assert ">Sign out<" not in html.split('id="profile-menu"')[0]
        assert 'id="open-agent-cam"' not in html


def test_admin_menu_has_traces_link():
    with _signed_in_client(role="admin") as client:
        html = client.get("/").text
        assert 'href="/admin/traces"' in html
        assert "Traces" in html
        assert 'id="open-agent-cam"' not in html
        assert 'id="horizon-open"' not in html


def test_admin_traces_page_requires_admin_and_renders():
    with _signed_in_client() as client:
        r = client.get("/admin/traces", follow_redirects=False)
        assert r.status_code in (303, 401, 403)
    with _signed_in_client(role="admin") as client:
        r = client.get("/admin/traces")
        assert r.status_code == 200
        assert "Traces" in r.text
        assert "LangSmith" in r.text
        # Share links go through our mint endpoint, not the private project URL.
        assert "/admin/traces/" in r.text or "No root traces" in r.text or "Set LANGSMITH" in r.text or "Could not load" in r.text


def test_catalog_hero_names_reckon():
    with TestClient(app) as client:
        db.init()
        html = client.get("/").text
        assert 'class="page-hero"' in html
        assert "Reckon" in html
