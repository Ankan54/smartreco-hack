"""The dossier: versioning, and the strike-out control.

The claim worth defending is that striking a claim changes what gets retrieved
and costs nothing. Both halves are asserted here -- the second one by making
mesh.chat explode if anything touches it.
"""

import uuid

import pytest

from app import auth, dossier, intent, retrieval, vectors


@pytest.fixture
def user_id() -> int:
    return auth.create_user(f"d{uuid.uuid4().hex[:8]}@example.com", "password-1234")


@pytest.fixture
def catalog():
    """Two products in unrelated corners of the space, so a claim aimed at one
    cannot accidentally retrieve the other."""
    made = []
    for pid, title, desc, cat in [
        (610001, "Underwater Basket Weaving with Reeds",
         "Submerged reed selection, buoyancy control and loom technique for "
         "weaving baskets beneath the surface of still water.", "Design"),
        (610002, "Sourdough Fermentation Science",
         "Wild yeast starters, hydration ratios, bulk fermentation timing and "
         "crumb structure for naturally leavened bread.", "Personal Development"),
    ]:
        vectors.upsert_product({
            "id": pid, "title": title, "description": desc, "category": cat,
            "level": "Beginner", "price": 999.0, "provider": "Test",
            "skills": "", "rating": 4.5, "prereq_ids": [], "is_active": 1})
        made.append(pid)
    yield made
    for pid in made:
        vectors.delete_product(pid)


def claim(text, polarity="+", kind="interest", strength=0.9, cid=None):
    # Caller ids are ignored on save — identity is derived from meaning.
    return {"id": cid or "ignored", "polarity": polarity, "kind": kind, "text": text,
            "strength": strength, "evidence": "test", "enabled": True}


def _id_for(text, polarity="+", kind="interest"):
    return dossier._stable_id(polarity, kind, text)


def test_versions_are_append_only(user_id):
    dossier.save(user_id, [claim("basket weaving")], "First.", "reflection")
    dossier.save(user_id, [claim("basket weaving"),
                           claim("sourdough baking")], "Second.", "reflection")
    assert dossier.current(user_id)["version"] == 2
    assert len(dossier.history(user_id)) == 2, "an earlier version was overwritten"


def test_diff_reports_what_changed(user_id):
    dossier.save(user_id, [claim("basket weaving")], "a", "reflection")
    v1 = dossier.current(user_id)
    dossier.save(user_id, [claim("basket weaving"),
                           claim("sourdough baking")], "b", "reflection")
    v2 = dossier.current(user_id)
    d = dossier.diff(v2, v1)
    assert d["added"] == [_id_for("sourdough baking")]
    assert d["removed"] == [] and d["changed"] == []


def test_same_text_keeps_id_even_when_llm_label_changes(user_id):
    dossier.save(user_id, [claim("basket weaving", cid="llm-aaa")], "a", "reflection")
    v1 = dossier.current(user_id)
    dossier.save(user_id, [claim("basket weaving", cid="llm-zzz")], "b", "reflection")
    v2 = dossier.current(user_id)
    assert v1["claims"][0]["id"] == v2["claims"][0]["id"]
    assert dossier.diff(v2, v1)["added"] == []


def test_striking_a_claim_removes_its_probe_from_retrieval(user_id, catalog, monkeypatch):
    """The headline behaviour. A '+interest' claim IS a retrieval probe, so
    disabling it must change what comes back."""
    weaving, sourdough = catalog
    dossier.save(user_id, [claim("underwater basket weaving with reeds"),
                           claim("sourdough fermentation and wild yeast")],
                 "You like both.", "reflection")
    weave_id = _id_for("underwater basket weaving with reeds")
    dough_id = _id_for("sourdough fermentation and wild yeast")

    both = retrieval.retrieve(user_id, claims=dossier.enabled_claims(user_id))
    ids = {c["id"] for c in both["candidates"]}
    assert weaving in ids and sourdough in ids, "fixture products were not retrievable"

    def explode(*a, **k):
        raise AssertionError("striking a claim must not call an LLM")
    monkeypatch.setattr("app.mesh.chat", explode)

    dossier.set_enabled(user_id, weave_id, False)
    after = retrieval.retrieve(user_id, claims=dossier.enabled_claims(user_id))

    probes = {p["name"] for p in after["probes"]}
    assert f"claim:{weave_id}" not in probes, "the struck claim still fired a probe"
    assert f"claim:{dough_id}" in probes, "the surviving claim stopped firing"

    scores = {c["id"]: c["rrf"] for c in after["candidates"]}
    assert scores.get(weaving, 0) < {c["id"]: c["rrf"] for c in both["candidates"]}[weaving]


def test_striking_writes_a_user_edit_version_with_template_prose(user_id, monkeypatch):
    dossier.save(user_id, [claim("basket weaving"),
                           claim("sourdough baking")], "LLM prose.", "reflection")

    def explode(*a, **k):
        raise AssertionError("no LLM call permitted on a user edit")
    monkeypatch.setattr("app.mesh.chat", explode)

    d = dossier.set_enabled(user_id, _id_for("basket weaving"), False)
    assert d["source"] == "user_edit"
    assert d["version"] == 2
    assert "sourdough" in d["prose"], "prose should reflect the surviving claims"
    assert "basket" not in d["prose"], "struck claim still appears in the prose"


def test_a_struck_claim_can_be_put_back(user_id):
    dossier.save(user_id, [claim("basket weaving")], "a", "reflection")
    cid = _id_for("basket weaving")
    dossier.set_enabled(user_id, cid, False)
    assert dossier.enabled_claims(user_id) == []
    dossier.set_enabled(user_id, cid, True)
    assert len(dossier.enabled_claims(user_id)) == 1


def test_candidates_carry_probe_attribution(user_id, catalog):
    """'surfaced by <claim>' in the UI has to be real, not decorative."""
    dossier.save(user_id, [claim("underwater basket weaving with reeds")],
                 "a", "reflection")
    cid = _id_for("underwater basket weaving with reeds")
    r = retrieval.retrieve(user_id, claims=dossier.enabled_claims(user_id))
    hit = next(c for c in r["candidates"] if c["id"] == catalog[0])
    assert hit["because"], "no attribution recorded"
    assert any(b["probe"] == f"claim:{cid}" for b in hit["because"])


def test_claim_count_is_capped(user_id):
    """A dossier that grows without bound stops being readable, and readability
    is the entire point of the slow tier."""
    many = [claim(f"subject number {i}") for i in range(20)]
    d = dossier.save(user_id, many, "lots", "reflection")
    assert len(d["claims"]) == dossier.MAX_CLAIMS
