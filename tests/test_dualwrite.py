"""Requirement 2: products are genuinely written to BOTH the main database and
the vector database, and stay in sync as they change.

This is the check that fails if the dual-write ever silently half-works, which
is exactly the failure the brief calls out ("a vector database that is never
queried"). It makes real embedding calls -- that is the point.
"""

import uuid

import pytest

from app import db, mesh, vectors


@pytest.fixture(scope="module", autouse=True)
def _schema():
    db.init()


def _product(**over) -> dict:
    p = {
        "id": 900000 + int(uuid.uuid4().int % 10000),
        "title": "Underwater Basket Weaving with Rust",
        "description": ("An intentionally unusual course so semantic search has "
                        "something unambiguous to find. Covers submerged reed "
                        "selection, buoyancy control and memory-safe loom drivers."),
        "category": "Design",
        "level": "Advanced",
        "price": 4321.0,
        "provider": "Test Provider",
        "skills": "basket weaving, rust, buoyancy",
        "rating": 4.9,
        "prereq_ids": [],
        "is_active": 1,
    }
    p.update(over)
    return p


def _in_chroma(pid: int) -> bool:
    return bool(vectors.collection().get(ids=[str(pid)])["ids"])


def _in_fts(pid: int) -> bool:
    return db.q1("SELECT COUNT(*) n FROM products_fts WHERE rowid = ?", (pid,))["n"] > 0


def test_create_update_unpublish_delete_keeps_all_three_stores_in_sync():
    p = _product()
    pid = p["id"]
    before = vectors.sync_status()

    try:
        # --- create: lands in all three -----------------------------------
        assert vectors.upsert_product(p) is True, "should have embedded a new product"
        assert db.q1("SELECT id FROM products WHERE id = ?", (pid,))
        assert _in_chroma(pid)
        assert _in_fts(pid)

        # --- semantic search actually finds it ----------------------------
        hits = vectors.collection().query(
            query_embeddings=[mesh.embed_one("weaving baskets underwater").tolist()],
            n_results=5,
        )
        assert str(pid) in hits["ids"][0], "new product not retrievable by meaning"

        # --- unchanged text does not re-embed -----------------------------
        assert vectors.upsert_product(p) is False, "content_hash gate did not hold"

        # --- changed text does re-embed -----------------------------------
        p["description"] += " Now also covers tidal loom scheduling."
        assert vectors.upsert_product(p) is True

        # --- unpublish removes it from the index, keeps the row -----------
        p["is_active"] = 0
        vectors.upsert_product(p)
        assert db.q1("SELECT id FROM products WHERE id = ?", (pid,)), "row should survive"
        assert not _in_chroma(pid), "unpublished product must leave the vector index"
        assert not _in_fts(pid), "unpublished product must leave the text index"

        # --- republish puts it back ---------------------------------------
        p["is_active"] = 1
        vectors.upsert_product(p)
        assert _in_chroma(pid) and _in_fts(pid)

    finally:
        vectors.delete_product(pid)

    # --- delete removes it everywhere, and totals return to baseline ------
    assert not db.q1("SELECT id FROM products WHERE id = ?", (pid,))
    assert not _in_chroma(pid)
    assert not _in_fts(pid)

    after = vectors.sync_status()
    assert after["in_sync"], f"stores disagree after CRUD cycle: {after}"
    assert after == before, f"counts drifted: {before} -> {after}"


def test_embed_cache_prevents_repeat_calls():
    text = f"cache probe {uuid.uuid4()}"
    v1 = mesh.embed_one(text)
    rows = db.q1("SELECT COUNT(*) n FROM embed_cache")["n"]
    v2 = mesh.embed_one(text)          # must be served from SQLite, not the API
    assert db.q1("SELECT COUNT(*) n FROM embed_cache")["n"] == rows
    assert (v1 == v2).all()
