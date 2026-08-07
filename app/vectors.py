"""The product store: SQLite row + FTS5 index + Chroma vector, written together.

Requirement 2 of the brief is that a product added by an admin lands in BOTH the
main database and the vector database, and that they stay in sync as products
change. Every write in this app goes through upsert_product()/delete_product()
so there is exactly one code path that can get that wrong.

Invariant (see PLAN.md 7.1): **Chroma contains exactly the active products.**
That makes the admin sync badge a real check -- COUNT(*) WHERE is_active=1 must
equal collection.count(), so a mismatch is a bug rather than expected drift.
It also means retrieval needs no is_active filter: inactive products aren't
in the index at all.
"""

import hashlib
import json
import logging
from functools import lru_cache

import chromadb
import numpy as np

from app import config, db, mesh

log = logging.getLogger(__name__)

_client = None
_collection = None


def collection():
    global _client, _collection
    if _collection is None:
        config.CHROMA_PATH.mkdir(parents=True, exist_ok=True)
        _client = chromadb.PersistentClient(path=str(config.CHROMA_PATH))
        _collection = _client.get_or_create_collection(
            name=config.COLLECTION,
            # Cosine, not the L2 default: our intent vectors are compared by
            # direction, and magnitude here means engagement, not similarity.
            metadata={"hnsw:space": "cosine"},
        )
    return _collection


def embed_text(p: dict) -> str:
    """What actually gets embedded. Claim text in the dossier is written to look
    like this on purpose -- a probe has to live in the same space as the docs."""
    return (
        f"{p['title']}\n"
        f"{p['category']} - {p.get('level') or 'All levels'}\n"
        f"{p.get('skills') or ''}\n"
        f"{p['description']}"
    )


def content_hash(p: dict) -> str:
    return hashlib.sha256(embed_text(p).encode()).hexdigest()


def _meta(p: dict) -> dict:
    # Chroma metadata must be scalars. These are the fields the agent is allowed
    # to filter on via `where`.
    return {
        "product_id": int(p["id"]),
        "category": p["category"],
        "level": p.get("level") or "",
        "price": float(p.get("price") or 0),
        "rating": float(p.get("rating") or 0),
    }


def upsert_product(p: dict, *, skip_unchanged: bool = True) -> bool:
    """Write one product to all three stores. Returns True if it was embedded.

    `p` needs id, title, description, category; the rest are optional.
    Re-embedding text that has not changed is the single most wasteful thing
    this app could do, so content_hash gates it.
    """
    new_hash = content_hash(p)
    row = db.q1("SELECT content_hash, is_active FROM products WHERE id = ?", (p["id"],))
    active = int(p.get("is_active", 1))
    unchanged = bool(row) and row["content_hash"] == new_hash and row["is_active"] == active

    with db.tx() as c:
        c.execute(
            """INSERT INTO products
                 (id,title,description,category,level,price,provider,skills,rating,
                  prereq_ids,content_hash,is_active,updated_at)
               VALUES (?,?,?,?,?,?,?,?,?,?,?,?,datetime('now'))
               ON CONFLICT(id) DO UPDATE SET
                 title=excluded.title, description=excluded.description,
                 category=excluded.category, level=excluded.level, price=excluded.price,
                 provider=excluded.provider, skills=excluded.skills, rating=excluded.rating,
                 prereq_ids=excluded.prereq_ids, content_hash=excluded.content_hash,
                 is_active=excluded.is_active, updated_at=datetime('now')""",
            (p["id"], p["title"], p["description"], p["category"], p.get("level"),
             float(p.get("price") or 0), p.get("provider"), p.get("skills"),
             p.get("rating"), json.dumps(p.get("prereq_ids") or []), new_hash, active),
        )
        # FTS5 has no upsert; delete-then-insert keeps rowid aligned with id.
        c.execute("DELETE FROM products_fts WHERE rowid = ?", (p["id"],))
        if active:
            c.execute(
                "INSERT INTO products_fts(rowid,title,skills,description) VALUES (?,?,?,?)",
                (p["id"], p["title"], p.get("skills") or "", p["description"]),
            )

    if not active:
        collection().delete(ids=[str(p["id"])])   # unpublish leaves the index
        return False

    if unchanged and skip_unchanged:
        return False

    _cached_product_vector.cache_clear()
    vec = mesh.embed_one(embed_text(p))
    collection().upsert(
        ids=[str(p["id"])],
        embeddings=[vec.tolist()],
        metadatas=[_meta(p)],
        documents=[p["title"]],
    )
    return True


def upsert_many(products: list[dict], *, skip_unchanged: bool = True) -> int:
    """Batched version. One embeddings call per 96 products beats one per product
    by roughly the round-trip latency times the catalog size."""
    todo = []
    for p in products:
        p = dict(p)
        p["_hash"] = content_hash(p)
        row = db.q1("SELECT content_hash FROM products WHERE id = ?", (p["id"],))
        p["_unchanged"] = bool(row) and row["content_hash"] == p["_hash"]
        todo.append(p)

    with db.tx() as c:
        for p in todo:
            c.execute(
                """INSERT INTO products
                     (id,title,description,category,level,price,provider,skills,rating,
                      prereq_ids,content_hash,is_active,updated_at)
                   VALUES (?,?,?,?,?,?,?,?,?,?,?,1,datetime('now'))
                   ON CONFLICT(id) DO UPDATE SET
                     title=excluded.title, description=excluded.description,
                     category=excluded.category, level=excluded.level, price=excluded.price,
                     provider=excluded.provider, skills=excluded.skills, rating=excluded.rating,
                     prereq_ids=excluded.prereq_ids, content_hash=excluded.content_hash,
                     is_active=1, updated_at=datetime('now')""",
                (p["id"], p["title"], p["description"], p["category"], p.get("level"),
                 float(p.get("price") or 0), p.get("provider"), p.get("skills"),
                 p.get("rating"), json.dumps(p.get("prereq_ids") or []), p["_hash"]),
            )
            c.execute("DELETE FROM products_fts WHERE rowid = ?", (p["id"],))
            c.execute(
                "INSERT INTO products_fts(rowid,title,skills,description) VALUES (?,?,?,?)",
                (p["id"], p["title"], p.get("skills") or "", p["description"]),
            )

    pending = [p for p in todo if not (p["_unchanged"] and skip_unchanged)]
    for i in range(0, len(pending), 96):
        chunk = pending[i : i + 96]
        vecs = mesh.embed([embed_text(p) for p in chunk])
        collection().upsert(
            ids=[str(p["id"]) for p in chunk],
            embeddings=[v.tolist() for v in vecs],
            metadatas=[_meta(p) for p in chunk],
            documents=[p["title"] for p in chunk],
        )
        log.info("indexed %d/%d", min(i + len(chunk), len(pending)), len(pending))
    _cached_product_vector.cache_clear()
    return len(pending)


def delete_product(product_id: int) -> None:
    _cached_product_vector.cache_clear()
    with db.tx() as c:
        c.execute("DELETE FROM products WHERE id = ?", (product_id,))
        c.execute("DELETE FROM products_fts WHERE rowid = ?", (product_id,))
    collection().delete(ids=[str(product_id)])


@lru_cache(maxsize=4096)
def _cached_product_vector(product_id: int, stamp: str) -> bytes | None:
    """`stamp` is the product's content_hash: it is only in the signature so
    that editing a product busts this cache automatically."""
    row = db.q1(
        "SELECT title, description, category, level, skills FROM products WHERE id = ?",
        (product_id,),
    )
    if not row:
        return None
    hit = db.q1("SELECT vec FROM embed_cache WHERE text_hash = ?",
                (mesh.cache_key(embed_text(dict(row))),))
    return hit["vec"] if hit else None


def product_vector(product_id: int) -> np.ndarray | None:
    """The catalog embedding for a product, read locally.

    NEVER a Chroma fetch: Chroma omits embeddings from get() by default, and
    this sits in the hot path of every single tracked event.

    Looks up embed_cache by the MODEL-SCOPED key (mesh.cache_key), not by
    products.content_hash -- those are different hashes answering different
    questions, and joining on the wrong one silently returns None for every
    product, which quietly kills the whole intent engine.
    """
    row = db.q1("SELECT content_hash FROM products WHERE id = ?", (product_id,))
    if not row:
        return None
    return db.from_blob(_cached_product_vector(product_id, row["content_hash"]))


def sync_status() -> dict:
    """Powers the admin sync badge. These two numbers must always match."""
    sql = db.q1("SELECT COUNT(*) n FROM products WHERE is_active = 1")["n"]
    try:
        vec = collection().count()
    except Exception as e:  # a broken index should show as red, not 500 the page
        log.warning("chroma count failed: %s", e)
        vec = -1
    fts = db.q1("SELECT COUNT(*) n FROM products_fts")["n"]
    return {"sql": sql, "vector": vec, "fts": fts, "in_sync": sql == vec == fts}
