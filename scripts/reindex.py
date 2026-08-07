"""Rebuild the vector index and FTS5 from SQLite, which stays authoritative.

Needed whenever the embedding model or the provider changes: ada-002 and
text-embedding-3-small are both 1536 dimensions, but their vectors are NOT
interchangeable, and neither are OpenAI-direct and Mesh for the same model id.
A stale index doesn't error -- it just quietly returns worse results, which is
the kind of bug you find during a demo.

    uv run python scripts/reindex.py
"""

import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, db, vectors  # noqa: E402


def main() -> None:
    db.init()
    rows = db.q("SELECT * FROM products WHERE is_active = 1")
    print(f"{len(rows)} active products in SQLite")

    try:
        vectors._client  # noqa: B018, SLF001
        vectors.collection()
        vectors._client.delete_collection(config.COLLECTION)  # noqa: SLF001
        print("dropped vector collection")
    except Exception as e:
        print(f"no collection to drop ({e.__class__.__name__})")
    vectors._collection = None

    with db.tx() as c:
        c.execute("DELETE FROM products_fts")
    print("cleared fts index")

    products = []
    for r in rows:
        p = dict(r)
        p["prereq_ids"] = json.loads(p.get("prereq_ids") or "[]")
        products.append(p)

    # skip_unchanged=False: the whole point is to re-embed even though the text
    # has not changed, because the model or provider has.
    n = vectors.upsert_many(products, skip_unchanged=False)
    print(f"re-embedded {n} products with {config.EMBED_MODEL}")

    s = vectors.sync_status()
    print(f"sql={s['sql']}  vector={s['vector']}  fts={s['fts']}  "
          f"{'IN SYNC' if s['in_sync'] else 'OUT OF SYNC'}")
    if not s["in_sync"]:
        sys.exit(1)


if __name__ == "__main__":
    main()
