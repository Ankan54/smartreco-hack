"""Hybrid, multi-probe retrieval over the catalog.

Two ideas do the work here.

**Multi-probe.** A single mean over a mixed browsing session lands in dead space
*between* clusters and retrieves nothing good. So we never query with one
vector: we probe with the overall bearing, with the single thing the user is
most focused on right now, with their last search in words, and with the
dossier's positive claims.

**Hybrid.** Dense embeddings are reliably bad at rare proper nouns and
acronyms -- "PySpark", "CISSP", "Terraform" -- which is exactly what a user
types into a search box. SQLite's FTS5 gives us BM25 for free (no dependency),
so a literal probe runs alongside the semantic ones.

Fusion is Reciprocal Rank Fusion, and RRF is the right choice *because* BM25
scores and cosine distances share no scale. Only rank position is comparable
between them, which is all RRF uses.
"""

import logging
import re

import numpy as np

from app import config, db, intent, mesh, vectors

log = logging.getLogger(__name__)

RECENT_FOCUS_EVENTS = 8
QUERY_WINDOW_MIN = 30


# --- probe construction ----------------------------------------------------

def _recent_events(user_id: int, limit: int = 40) -> list[dict]:
    return [dict(r) for r in db.q(
        "SELECT e.*, p.title, p.category FROM events e "
        "LEFT JOIN products p ON p.id = e.product_id "
        "WHERE e.user_id = ? ORDER BY e.rowid DESC LIMIT ?", (user_id, limit))]


def _focus_product(events: list[dict]) -> int | None:
    """The single heaviest product in the last few events -- what the user is
    circling right now, as opposed to the session average."""
    best, best_w = None, 0.0
    for e in events[:RECENT_FOCUS_EVENTS]:
        if not e.get("product_id"):
            continue
        w = config.EVENT_WEIGHTS.get(e["type"], 0)
        if w > best_w:
            best, best_w = e["product_id"], w
    return best


def _latest_query(events: list[dict]) -> str | None:
    for e in events:
        if e["type"] == "search" and (e.get("query") or "").strip():
            return e["query"].strip()
    return None


def build_probes(user_id: int, claims: list[dict] | None = None) -> list[dict]:
    """Returns [{name, kind, weight, vector|text}]. Cheap: everything is either
    already-cached embedding or a local read."""
    events = _recent_events(user_id)
    state = intent.get_state(user_id)
    probes: list[dict] = []

    bearing = intent.unit(state["intent_vec"])
    if bearing is not None:
        probes.append({"name": "bearing", "kind": "dense",
                       "weight": config.PROBE_WEIGHTS["bearing"], "vector": bearing})

    pid = _focus_product(events)
    if pid:
        v = vectors.product_vector(pid)
        if v is not None:
            probes.append({"name": "focus", "kind": "dense", "product_id": pid,
                           "weight": config.PROBE_WEIGHTS["focus"], "vector": v})

    q = _latest_query(events)
    if q:
        probes.append({"name": "words", "kind": "dense", "text": q,
                       "weight": config.PROBE_WEIGHTS["words"],
                       "vector": mesh.embed_one(q)})
        # The literal half of the hybrid: BM25 finds the exact token that
        # embeddings blur away.
        probes.append({"name": "literal", "kind": "bm25", "text": q,
                       "weight": config.PROBE_WEIGHTS["literal"]})

    for c in claims or []:
        if not c.get("enabled", True) or c.get("kind") != "interest":
            continue
        w = config.CLAIM_PROBE_WEIGHT * float(c.get("strength", 0.5))
        probes.append({
            "name": f"claim:{c['id']}", "kind": "dense",
            "polarity": c.get("polarity", "+"), "text": c["text"],
            "weight": w if c.get("polarity", "+") == "+" else
                      config.CLAIM_PENALTY_WEIGHT * float(c.get("strength", 0.5)),
            "vector": mesh.embed_one(c["text"]),
        })
    return probes


# --- the probes themselves -------------------------------------------------

def _fts_match(text: str) -> str | None:
    """FTS5 MATCH has its own syntax; raw user input will throw on characters
    like '-' or '"'. Reduce to quoted alphanumeric terms OR'd together."""
    terms = [t for t in re.findall(r"[A-Za-z0-9+#.]{2,}", text)][:8]
    return " OR ".join(f'"{t}"' for t in terms) if terms else None


def _dense_rank(vec: np.ndarray, where: dict | None, k: int) -> list[int]:
    res = vectors.collection().query(
        query_embeddings=[np.asarray(vec, dtype=np.float32).tolist()],
        n_results=k, where=where or None,
    )
    return [int(i) for i in res["ids"][0]] if res["ids"] else []


def _bm25_rank(text: str, where: dict | None, k: int) -> list[int]:
    match = _fts_match(text)
    if not match:
        return []
    sql = ("SELECT f.rowid AS id FROM products_fts f JOIN products p ON p.id = f.rowid "
           "WHERE products_fts MATCH ? AND p.is_active = 1")
    args: list = [match]
    for clause, val in (where or {}).items():
        if clause == "category":
            sql += " AND p.category = ?"; args.append(val)
        elif clause == "level":
            sql += " AND p.level = ?"; args.append(val)
        elif clause == "price":
            sql += " AND p.price <= ?"; args.append(val["$lte"])
    sql += " ORDER BY bm25(products_fts) LIMIT ?"
    args.append(k)
    try:
        return [r["id"] for r in db.q(sql, tuple(args))]
    except Exception as e:                       # malformed MATCH must not 500
        log.warning("fts query failed for %r: %s", text, e)
        return []


def _chroma_where(filters: dict | None) -> dict | None:
    """Translate the agent's filter request into Chroma's `where` syntax.
    No is_active clause: inactive products are not in the index at all."""
    if not filters:
        return None
    clauses = []
    if filters.get("category"):
        clauses.append({"category": filters["category"]})
    if filters.get("level"):
        clauses.append({"level": filters["level"]})
    if filters.get("max_price"):
        clauses.append({"price": {"$lte": float(filters["max_price"])}})
    if not clauses:
        return None
    return clauses[0] if len(clauses) == 1 else {"$and": clauses}


# --- fusion ----------------------------------------------------------------

def retrieve(user_id: int, claims: list[dict] | None = None,
             filters: dict | None = None, limit: int | None = None) -> dict:
    """Run every probe, fuse by RRF, exclude what the user has already seen.
    Returns {candidates, probes, excluded} -- the probe detail is what Agent Cam
    renders, so nothing here is a black box."""
    limit = limit or config.CANDIDATE_LIMIT
    probes = build_probes(user_id, claims)
    where = _chroma_where(filters)

    scores: dict[int, float] = {}
    # Which probes surfaced each candidate, and how much each contributed. This
    # is what lets the UI say "this pick is here because of that claim" -- and
    # what makes striking a claim out a visible, causal change rather than a
    # vague promise.
    credit: dict[int, list[tuple[str, float]]] = {}
    detail: list[dict] = []

    for p in probes:
        if p["kind"] == "dense":
            ranked = _dense_rank(p["vector"], where, config.PROBE_TOP_K)
        else:
            ranked = _bm25_rank(p["text"], where, config.PROBE_TOP_K)

        negative = p.get("polarity") == "-"
        for rank, pid in enumerate(ranked):
            contrib = p["weight"] / (config.RRF_K + rank + 1)
            # A negative claim SUBTRACTS from anything it ranks highly.
            scores[pid] = scores.get(pid, 0.0) + (-contrib if negative else contrib)
            credit.setdefault(pid, []).append(
                (p["name"], -contrib if negative else contrib))

        detail.append({"name": p["name"], "kind": p["kind"], "weight": round(p["weight"], 3),
                       "polarity": p.get("polarity", "+"), "text": p.get("text"),
                       "hits": ranked[:8]})

    # Exclude what they have already given real attention to. Handing someone
    # back the course they just spent a minute reading looks broken, however
    # good the similarity score is -- the job is to find the NEXT thing.
    #   enroll  -> committed
    #   dwell   -> read it properly
    #   view x2 -> came back to it, they know it exists
    excluded = {r["product_id"] for r in db.q(
        "SELECT product_id FROM events "
        " WHERE user_id = ? AND product_id IS NOT NULL "
        "   AND type IN ('enroll','dwell','view') "
        " GROUP BY product_id "
        "HAVING SUM(type IN ('enroll','dwell')) > 0 OR COUNT(*) >= 2", (user_id,))}

    ordered = sorted(
        ((pid, s) for pid, s in scores.items() if s > 0 and pid not in excluded),
        key=lambda kv: kv[1], reverse=True,
    )[:limit]

    label = {p["name"]: (p.get("text") or p["name"]) for p in probes}
    candidates = []
    for pid, score in ordered:
        row = db.q1("SELECT * FROM products WHERE id = ? AND is_active = 1", (pid,))
        if not row:
            continue
        c = dict(row)
        c["rrf"] = round(score, 5)
        top = sorted(credit.get(pid, []), key=lambda kv: kv[1], reverse=True)[:2]
        c["because"] = [{"probe": n, "label": label.get(n, n), "share": round(v, 5)}
                        for n, v in top if v > 0]
        candidates.append(c)

    log.info("retrieve user=%s probes=%d pooled=%d excluded=%d -> %d candidates",
             user_id, len(probes), len(scores), len(excluded), len(candidates))
    return {"candidates": candidates, "probes": detail, "excluded": sorted(excluded)}
