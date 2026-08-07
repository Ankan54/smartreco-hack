"""Behavioural event ingest and the state the frontend polls.

The brief requires tracking that is "efficient and non-blocking -- it must not
slow down or break the frontend". Concretely:

  * the client batches and sends via navigator.sendBeacon (see track.js)
  * this endpoint returns 202 after a single executemany, before ANY embedding
    or LLM work happens
  * the intent update runs in a BackgroundTask
  * events carry a client-generated UUID and insert with OR IGNORE, because
    sendBeacon double-fires on tab close (pagehide AND visibilitychange) and
    duplicated events would silently corrupt the drift signal
"""

import json
import logging

from fastapi import APIRouter, BackgroundTasks, Depends, Request
from pydantic import BaseModel, Field

from app import agent, auth, db, dossier, intent, retrieval

log = logging.getLogger(__name__)
router = APIRouter(prefix="/api")

ALLOWED = {"view", "click", "scroll", "search", "dwell", "enroll", "bounce"}


class Event(BaseModel):
    id: str = Field(max_length=64)
    type: str = Field(max_length=16)
    session_id: str = Field(default="web", max_length=64)
    product_id: int | None = None
    query: str | None = Field(default=None, max_length=200)
    value: float | None = None
    ts: str | None = None


class EventBatch(BaseModel):
    events: list[Event] = Field(max_length=50)


def process(user_id: int, events: list[dict]) -> None:
    """Runs in a BackgroundTask. Never let this raise into the response path."""
    try:
        state = intent.apply_events(user_id, events)
        fire, reason, drift = intent.should_fire(state)
        if not fire:
            log.debug("user %s: no fire (%s) drift=%.4f", user_id, reason, drift)
            return
        log.info("user %s: firing agent (%s) drift=%.4f events=%s",
                 user_id, reason, drift, state["events_since_reco"])
        agent.run(user_id, reason, drift=drift)
    except Exception:
        # A failed agent run must never break tracking, and must never leave the
        # user stuck: the drift baseline is untouched, so the next batch retries.
        log.exception("agent pipeline failed for user %s", user_id)


@router.post("/events", status_code=202)
async def ingest(batch: EventBatch, bg: BackgroundTasks, request: Request):
    user = auth.current_user(request)
    if not user:
        return {"accepted": 0}          # anonymous browsing is not tracked

    rows, keep = [], []
    for e in batch.events:
        if e.type not in ALLOWED:
            continue
        rows.append((e.id, user["id"], e.session_id, e.type, e.product_id,
                     e.query, e.value, e.ts or ""))
        keep.append(e.model_dump())

    if rows:
        with db.tx() as c:
            c.executemany(
                "INSERT OR IGNORE INTO events(id,user_id,session_id,type,product_id,"
                "query,value,ts) VALUES (?,?,?,?,?,?,?,COALESCE(NULLIF(?,''),datetime('now')))",
                rows,
            )
        bg.add_task(process, user["id"], keep)

    return {"accepted": len(rows)}


@router.get("/recommendations")
def recommendations(since: int = 0, user=Depends(auth.require_user)):
    """Polled by the dashboard. `since` lets the client re-render only when the
    agent has actually produced something new."""
    row = db.q1("SELECT * FROM recommendations WHERE user_id = ? AND is_current = 1 "
                "ORDER BY id DESC LIMIT 1", (user["id"],))
    if not row or row["id"] <= since:
        return {"changed": False}
    items = []
    for i in json.loads(row["items_json"]):
        p = db.q1("SELECT id,title,category,level,price FROM products WHERE id = ?",
                  (i["product_id"],))
        if p:
            items.append({**i, "product": dict(p)})
    return {"changed": True, "id": row["id"], "headline": row["headline"],
            "narrative": row["narrative"], "trigger": row["trigger"],
            "drift": row["drift"], "created_at": row["created_at"], "items": items}


@router.get("/horizon")
def horizon(user=Depends(auth.require_user)):
    """Polled by horizon.js. Returns the real drift -- the tilt is never faked."""
    return intent.horizon_state(user["id"])


# --- the dossier: read, and argue with ------------------------------------

def _dossier_payload(user_id: int) -> dict:
    d = dossier.current(user_id)
    if not d:
        return {"version": 0, "prose": "", "claims": [], "changed": {}}
    hist = dossier.history(user_id, limit=2)
    prev = hist[1] if len(hist) > 1 else None
    return {"version": d["version"], "prose": d["prose"], "source": d["source"],
            "claims": d["claims"], "changed": dossier.diff(d, prev)}


@router.get("/dossier")
def get_dossier(user=Depends(auth.require_user)):
    return _dossier_payload(user["id"])


@router.post("/dossier/claims/{claim_id}/toggle")
def toggle_claim(claim_id: str, enabled: bool = True, user=Depends(auth.require_user)):
    """Strike a claim out, or put it back.

    Deliberately ZERO LLM calls: a '+interest' claim IS a retrieval probe, so
    disabling it removes that probe from the next query. The prose is
    regenerated by template. Nobody should pay for a model round-trip to untick
    a box, and the instant response is what makes the mechanism believable.

    Returns the freshly retrieved picks so the caller can show the consequence
    immediately, with the probe that surfaced each one.
    """
    uid = user["id"]
    d = dossier.set_enabled(uid, claim_id, enabled)
    if not d:
        return {"error": "no dossier yet"}

    r = retrieval.retrieve(uid, claims=dossier.enabled_claims(uid))
    picks = [{
        "product_id": c["id"],
        "product": {"id": c["id"], "title": c["title"], "category": c["category"],
                    "level": c["level"], "price": c["price"]},
        "because": c.get("because", []),
        "rrf": c["rrf"],
    } for c in r["candidates"][:5]]

    log.info("user %s toggled claim %s -> %s (0 LLM calls)", uid, claim_id, enabled)
    return {"dossier": _dossier_payload(uid), "picks": picks, "llm_calls": 0}


@router.post("/recommendations/refresh")
def refresh(bg: BackgroundTasks, user=Depends(auth.require_user)):
    """Explicit user request for a fresh narrative. This one does cost LLM calls,
    which is why it is a button and not automatic."""
    bg.add_task(agent.run, user["id"], "manual", None)
    return {"queued": True}
