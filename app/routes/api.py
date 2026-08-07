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

from app import agent, auth, db, intent

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
