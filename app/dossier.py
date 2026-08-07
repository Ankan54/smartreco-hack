"""The slow tier: a short list of typed claims the agent believes about a user.

Store claims, generate the paragraph FROM them. Getting that backwards breaks
everything downstream, because a paragraph of commentary about a person ("you
keep bouncing off beginner material") embeds nothing like a course description,
and a probe has to live in the same space as the documents it searches.

Each claim kind acts on retrieval mechanically, so editing the dossier is a
set-membership change rather than a re-embedding problem:

    + interest   -> becomes a retrieval probe
    - interest   -> subtracts from anything it ranks highly
    + constraint -> becomes a metadata filter

That is why striking out a claim changes the next recommendation instantly and
costs zero LLM calls.

Versions are append-only: the UI diffs v(n) against v(n-1), and the history is
what makes the agent's belief-change legible instead of magical.
"""

import json
import logging

from app import db

log = logging.getLogger(__name__)

MAX_CLAIMS = 8


def current(user_id: int) -> dict | None:
    row = db.q1(
        "SELECT * FROM dossier_versions WHERE user_id = ? ORDER BY version DESC LIMIT 1",
        (user_id,),
    )
    if not row:
        return None
    d = dict(row)
    d["claims"] = json.loads(d["claims_json"])
    return d


def history(user_id: int, limit: int = 10) -> list[dict]:
    out = []
    for r in db.q("SELECT * FROM dossier_versions WHERE user_id = ? "
                  "ORDER BY version DESC LIMIT ?", (user_id, limit)):
        d = dict(r)
        d["claims"] = json.loads(d["claims_json"])
        out.append(d)
    return out


def enabled_claims(user_id: int) -> list[dict]:
    d = current(user_id)
    return [c for c in (d["claims"] if d else []) if c.get("enabled", True)]


def save(user_id: int, claims: list[dict], prose: str, source: str) -> dict:
    """Append a new version. Never mutates an existing one -- the history is
    what powers the on-screen diff."""
    prev = current(user_id)
    version = (prev["version"] + 1) if prev else 1

    clean = []
    for i, c in enumerate(claims[:MAX_CLAIMS]):
        clean.append({
            "id": str(c.get("id") or f"c{i + 1}"),
            "polarity": "-" if c.get("polarity") == "-" else "+",
            "kind": "constraint" if c.get("kind") == "constraint" else "interest",
            "text": (c.get("text") or "").strip(),
            "strength": max(0.0, min(1.0, float(c.get("strength", 0.5)))),
            "evidence": (c.get("evidence") or "").strip(),
            "enabled": bool(c.get("enabled", True)),
        })
    clean = [c for c in clean if c["text"]]

    with db.tx() as conn:
        conn.execute(
            "INSERT INTO dossier_versions(user_id,version,claims_json,prose,source) "
            "VALUES (?,?,?,?,?)",
            (user_id, version, json.dumps(clean), prose.strip(), source),
        )
    log.info("dossier user=%s v%d source=%s claims=%d", user_id, version, source, len(clean))
    return current(user_id)


def set_enabled(user_id: int, claim_id: str, enabled: bool) -> dict | None:
    """A user striking out a claim. Writes a new version with source='user_edit'
    and regenerates the prose by template -- deliberately NO LLM call, because
    nobody should pay for a model round-trip to untick a box."""
    d = current(user_id)
    if not d:
        return None
    claims = [dict(c) for c in d["claims"]]
    found = False
    for c in claims:
        if c["id"] == claim_id:
            c["enabled"] = enabled
            found = True
    if not found:
        return d
    return save(user_id, claims, prose_from_claims(claims), "user_edit")


def prose_from_claims(claims: list[dict]) -> str:
    """Template fallback for user edits. The LLM writes better prose during
    reflection; this only has to stay honest and readable."""
    likes = [c["text"] for c in claims if c.get("enabled", True)
             and c["polarity"] == "+" and c["kind"] == "interest"]
    avoids = [c["text"] for c in claims if c.get("enabled", True)
              and c["polarity"] == "-"]
    limits = [c["text"] for c in claims if c.get("enabled", True)
              and c["kind"] == "constraint" and c["polarity"] == "+"]

    if not (likes or avoids or limits):
        return "Nothing to go on yet."

    bits = []
    if likes:
        bits.append("You're circling " + _join(likes) + ".")
    if avoids:
        bits.append("You skip " + _join(avoids) + ".")
    if limits:
        bits.append("You stay within " + _join(limits) + ".")
    return " ".join(bits)


def _join(items: list[str]) -> str:
    items = items[:3]
    if len(items) == 1:
        return items[0]
    return ", ".join(items[:-1]) + " and " + items[-1]


def diff(newer: dict, older: dict | None) -> dict:
    """What changed between two versions, for the UI."""
    if not older:
        return {"added": [c["id"] for c in newer["claims"]], "removed": [], "changed": []}
    old = {c["id"]: c for c in older["claims"]}
    new = {c["id"]: c for c in newer["claims"]}
    added = [i for i in new if i not in old]
    removed = [i for i in old if i not in new]
    changed = [i for i in new if i in old and (
        new[i]["text"] != old[i]["text"]
        or new[i]["enabled"] != old[i]["enabled"]
        or abs(new[i]["strength"] - old[i]["strength"]) > 0.05)]
    return {"added": added, "removed": removed, "changed": changed}
