"""The ONLY place an LLM or embedding client is constructed in this repo.

Every AI call in this project goes through Mesh API (https://api.meshapi.ai/v1),
which is mandatory for this hackathon. Developing against OpenAI directly is an
env-var swap (MESH_BASE_URL + model ids) -- never a second code path.

Behaviours below are measured against the live Mesh API, not assumed:
  * Mesh degrades SILENTLY on models that ignore response_format -- you get prose
    with finish_reason 'stop', no error. Hence the JSONDecodeError branch.
  * The /models catalog reports supports_structured_output=False for gpt-5-nano,
    which is WRONG -- it works. Trust the live probe, not the flag.
  * gpt-5-nano rejects reasoning_effort='none'; 'low' is the cheapest accepted.
  * temperature is omitted entirely: OpenAI-direct rejects !=1 on gpt-5, and
    omitting it still qualifies for Mesh's gateway response cache.
"""

import hashlib
import json
import logging

import numpy as np
from openai import OpenAI

from app import config, db

log = logging.getLogger(__name__)

_raw = OpenAI(base_url=config.MESH_BASE_URL, api_key=config.MESH_API_KEY, timeout=90.0)

if config.LANGSMITH_API_KEY:
    import os

    os.environ.setdefault("LANGSMITH_TRACING", "true")
    os.environ.setdefault("LANGSMITH_PROJECT", config.LANGSMITH_PROJECT)
    from langsmith.wrappers import wrap_openai

    client = wrap_openai(_raw)
else:
    client = _raw


def chat(messages: list[dict], schema: dict | None = None, model: str | None = None):
    """Returns (result, meta). result is a dict when schema is given, else str,
    or None if the model degraded to prose. Never raises on bad JSON."""
    kwargs: dict = {}
    if schema:
        kwargs["response_format"] = {
            "type": "json_schema",
            "json_schema": {"name": schema["name"], "strict": True, "schema": schema["schema"]},
        }

    raw = client.chat.completions.with_raw_response.create(
        model=model or config.CHAT_MODEL,
        messages=messages,
        reasoning_effort=config.REASONING_EFFORT,
        **kwargs,
    )
    completion = raw.parse()
    usage = completion.usage
    meta = {
        "model": model or config.CHAT_MODEL,
        "cache": raw.headers.get("x-cache", "MISS"),
        "request_id": raw.headers.get("x-request-id"),
        "prompt_tokens": getattr(usage, "prompt_tokens", None),
        "completion_tokens": getattr(usage, "completion_tokens", None),
    }
    text = completion.choices[0].message.content or ""
    if not schema:
        return text, meta
    try:
        return json.loads(text), meta
    except json.JSONDecodeError:
        log.warning("Mesh degraded to prose for schema %s", schema["name"])
        meta["degraded"] = True
        return None, meta


# --- embeddings ------------------------------------------------------------

def cache_key(text: str) -> str:
    """embed_cache key. Model-scoped on purpose: the same text under a different
    embedding model is a different vector, so switching models must miss the
    cache rather than silently return stale vectors from the old space.

    NOTE this is deliberately NOT the same as vectors.content_hash(), which
    hashes only the text and answers a different question ("has this product's
    text changed?"). Conflating the two is what made product_vector() always
    return None the first time round.
    """
    return hashlib.sha256(f"{config.EMBED_MODEL}\x00{text}".encode()).hexdigest()


_hash = cache_key   # backwards-compatible alias


def embed(texts: list[str]) -> list[np.ndarray]:
    """Batched, cached by content hash. Re-embedding unchanged text is the most
    wasteful thing this app could do, so it simply never happens."""
    if not texts:
        return []
    hashes = [_hash(t) for t in texts]
    cached: dict[str, np.ndarray] = {}
    placeholders = ",".join("?" * len(set(hashes)))
    rows = db.q(
        f"SELECT text_hash, vec FROM embed_cache WHERE text_hash IN ({placeholders})",
        tuple(set(hashes)),
    )
    for r in rows:
        cached[r["text_hash"]] = db.from_blob(r["vec"])

    missing = [(h, t) for h, t in zip(hashes, texts) if h not in cached]
    # dedupe within this batch too
    seen: dict[str, str] = {}
    for h, t in missing:
        seen.setdefault(h, t)

    if seen:
        items = list(seen.items())
        for i in range(0, len(items), 96):
            chunk = items[i : i + 96]
            resp = client.embeddings.create(
                model=config.EMBED_MODEL, input=[t for _, t in chunk]
            )
            # Mesh preserves input order in data[] -- documented and verified.
            with db.tx() as c:
                for (h, _), item in zip(chunk, resp.data):
                    v = np.asarray(item.embedding, dtype=np.float32)
                    cached[h] = v
                    c.execute(
                        "INSERT OR IGNORE INTO embed_cache(text_hash, model, vec) VALUES (?,?,?)",
                        (h, config.EMBED_MODEL, db.to_blob(v)),
                    )

    return [cached[h] for h in hashes]


def embed_one(text: str) -> np.ndarray:
    return embed([text])[0]
