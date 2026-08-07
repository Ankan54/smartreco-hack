"""Caching. The brief judges "don't fire an LLM call on every user action" and
"use caching to avoid wasteful, redundant calls", so these are load-bearing.

Three layers, cheapest first:
  1. in-process LRU      -- no I/O
  2. embed_cache (SQLite) -- survives restarts
  3. Mesh gateway cache   -- x-cache: HIT, zero tokens billed
Only the first two are ours to test.
"""

import uuid

import numpy as np

from app import mesh


class Counter:
    """Counts real API calls so a cache hit is provable, not assumed."""

    def __init__(self, payload):
        self.n = 0
        self.payload = payload

    def create(self, **kwargs):
        self.n += 1
        return self.payload


def test_identical_chat_prompts_do_not_hit_the_network(monkeypatch):
    class FakeRaw:
        headers = {"x-cache": "MISS", "x-request-id": "req_1"}

        @staticmethod
        def parse():
            class C:
                usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
                choices = [type("Ch", (), {"message": type("M", (), {"content": "Django"})()})()]
            return C()

    counter = Counter(FakeRaw())
    monkeypatch.setattr(mesh.client.chat.completions, "with_raw_response", counter)
    mesh._chat_cache._d.clear()

    msgs = [{"role": "user", "content": f"unique prompt {uuid.uuid4()}"}]
    r1, m1 = mesh.chat(msgs)
    r2, m2 = mesh.chat(msgs)

    assert counter.n == 1, "second identical prompt still called the API"
    assert r1 == r2 == "Django"
    assert m2["cache"] == "memory"


def test_a_degraded_response_is_never_cached(monkeypatch):
    """Mesh returns prose when a model ignores response_format. Memoising that
    would make one bad roll permanent for the process lifetime."""
    class FakeRaw:
        headers = {"x-cache": "MISS"}

        @staticmethod
        def parse():
            class C:
                usage = type("U", (), {"prompt_tokens": 5, "completion_tokens": 2})()
                choices = [type("Ch", (), {"message": type("M", (), {
                    "content": "Sorry, here is some prose instead."})()})()]
            return C()

    counter = Counter(FakeRaw())
    monkeypatch.setattr(mesh.client.chat.completions, "with_raw_response", counter)
    mesh._chat_cache._d.clear()

    schema = {"name": "probe", "schema": {"type": "object", "properties": {}}}
    msgs = [{"role": "user", "content": f"prompt {uuid.uuid4()}"}]

    r1, m1 = mesh.chat(msgs, schema=schema)
    assert r1 is None and m1.get("degraded")

    mesh.chat(msgs, schema=schema)
    assert counter.n == 2, "a degraded response was cached and the retry never happened"


def test_repeat_embeddings_skip_both_the_api_and_sqlite(monkeypatch):
    dim = 8
    payload = type("R", (), {"data": [type("D", (), {"embedding": [0.1] * dim})()]})()
    counter = Counter(payload)
    monkeypatch.setattr(mesh.client, "embeddings", counter)
    mesh._embed_cache._d.clear()

    text = f"probe {uuid.uuid4()}"
    v1 = mesh.embed([text])[0]
    v2 = mesh.embed([text])[0]

    assert counter.n == 1, "the second embed call hit the API"
    assert np.array_equal(v1, v2)

    # And drop the in-process layer: SQLite must still serve it without the API.
    mesh._embed_cache._d.clear()
    v3 = mesh.embed([text])[0]
    assert counter.n == 1, "SQLite layer missed after the memory layer was cleared"
    assert np.array_equal(v1, v3)


def test_a_batch_with_duplicates_embeds_each_text_once(monkeypatch):
    dim = 8
    calls = []

    class Emb:
        @staticmethod
        def create(model, input):
            calls.append(list(input))
            return type("R", (), {"data": [type("D", (), {"embedding": [0.1] * dim})()
                                           for _ in input]})()

    monkeypatch.setattr(mesh.client, "embeddings", Emb())
    mesh._embed_cache._d.clear()

    a, b = f"x {uuid.uuid4()}", f"y {uuid.uuid4()}"
    out = mesh.embed([a, b, a, b, a])

    assert len(out) == 5, "caller must get one vector per input, in order"
    assert len(calls) == 1 and len(calls[0]) == 2, f"deduped batch expected, got {calls}"


def test_lru_evicts_and_stays_bounded():
    """A cache that grows forever is a memory leak wearing a disguise."""
    lru = mesh._LRU(maxsize=3)
    for i in range(10):
        lru.put(f"k{i}", i)
    assert lru.stats()["size"] == 3
    assert lru.get("k0") is None      # evicted
    assert lru.get("k9") == 9         # newest survives
