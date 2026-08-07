"""Verify the LLM configuration actually works -- and, with --strict, that it
points at Mesh.

Run this before writing agent code, and again before pushing. Mesh is mandatory
for this hackathon: a submission whose calls do not go through it is not a valid
submission, and the failure mode is silent (everything works locally against
OpenAI and the config is simply wrong at judging time).

    uv run python scripts/check_mesh.py            # does the config work?
    uv run python scripts/check_mesh.py --strict   # ...and is it Mesh? (CI gate)

Findings this script exists to protect, all measured against the live API:
  * Mesh degrades SILENTLY on models that ignore response_format -- prose comes
    back with finish_reason "stop" and no error.
  * The /models catalog reports supports_structured_output=False for
    openai/gpt-5-nano, which is WRONG. It works. Never trust the flag; run the
    call.
  * gpt-5-nano rejects reasoning_effort='none' and, on OpenAI direct, any
    temperature other than 1.
"""

import argparse
import json
import sys
from pathlib import Path

import httpx

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, mesh, schemas  # noqa: E402

OK, BAD = "  [PASS]", "  [FAIL]"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--strict", action="store_true",
                    help="fail unless the base URL is Mesh (use in CI before pushing)")
    args = ap.parse_args()
    failures = 0

    print(f"base_url : {config.MESH_BASE_URL}")
    print(f"chat     : {config.CHAT_MODEL}")
    print(f"generate : {config.GENERATE_MODEL}")
    print(f"embed    : {config.EMBED_MODEL}")
    print(f"key      : {config.MESH_API_KEY[:8]}...\n")

    # --- 0. the validity gate ---------------------------------------------
    is_mesh = "meshapi.ai" in config.MESH_BASE_URL
    print("> Routing through Mesh API")
    if is_mesh:
        print(OK, "base URL is Mesh")
    elif args.strict:
        print(BAD, "NOT routing through Mesh. Every LLM call must go through "
                   "Mesh API or the submission is invalid.")
        print("         Set MESH_BASE_URL=https://api.meshapi.ai/v1, use the rsk_ key,")
        print("         and prefix model ids with 'openai/'.")
        failures += 1
    else:
        print("  [WARN] not Mesh -- fine for local development, invalid to submit")

    # a mismatched key/URL pair is the usual half-finished switch
    if is_mesh and not config.MESH_API_KEY.startswith("rsk_"):
        print(BAD, "Mesh URL but the key is not an rsk_ key")
        failures += 1
    if not is_mesh and config.MESH_API_KEY.startswith("rsk_"):
        print(BAD, "OpenAI URL but the key is a Mesh rsk_ key -- switch is half done")
        failures += 1

    # --- 1. catalog --------------------------------------------------------
    if is_mesh:
        print("\n> Model catalog")
        try:
            r = httpx.get(f"{config.MESH_BASE_URL}/models",
                          headers={"Authorization": f"Bearer {config.MESH_API_KEY}"},
                          timeout=60)
            items = r.json()
            items = items if isinstance(items, list) else items.get("data", [])
            by_id = {m.get("id"): m for m in items}
            print(f"  [INFO] {len(items)} models available")
            for want in (config.CHAT_MODEL, config.GENERATE_MODEL, config.EMBED_MODEL):
                m = by_id.get(want)
                if not m:
                    print(BAD, f"{want} not in the catalog")
                    failures += 1
                else:
                    p = m.get("pricing") or {}
                    print(OK, f"{want}  in=${p.get('prompt_usd_per_1m')}/1M "
                              f"out=${p.get('completion_usd_per_1m')}/1M "
                              f"(catalog says structured_output="
                              f"{m.get('supports_structured_output')}, not trusted)")
        except Exception as e:
            print(BAD, f"catalog unreachable: {type(e).__name__}: {e}")
            failures += 1

    # --- 2. chat + structured output on a REAL schema ----------------------
    print("\n> Structured output on the dossier schema")
    summary = ("viewed 'LangGraph Multi-Agent Systems' 3 times, searched 'rag pipeline' "
               "twice, left 'Python for Absolute Beginners' after 2 seconds, every "
               "course opened was under Rs 3000")
    try:
        result, meta = mesh.chat(
            [{"role": "system", "content": "Extract belief claims about this learner. "
                                           "Include at least one negative claim."},
             {"role": "user", "content": summary}],
            schema=schemas.DOSSIER)
        if result is None:
            print(BAD, "model returned PROSE instead of JSON (silent degradation)")
            failures += 1
        else:
            claims = result.get("claims", [])
            neg = [c for c in claims if c.get("polarity") == "-"]
            print(OK, f"{len(claims)} claims, {len(neg)} negative, cache={meta['cache']}")
            if not neg:
                print("  [WARN] no negative claim -- the dossier will be a tag list. "
                      "Consider a stronger model for reflect.")
            for c in claims[:3]:
                print(f"         {c['polarity']} {c['text']}")
    except Exception as e:
        print(BAD, f"{type(e).__name__}: {str(e)[:200]}")
        failures += 1

    # --- 3. embeddings -----------------------------------------------------
    print("\n> Embeddings")
    try:
        vs = mesh.embed(["hello world", "quantum finance"])
        dim = vs[0].shape[0]
        print(OK, f"{len(vs)} vectors, dim={dim}")
        if dim != config.EMBED_DIM:
            print(BAD, f"EMBED_DIM is {config.EMBED_DIM} but the model returns {dim}. "
                       "Set EMBED_DIM and run scripts/reindex.py.")
            failures += 1
    except Exception as e:
        print(BAD, f"{type(e).__name__}: {str(e)[:200]}")
        failures += 1

    print("\n" + ("all checks passed" if not failures else f"{failures} check(s) failed"))
    return 1 if failures else 0


if __name__ == "__main__":
    sys.exit(main())
