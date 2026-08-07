"""Step 2 of 4: add what the source data lacks -> data/catalog.json

Coursera's export has no category taxonomy, no price, and no prerequisite links.
The recommender needs all three: categories drive metadata filtering and the
cold-start centroids, price is a dossier constraint, and prerequisites are what
let the agent order its picks as a progression instead of a flat list.

Runs ONCE and its output is committed, so it uses a bigger model than the app
does -- quality here compounds across every later demo. Resumable: existing
entries in data/catalog.json are kept, so a crash mid-run costs only the
unfinished chunks.

    uv run python scripts/enrich_catalog.py [--model openai/gpt-5-mini]
"""

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, mesh  # noqa: E402

CHUNK = 20

SCHEMA = {
    "name": "enriched_courses",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["courses"],
        "properties": {
            "courses": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "category", "level", "price_inr", "tagline"],
                    "properties": {
                        "id": {"type": "integer"},
                        "category": {"type": "string", "enum": config.CATEGORIES},
                        "level": {"type": "string",
                                  "enum": ["Beginner", "Intermediate", "Advanced"]},
                        "price_inr": {"type": "integer"},
                        "tagline": {"type": "string"},
                    },
                },
            }
        },
    },
}

SYSTEM = f"""You are cataloguing an online course marketplace for an Indian audience.

For each course, decide:
- category: exactly one of {config.CATEGORIES}. Choose by what the learner would
  actually be doing, not by the awarding organisation.
- level: Beginner, Intermediate or Advanced. Trust the course content over any
  level already claimed in the title.
- price_inr: a realistic Indian market price, 0 for genuinely introductory
  material, otherwise 799-6999. Advanced and specialised courses cost more.
  Use varied, non-round numbers so the catalogue does not look generated.
- tagline: one sentence, max 90 characters, stating the concrete outcome. No
  marketing adjectives, no "master", no "unlock", no exclamation marks.

Return every course you were given, keyed by the id you were given."""


def load_json(path: Path, default):
    return json.loads(path.read_text(encoding="utf-8")) if path.exists() else default


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5-mini",
                    help="bigger than the app default on purpose; output is committed")
    args = ap.parse_args()

    raw = load_json(config.DATA_DIR / "catalog_raw.json", None)
    if raw is None:
        sys.exit("run scripts/download_catalog.py first")

    out_path = config.CATALOG_JSON
    done = {c["id"]: c for c in load_json(out_path, [])}
    todo = [p for p in raw if p["id"] not in done]
    print(f"{len(raw)} courses, {len(done)} already enriched, {len(todo)} to go")

    for i in range(0, len(todo), CHUNK):
        chunk = todo[i : i + CHUNK]
        payload = [
            {"id": p["id"], "title": p["title"], "skills": p["skills"],
             "description": p["description"][:600], "claimed_level": p["level"]}
            for p in chunk
        ]
        result, meta = mesh.chat(
            [{"role": "system", "content": SYSTEM},
             {"role": "user", "content": json.dumps(payload, ensure_ascii=False)}],
            schema=SCHEMA, model=args.model,
        )
        if result is None:
            print(f"  chunk {i // CHUNK}: model returned prose, skipping (rerun to retry)")
            continue

        by_id = {c["id"]: c for c in result["courses"]}
        for p in chunk:
            e = by_id.get(p["id"])
            if not e:
                continue
            done[p["id"]] = {
                "id": p["id"],
                "title": p["title"],
                "description": p["description"],
                "category": e["category"],
                "level": e["level"],
                "price": float(e["price_inr"]),
                "provider": p["provider"],
                "skills": p["skills"],
                "rating": p["rating"],
                "tagline": e["tagline"],
                "url": p["url"],
                "prereq_ids": [],          # filled by link_prerequisites()
                "synthetic": False,
            }
        # Write after every chunk so a crash costs one chunk, not the whole run.
        out_path.write_text(
            json.dumps(sorted(done.values(), key=lambda c: c["id"]), indent=1,
                       ensure_ascii=False), encoding="utf-8")
        print(f"  {len(done)}/{len(raw)}  cache={meta['cache']}  "
              f"tokens={meta.get('prompt_tokens')}+{meta.get('completion_tokens')}")

    link_prerequisites(list(done.values()))
    out_path.write_text(
        json.dumps(sorted(done.values(), key=lambda c: c["id"]), indent=1,
                   ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(done)} courses -> {out_path}")


def link_prerequisites(courses: list[dict]) -> None:
    """Beginner -> Intermediate -> Advanced within a category. No LLM needed:
    the ordering is already implied by level, and a deterministic rule here is
    both cheaper and more consistent than asking a model to invent edges."""
    rank = {"Beginner": 0, "Intermediate": 1, "Advanced": 2}
    by_cat: dict[str, list[dict]] = {}
    for c in courses:
        by_cat.setdefault(c["category"], []).append(c)

    linked = 0
    for group in by_cat.values():
        group.sort(key=lambda c: (rank.get(c["level"], 1), -(c["rating"] or 0)))
        tiers: dict[int, list[dict]] = {0: [], 1: [], 2: []}
        for c in group:
            tiers[rank.get(c["level"], 1)].append(c)
        for tier in (1, 2):
            below = tiers[tier - 1][:2]     # up to 2 prerequisites, best-rated first
            for c in tiers[tier]:
                c["prereq_ids"] = [b["id"] for b in below]
                linked += bool(below)
    print(f"linked prerequisites for {linked} courses")


if __name__ == "__main__":
    main()
