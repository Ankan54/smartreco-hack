"""Step 3 of 4: generate the courses the real data doesn't provide.

Real Coursera data is lumpy. It gives us breadth but two specific gaps hurt the
demo, and both are structural rather than cosmetic:

  1. A category with four courses retrieves badly. When a user drifts into it,
     the agent has nothing good to offer and the whole mechanism looks weak.
  2. There are almost no clean Beginner->Advanced ladders inside one topic, so
     `generate`'s "order the picks as a progression" instruction has nothing to
     order.

So we generate to floor every category and to build three explicit ladders that
the demo personas climb.

These are clearly labelled `provider: "Reckon Originals"` and `synthetic: true`.
Do NOT disguise them as Coursera courses -- a judge who spots invented products
passed off as real data discounts the entire submission, and the honest label
costs nothing.

    uv run python scripts/generate_synthetic.py [--model openai/gpt-5-mini]
"""

import argparse
import json
import sys
from collections import Counter
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))
from app import config, mesh  # noqa: E402

FLOOR = 15          # minimum active courses per category

# Each ladder is a topic the demo personas walk up. Four rungs, so the agent can
# say "you've done this, do that next" with real prerequisite edges behind it.
LADDERS = [
    ("AI/ML", "building AI agents", [
        "Beginner", "Beginner", "Intermediate", "Intermediate", "Advanced"]),
    ("Data Engineering", "production data pipelines", [
        "Beginner", "Intermediate", "Intermediate", "Advanced"]),
    ("Finance", "financial modelling and valuation", [
        "Beginner", "Intermediate", "Intermediate", "Advanced"]),
]

SCHEMA = {
    "name": "generated_courses",
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
                    "required": ["title", "description", "skills", "level",
                                 "price_inr", "tagline", "rating"],
                    "properties": {
                        "title": {"type": "string"},
                        "description": {"type": "string"},
                        "skills": {"type": "string"},
                        "level": {"type": "string",
                                  "enum": ["Beginner", "Intermediate", "Advanced"]},
                        "price_inr": {"type": "integer"},
                        "tagline": {"type": "string"},
                        "rating": {"type": "number"},
                    },
                },
            }
        },
    },
}

SYSTEM = """You write catalogue entries for an online course marketplace.

Rules:
- description: 60-110 words, concrete about what the learner builds or can do
  afterwards. Name real tools and techniques. No marketing language, no
  "unlock", no "master", no exclamation marks, no "in today's world".
- skills: 4-7 comma-separated skills, specific and searchable (real tool and
  technique names, not "critical thinking").
- price_inr: 0 for genuinely introductory, otherwise 799-6999, varied and
  non-round. Higher for advanced.
- rating: between 4.0 and 4.9, one decimal, varied.
- tagline: one sentence, max 90 characters, the concrete outcome.
- Titles must be distinct from each other and from any titles listed as
  already existing."""


def load_catalog() -> list[dict]:
    if not config.CATALOG_JSON.exists():
        sys.exit("run scripts/enrich_catalog.py first")
    return json.loads(config.CATALOG_JSON.read_text(encoding="utf-8"))


def generate(model: str, prompt: str, n: int) -> list[dict]:
    result, meta = mesh.chat(
        [{"role": "system", "content": SYSTEM}, {"role": "user", "content": prompt}],
        schema=SCHEMA, model=model,
    )
    if result is None:
        print("  model returned prose, skipping this batch")
        return []
    print(f"  got {len(result['courses'])} (asked {n}) cache={meta['cache']}")
    return result["courses"]


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--model", default="openai/gpt-5-mini")
    args = ap.parse_args()

    catalog = load_catalog()
    next_id = max(c["id"] for c in catalog) + 1
    existing_titles = {c["title"].lower() for c in catalog}
    counts = Counter(c["category"] for c in catalog)
    new: list[dict] = []

    def add(raw: dict, category: str) -> dict | None:
        nonlocal next_id
        if raw["title"].lower() in existing_titles:
            return None
        existing_titles.add(raw["title"].lower())
        course = {
            "id": next_id,
            "title": raw["title"],
            "description": raw["description"],
            "category": category,
            "level": raw["level"],
            "price": float(raw["price_inr"]),
            "provider": "Reckon Originals",
            "skills": raw["skills"],
            "rating": round(float(raw["rating"]), 1),
            "tagline": raw["tagline"],
            "url": "",
            "prereq_ids": [],
            "synthetic": True,
        }
        next_id += 1
        new.append(course)
        return course

    # --- 1. the ladders ----------------------------------------------------
    for category, topic, levels in LADDERS:
        print(f"ladder: {topic} ({category}), {len(levels)} rungs")
        prompt = (
            f"Write {len(levels)} courses forming ONE coherent learning path in "
            f"{topic}, for the '{category}' category.\n"
            f"Rung levels in order: {levels}.\n"
            "Each rung must build directly on the previous one -- the later "
            "courses should assume the earlier courses' skills. Order matters: "
            "return them from first to last.\n"
            f"Avoid these existing titles: "
            f"{sorted(t for t in existing_titles if topic.split()[0].lower() in t)[:10]}"
        )
        rungs = [c for raw in generate(args.model, prompt, len(levels))
                 if (c := add(raw, category))]
        # Explicit prerequisite chain: each rung requires the one before it.
        for prev, cur in zip(rungs, rungs[1:]):
            cur["prereq_ids"] = [prev["id"]]
        counts[category] += len(rungs)

    # --- 2. floor the thin categories --------------------------------------
    for category in config.CATEGORIES:
        gap = FLOOR - counts.get(category, 0)
        if gap <= 0:
            continue
        print(f"topping up {category}: {counts.get(category, 0)} -> {FLOOR}")
        titles = [c["title"] for c in catalog if c["category"] == category][:8]
        prompt = (
            f"Write {gap} courses for the '{category}' category of a course "
            "marketplace. Spread them across Beginner, Intermediate and Advanced, "
            "and across clearly different sub-topics within the category.\n"
            f"Already in the catalogue, do not duplicate or closely paraphrase: {titles}"
        )
        made = [c for raw in generate(args.model, prompt, gap) if add(raw, category)]
        counts[category] += len(made)

    if not new:
        print("nothing generated")
        return

    merged = catalog + new
    config.CATALOG_JSON.write_text(
        json.dumps(merged, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nadded {len(new)} synthetic courses, catalogue now {len(merged)}")
    for cat in config.CATEGORIES:
        print(f"  {cat:22} {counts.get(cat, 0):3d}")


if __name__ == "__main__":
    main()
