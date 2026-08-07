"""Step 1 of 4: real Coursera courses -> data/catalog_raw.json

Source: huggingface.co/datasets/azrai99/coursera-course-dataset (Apache-2.0,
6,645 rows, direct CSV, no auth). Run once; the output feeds enrich_catalog.py.

Two things about this data that cost time to find:
  * It decodes as utf-8, but some rows contain a literal U+FFFD where an
    apostrophe or dash used to be -- the mojibake is baked into the source, not
    a decoding error. Measured: every such row also fails the description-length
    filter, so is_corrupt() currently drops 0. It stays as a cheap guard, since
    with 6,645 rows available and ~260 needed we can afford to discard rather
    than repair if the source ever changes.
  * `Skills` is a stringified Python list, frequently '[]', and frequently with
    every entry duplicated.

    uv run python scripts/download_catalog.py
"""

import ast
import csv
import io
import json
import random
import re
import sys
from collections import defaultdict

import httpx

sys.path.insert(0, str(__import__("pathlib").Path(__file__).resolve().parent.parent))
from app import config  # noqa: E402

URL = ("https://huggingface.co/datasets/azrai99/coursera-course-dataset/"
       "resolve/main/coursera_course_2024.csv")
TARGET_PER_CATEGORY = 22
MIN_DESC = 250

# Keyword buckets exist ONLY to spread the sample across topics before we pay an
# LLM anything. enrich_catalog.py assigns the authoritative category; if it
# disagrees with a hint here, that is fine and expected.
HINTS: dict[str, list[str]] = {
    "AI/ML": ["machine learning", "deep learning", "neural", "artificial intelligence",
              "tensorflow", "pytorch", "nlp", "computer vision", "generative ai", "llm",
              "data science", "reinforcement"],
    "Data Engineering": ["sql", "database", "etl", "data warehouse", "spark", "hadoop",
                         "big data", "data pipeline", "airflow", "bigquery", "snowflake"],
    "Cloud": ["aws", "azure", "google cloud", "kubernetes", "docker", "devops",
              "terraform", "serverless", "cloud computing", "microservices"],
    "Cybersecurity": ["security", "cyber", "penetration", "cryptograph", "forensic",
                      "malware", "network defense", "ethical hacking", "risk management"],
    "Product": ["product management", "agile", "scrum", "roadmap", "user research",
                "product owner", "stakeholder", "requirements"],
    "Design": ["ux", "ui design", "user experience", "graphic design", "figma",
               "typography", "visual design", "prototyp", "interaction design"],
    "Finance": ["finance", "accounting", "investment", "valuation", "financial",
                "trading", "economics", "portfolio", "corporate finance", "fintech"],
    "Marketing": ["marketing", "seo", "advertising", "brand", "social media",
                  "content strategy", "google analytics", "customer acquisition"],
    "Leadership": ["leadership", "management", "negotiation", "team", "strategy",
                   "organizational", "coaching", "decision making", "influence"],
    "Healthcare": ["health", "medicine", "clinical", "patient", "nursing",
                   "epidemiolog", "public health", "anatomy", "pharma"],
    "Language": ["english", "spanish", "chinese", "language", "grammar", "writing skills",
                 "communication skills", "toefl", "korean", "french"],
    "Personal Development": ["personal", "productivity", "mindfulness", "career",
                             "learning how to learn", "creativity", "wellbeing",
                             "resilience", "habits", "public speaking"],
}


def clean(s: str) -> str:
    return re.sub(r"\s+", " ", (s or "").strip())


def is_corrupt(*parts: str) -> bool:
    return any("�" in (p or "") for p in parts)


def parse_skills(raw: str) -> str:
    try:
        items = ast.literal_eval(raw or "[]")
    except (ValueError, SyntaxError):
        return ""
    seen, out = set(), []
    for s in items:                       # the source duplicates every entry
        s = clean(str(s))
        if s and s.lower() not in seen:
            seen.add(s.lower())
            out.append(s)
    return ", ".join(out[:10])


def parse_level(raw: str) -> str | None:
    lv = clean(raw).replace(" level", "")
    return lv if lv in ("Beginner", "Intermediate", "Advanced") else None


def parse_num(raw: str, cast=float):
    try:
        return cast(clean(raw).replace(",", ""))
    except (ValueError, TypeError):
        return None


def hint_for(text: str) -> str | None:
    text = text.lower()
    best, score = None, 0
    for cat, words in HINTS.items():
        n = sum(1 for w in words if w in text)
        if n > score:
            best, score = cat, n
    return best


def main() -> None:
    print(f"downloading {URL}")
    raw = httpx.get(URL, follow_redirects=True, timeout=180).content
    print(f"  {len(raw) / 1e6:.1f} MB")

    rows = list(csv.DictReader(io.StringIO(raw.decode("utf-8"))))
    print(f"  {len(rows)} rows")

    buckets: dict[str, list[dict]] = defaultdict(list)
    seen_titles: set[str] = set()
    dropped_corrupt = 0

    for r in rows:
        title, desc = clean(r.get("title")), clean(r.get("Description"))
        skills = parse_skills(r.get("Skills"))
        if not title or len(desc) < MIN_DESC or not skills:
            continue
        if is_corrupt(title, desc, skills):
            dropped_corrupt += 1
            continue
        key = title.lower()
        if key in seen_titles:
            continue
        seen_titles.add(key)

        hint = hint_for(f"{title} {skills}")
        if not hint:
            continue
        buckets[hint].append({
            "title": title,
            "description": desc[:1200],
            "skills": skills,
            "level": parse_level(r.get("Level")),
            "rating": parse_num(r.get("rating")),
            "enrolled": parse_num(r.get("enrolled"), int),
            "provider": clean(r.get("Organization")) or "Coursera",
            "url": clean(r.get("URL")),
            "hint": hint,
        })

    print(f"  dropped {dropped_corrupt} corrupted rows")

    rng = random.Random(20260806)          # deterministic: reruns give the same catalog
    out: list[dict] = []
    for cat in config.CATEGORIES:
        pool = buckets.get(cat, [])
        # Prefer well-reviewed courses, then sample for variety within that.
        pool.sort(key=lambda p: (p["rating"] or 0, p["enrolled"] or 0), reverse=True)
        top = pool[: TARGET_PER_CATEGORY * 3]
        rng.shuffle(top)
        picked = top[:TARGET_PER_CATEGORY]
        out.extend(picked)
        flag = "" if len(picked) >= TARGET_PER_CATEGORY else "  <- thin, synthetic will top up"
        print(f"  {cat:22} {len(picked):3d} of {len(pool):4d} available{flag}")

    for i, p in enumerate(out, start=1):
        p["id"] = i

    config.DATA_DIR.mkdir(exist_ok=True)
    path = config.DATA_DIR / "catalog_raw.json"
    path.write_text(json.dumps(out, indent=1, ensure_ascii=False), encoding="utf-8")
    print(f"\nwrote {len(out)} courses -> {path}")


if __name__ == "__main__":
    main()
