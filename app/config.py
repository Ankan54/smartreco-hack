"""Every tunable in one place. Defaults point at Mesh so a missing .env can never
silently route LLM traffic somewhere else (hackathon rule 5)."""

import os
from pathlib import Path

from dotenv import load_dotenv

ROOT = Path(__file__).resolve().parent.parent
load_dotenv(ROOT / ".env")


def _s(name: str, default: str = "") -> str:
    # .env values here are quoted; strip so "openai/x" doesn't become '"openai/x"'
    return os.getenv(name, default).strip().strip('"').strip("'")


# --- LLM -------------------------------------------------------------------
# Named MESH_* because the hackathon CI passes MESH_API_KEY as a secret.
# Dev against OpenAI = point these at api.openai.com. Never a code branch.
MESH_BASE_URL = _s("MESH_BASE_URL", "https://api.meshapi.ai/v1").rstrip("/")
MESH_API_KEY = _s("MESH_API_KEY")
CHAT_MODEL = _s("LLM_MODEL_ID", "openai/gpt-5-nano")
EMBED_MODEL = _s("EMBEDDING_MODEL_ID", "openai/text-embedding-3-small")
EMBED_DIM = int(_s("EMBED_DIM", "1536"))
# gpt-5-nano burns ~128 reasoning tokens on a trivial prompt; "low" halves it.
# "none" is rejected by this model. Measured, not guessed.
REASONING_EFFORT = _s("REASONING_EFFORT", "low")
# The generate node writes the only prose a human reads, so it gets the better
# model. grade/critic/reflect stay on the cheap one -- nobody reads their output.
GENERATE_MODEL = _s("GENERATE_MODEL", "openai/gpt-5-mini")

LANGSMITH_API_KEY = _s("LANGSMITH_API_KEY")
LANGSMITH_PROJECT = _s("LANGSMITH_PROJECT", "reckon")

# --- storage ---------------------------------------------------------------
DATA_DIR = ROOT / "data"
DB_PATH = Path(_s("DB_PATH", str(DATA_DIR / "app.db")))
CHROMA_PATH = Path(_s("CHROMA_PATH", str(DATA_DIR / "chroma")))
CATALOG_JSON = DATA_DIR / "catalog.json"
COLLECTION = "products"

SECRET_KEY = _s("SECRET_KEY", "dev-only-not-a-secret-change-in-prod")

# --- intent engine (§8) ----------------------------------------------------
HALF_LIFE_MIN = 240.0          # 4h: a morning's browsing still shapes the afternoon
DRIFT_THRESHOLD = 0.15         # cosine distance that counts as "changed course"
MIN_EVENTS_FOR_DRIFT = 3
VOLUME_TRIGGER = 25            # fire regardless after this many events
STALE_HOURS = 6
MIN_RECO_INTERVAL_SEC = 90     # hard rate limit, overrides every other trigger

EVENT_WEIGHTS = {
    "view": 1.0,
    "click": 1.5,
    "scroll": 0.5,
    "search": 2.5,     # explicit intent
    "dwell": 2.0,      # only emitted at >= DWELL_MIN_SEC
    "enroll": 4.0,
    "bounce": -0.6,    # push AWAY: an only-additive vector is a popularity vector
}
DWELL_MIN_SEC = 20.0
BOUNCE_MAX_SEC = 3.0

# --- retrieval (§8.5) ------------------------------------------------------
PROBE_TOP_K = 12
CANDIDATE_LIMIT = 15
RRF_K = 60
PROBE_WEIGHTS = {"bearing": 1.0, "focus": 0.8, "words": 0.9, "literal": 0.9}
CLAIM_PROBE_WEIGHT = 0.7       # x strength, for +interest claims
CLAIM_PENALTY_WEIGHT = 0.5     # x strength, for -interest claims

CATEGORIES = [
    "AI/ML", "Data Engineering", "Cloud", "Cybersecurity",
    "Product", "Design", "Finance", "Marketing",
    "Leadership", "Healthcare", "Language", "Personal Development",
]

DATA_DIR.mkdir(exist_ok=True)
