"""JSON schemas for the agent's structured outputs.

Mesh degrades SILENTLY on a model that ignores response_format -- you get prose
with finish_reason "stop" and no error -- so every one of these is paired with a
caller that handles `None` rather than assuming success. Verified working on
openai/gpt-5-nano with strict:true, despite the /models catalog reporting
supports_structured_output=False for it.
"""

from app import config

LEVELS = ["Beginner", "Intermediate", "Advanced"]


# --- reflect: rewrite what we believe about the user ------------------------
DOSSIER = {
    "name": "dossier",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["prose", "claims"],
        "properties": {
            "prose": {"type": "string"},
            "claims": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["id", "polarity", "kind", "text", "strength", "evidence"],
                    "properties": {
                        "id": {"type": "string"},
                        "polarity": {"type": "string", "enum": ["+", "-"]},
                        "kind": {"type": "string", "enum": ["interest", "constraint"]},
                        # Written to look like a course description, not like
                        # commentary about the user -- a probe has to live in
                        # the same embedding space as the documents.
                        "text": {"type": "string"},
                        "strength": {"type": "number"},
                        "evidence": {"type": "string"},
                    },
                },
            },
        },
    },
}


# --- grade: is this candidate set good enough to write from? ---------------
GRADE = {
    "name": "retrieval_grade",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["sufficient", "reason", "pseudo_query", "filters"],
        "properties": {
            "sufficient": {"type": "boolean"},
            "reason": {"type": "string"},
            # A hypothetical course description to re-probe with when the
            # candidates miss -- HyDE, essentially.
            "pseudo_query": {"type": "string"},
            "filters": {
                "type": "object",
                "additionalProperties": False,
                "required": ["category", "level", "max_price"],
                "properties": {
                    "category": {"type": "string", "enum": [""] + config.CATEGORIES},
                    "level": {"type": "string", "enum": [""] + LEVELS},
                    "max_price": {"type": "number"},
                },
            },
        },
    },
}


# --- generate: the recommendation the user actually reads ------------------
RECOMMENDATION = {
    "name": "recommendation",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["headline", "narrative", "items"],
        "properties": {
            "headline": {"type": "string"},
            "narrative": {"type": "string"},
            "items": {
                "type": "array",
                "items": {
                    "type": "object",
                    "additionalProperties": False,
                    "required": ["product_id", "reason", "confidence"],
                    "properties": {
                        "product_id": {"type": "integer"},
                        "reason": {"type": "string"},
                        "confidence": {"type": "number"},
                    },
                },
            },
        },
    },
}


# --- critic: an adversarial read of the draft ------------------------------
CRITIQUE = {
    "name": "critique",
    "schema": {
        "type": "object",
        "additionalProperties": False,
        "required": ["verdict", "faults"],
        "properties": {
            "verdict": {"type": "string", "enum": ["pass", "revise"]},
            "faults": {"type": "array", "items": {"type": "string"}},
        },
    },
}
