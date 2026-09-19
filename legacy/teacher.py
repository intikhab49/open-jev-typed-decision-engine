"""Shared teacher plumbing: the prompt, the schema-constrained call, cost math."""
import json
from schema import AgentDecision

SYSTEM = (
    "You label states for an autonomous coding agent's routing layer.\n"
    "Given a terminal log, diff, or issue body, emit exactly one decision object.\n"
    "Definitions:\n"
    "  refactor          - the code is wrong or unclear; change it before running again\n"
    "  retry             - transient/environmental; rerun unchanged\n"
    "  escalate_to_human - ambiguous requirements, or a decision the agent may not make\n"
    "  deploy            - the state is green and the change is ready to ship\n"
    "Label what the state supports, not what you wish it said."
)

# output_config.format guarantees the first text block is valid JSON for this schema
OUTPUT_CONFIG = {
    "format": {
        "type": "json_schema",
        "schema": {
            "type": "object",
            "properties": {
                "is_terminal_error": {"type": "boolean"},
                "action_required": {
                    "type": "string",
                    "enum": ["refactor", "retry", "escalate_to_human", "deploy"],
                },
                "confidence_score": {"type": "integer", "minimum": 1, "maximum": 5},
            },
            "required": ["is_terminal_error", "action_required", "confidence_score"],
            "additionalProperties": False,
        },
    }
}

def user_msg(text: str) -> str:
    return f"[STATE]\n{text}"

def parse(response) -> AgentDecision:
    txt = next(b.text for b in response.content if b.type == "text")
    return AgentDecision(**json.loads(txt))

# $/1M tokens, first-party API. Batch API is 50% of these.
PRICES = {
    "claude-opus-5":   (5.00, 25.00),
    "claude-sonnet-5": (2.00, 10.00),
    "claude-haiku-4-5": (1.00, 5.00),
}

def cost(model: str, in_tok: int, out_tok: int, batch: bool = False) -> float:
    i, o = PRICES[model]
    c = in_tok / 1e6 * i + out_tok / 1e6 * o
    return c * 0.5 if batch else c
