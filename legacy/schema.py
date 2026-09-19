"""The contract. Everything else in this repo is derived from this file.

Change the schema -> regenerate labels -> retrain. Nothing else.
"""
from typing import Literal
from pydantic import BaseModel, Field

ACTIONS = ["refactor", "retry", "escalate_to_human", "deploy"]


class AgentDecision(BaseModel):
    is_terminal_error: bool = Field(
        description="True if the state contains a fatal execution error "
                    "(crash, compile failure, non-zero exit)."
    )
    action_required: Literal["refactor", "retry", "escalate_to_human", "deploy"] = Field(
        description="The single routing decision an autonomous coding agent should take next."
    )
    confidence_score: int = Field(
        ge=1, le=5,
        description="1-5. How unambiguous is action_required given only this state? "
                    "1 = could plausibly be any action, 5 = only one sane choice."
    )


# Head layout the model builds against. Keep in sync with the class above.
HEADS = {
    "is_terminal_error": {"kind": "binary", "n": 1},
    "action_required": {"kind": "multiclass", "n": len(ACTIONS)},
    "confidence_score": {"kind": "multiclass", "n": 5},
}
