"""The Jev interface, reimplemented.

A *state* (any JSON or text) plus N *typed questions*. One forward pass answers
all of them. Three primitive types, matching Jev:

  noul   - boolean          criteria: {"true": desc, "false": desc}
  choice - enum             criteria: {label: desc, ...}
  score  - ordered scale    criteria: [desc_0, desc_1, ...]   labels are "0".."n-1"

Everything downstream reads questions through `iter_labels`, so adding a new
primitive means touching this file only.
"""
from __future__ import annotations
import json
from typing import Any

TYPES = ["noul", "choice", "score"]
TYPE_ID = {t: i for i, t in enumerate(TYPES)}


# A bare noul carries no criteria at all - the instruction is the statement and
# the labels are implicit. 10% of typed-decisions is shaped this way, so treat it
# as part of the format rather than as malformed input.
BARE_NOUL = [("false", "The statement does not hold."),
             ("true", "The statement holds.")]


def iter_labels(q: dict) -> list[tuple[str, str]]:
    """Normalize a question's criteria to an ordered [(label, description)] list.

    `score` criteria arrive as an ordered list; its labels are the indices as
    strings, which is how the gold distributions key them.
    """
    crit = q.get("criteria")
    if crit is None:
        if q["type"] != "noul":
            raise ValueError(
                f"a {q['type']} question needs criteria - only noul may omit "
                f"them: {q.get('instructions', '')[:80]!r}")
        return list(BARE_NOUL)
    if isinstance(crit, list):                      # score
        return [(str(i), d) for i, d in enumerate(crit)]
    if q["type"] == "noul":                         # pin order: false, true
        return [(k, crit[k]) for k in ("false", "true") if k in crit]
    return sorted(crit.items())                     # choice - stable order


def state_to_text(state: Any) -> str:
    if isinstance(state, str):
        return state
    return json.dumps(state, indent=1, sort_keys=True, ensure_ascii=False)


def label_text(label: str, desc: str) -> str:
    return f"{label}: {desc}"


class Question(dict):
    """Convenience builder so you can define a schema in code.

    >>> Question.choice("What should we do?", {"retry": "...", "stop": "..."})
    """

    @staticmethod
    def noul(instructions: str, true_desc: str = None,
             false_desc: str = None) -> dict:
        """Descriptions are optional, matching the data: a bare noul carries no
        criteria at all and the instruction is the statement being judged."""
        q = {"type": "noul", "instructions": instructions}
        if true_desc is not None or false_desc is not None:
            q["criteria"] = {"true": true_desc or "The statement holds.",
                             "false": false_desc or "The statement does not hold."}
        return q

    @staticmethod
    def choice(instructions: str, criteria: dict[str, str]) -> dict:
        return {"type": "choice", "instructions": instructions, "criteria": criteria}

    @staticmethod
    def score(instructions: str, levels: list[str]) -> dict:
        return {"type": "score", "instructions": instructions, "criteria": levels}
