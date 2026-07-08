"""
core/reconciler.py

Checks the detected diff against the stated purpose of the revision.
Two backends, chosen in config.py:
  - "ollama": semantic reconciliation via a local text LLM (recommended).
  - "rules":  keyword/fuzzy text matching only, no LLM.
"""

import json
import re
from dataclasses import asdict
from difflib import SequenceMatcher

import config
from core.ollama_client import chat_json

RECONCILE_PROMPT = """You are auditing an engineering drawing revision for \
an airbag manufacturer. You are given:

1. PURPOSE: the stated reasons for this revision.
2. DETECTED CHANGES: everything that was actually found to differ between \
the previous revision and this one. Some entries are marked
"low_confidence": true -- these are rows OCR could not fully read (common
on struck-through rows where a red line corrupts the text); treat them as
"needs manual check" rather than confidently met/not-met either way.

For each purpose item, decide "met", "partially_met", or "not_met", citing \
which detected change(s) support that verdict. List every detected change \
that does NOT correspond to any stated purpose item under \
"unexpected_changes". Be strict -- a purpose item is only "met" if a \
change was actually found that accomplishes it.

Return ONLY valid JSON in this schema:

{{
  "purpose_verdicts": [
    {{"purpose": "...", "verdict": "met|partially_met|not_met",
     "evidence": ["..."], "explanation": "..."}}
  ],
  "unexpected_changes": [
    {{"description": "...", "risk_note": "..."}}
  ]
}}

PURPOSE:
{purpose_list}

DETECTED CHANGES:
{changes_json}
"""


def _serialize_changes(diff: dict) -> str:
    out = {
        "bom_changes": [asdict(c) for c in diff["bom_changes"] if c.change_type != "unchanged"],
        "annotation_changes": [asdict(c) for c in diff["annotation_changes"]],
    }
    return json.dumps(out, indent=2, default=str)


def _reconcile_ollama(diff: dict, purpose_items: list) -> dict:
    prompt = RECONCILE_PROMPT.format(
        purpose_list="\n".join(f"- {p}" for p in purpose_items),
        changes_json=_serialize_changes(diff),
    )
    return chat_json(prompt, model=config.OLLAMA_TEXT_MODEL)


# --- rule-based fallback (no LLM at all) -----------------------------------

STOPWORDS = {"the", "a", "an", "to", "of", "on", "in", "for", "and", "change",
             "changed", "update", "updated", "add", "added", "remove", "removed"}


def _keywords(text: str) -> set:
    words = re.findall(r"[a-zA-Z0-9]+", text.lower())
    return {w for w in words if w not in STOPWORDS and len(w) > 1}


def _describe_change(c) -> str:
    prefix = "[NEEDS MANUAL CHECK -- low OCR confidence] " if getattr(c, "low_confidence", False) else ""
    if c.component_type == "bom_row":
        if c.change_type == "removed":
            return f"{prefix}Removed FN{c.key} ({c.old_value.get('title', '')})"
        if c.change_type == "added":
            return f"{prefix}Added FN{c.key} ({c.new_value.get('title', '')})"
        if c.change_type == "modified":
            parts = [f"{fc.field} changed from '{fc.old}' to '{fc.new}'" for fc in c.changes]
            return f"{prefix}FN{c.key} ({c.old_value.get('title', '')}): " + "; ".join(parts)
    else:
        if c.change_type == "removed":
            return f"Removed {c.key}: '{c.old_value.get('text', '')}'"
        if c.change_type == "added":
            return f"Added {c.key}: '{c.new_value.get('text', '')}'"
        if c.change_type == "modified":
            parts = [f"{fc.field} changed from '{fc.old}' to '{fc.new}'" for fc in c.changes]
            return f"{c.key}: " + "; ".join(parts)
    return f"{c.key}: {c.change_type}"


def _reconcile_rules(diff: dict, purpose_items: list, threshold: float = None) -> dict:
    threshold = threshold if threshold is not None else config.RULES_MATCH_THRESHOLD
    all_changes = [c for c in diff["bom_changes"] if c.change_type != "unchanged"] + diff["annotation_changes"]
    descriptions = [(c, _describe_change(c)) for c in all_changes]

    purpose_verdicts, matched_idx = [], set()
    for purpose in purpose_items:
        fn_refs = set(re.findall(r"fn\s?(\d+)", purpose.lower()))
        scored = []
        for idx, (c, desc) in enumerate(descriptions):
            text_score = SequenceMatcher(None, purpose.lower(), desc.lower()).ratio()
            overlap = len(_keywords(purpose) & _keywords(desc)) / max(len(_keywords(purpose)), 1)
            score = max(text_score, overlap)
            if fn_refs and c.component_type == "bom_row" and c.key in fn_refs:
                score = max(score, 0.9)
            if score >= threshold:
                scored.append((score, idx, desc))
        scored.sort(reverse=True)
        if scored:
            for _, idx, _ in scored[:3]:
                matched_idx.add(idx)
            purpose_verdicts.append({"purpose": purpose, "verdict": "met",
                                      "evidence": [d for _, _, d in scored[:3]],
                                      "explanation": "Matched by keyword/text similarity (heuristic, not semantic)."})
        else:
            purpose_verdicts.append({"purpose": purpose, "verdict": "not_met",
                                      "evidence": [], "explanation": "No change scored above threshold -- check manually."})

    unexpected = [{"description": desc, "risk_note": "Did not match any stated purpose item."}
                  for idx, (c, desc) in enumerate(descriptions) if idx not in matched_idx]
    return {"purpose_verdicts": purpose_verdicts, "unexpected_changes": unexpected}


def reconcile(diff: dict, purpose_items: list) -> dict:
    if config.RECONCILE_BACKEND == "ollama":
        return _reconcile_ollama(diff, purpose_items)
    return _reconcile_rules(diff, purpose_items)
