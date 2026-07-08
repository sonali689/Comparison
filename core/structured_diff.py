"""
core/structured_diff.py

Plain dictionary/key-based diffing of two extracted drawings, shared by
all extraction backends. No image alignment and no pixel comparison --
every component is matched by its own ID (FN number for BOM rows,
kind+ref_id for annotations), so it doesn't matter where the component
sits on the page in either revision.

Note on `unread_row_p{page}_r{idx}` keys (from core/ocr_extractor.py):
these are rows OCR couldn't confidently read an FN number for (very
common on struck-through rows, since the red line corrupts the digits
too). Matching those keys across revisions is a position-based fallback,
scoped ONLY to cells that couldn't be read at all -- it never applies to
anything with a real, confidently-read FN. See ocr_extractor.py's module
docstring for the reasoning.
"""

from dataclasses import dataclass
from difflib import SequenceMatcher

BOM_FIELDS = ["qty", "title", "material", "material_spec", "usage", "struck_through"]


@dataclass
class FieldChange:
    field: str
    old: str
    new: str


@dataclass
class ComponentChange:
    component_type: str    # "bom_row" or "annotation"
    key: str                # FN number, or "<kind>:<ref_id>"
    change_type: str        # "added", "removed", "modified", "unchanged"
    changes: list            # list[FieldChange], empty for added/removed/unchanged
    old_value: dict = None
    new_value: dict = None
    low_confidence: bool = False


def _text_similar(a: str, b: str, threshold: float = 0.98) -> bool:
    return SequenceMatcher(None, (a or "").strip().lower(), (b or "").strip().lower()).ratio() >= threshold


def _sort_key(fn: str):
    return int(fn) if fn.isdigit() else 10_000  # unread_row_* keys sort last


def diff_bom(old_rows: dict, new_rows: dict) -> list:
    changes = []
    all_fns = set(old_rows) | set(new_rows)
    for fn in sorted(all_fns, key=_sort_key):
        old, new = old_rows.get(fn), new_rows.get(fn)
        low_conf = (old or {}).get("fn_confidence") == "low" or (new or {}).get("fn_confidence") == "low"

        if old and not new:
            changes.append(ComponentChange("bom_row", fn, "removed", [], old_value=old, low_confidence=low_conf))
        elif new and not old:
            changes.append(ComponentChange("bom_row", fn, "added", [], new_value=new, low_confidence=low_conf))
        else:
            field_changes = []
            for f in BOM_FIELDS:
                ov, nv = old.get(f), new.get(f)
                if str(ov).strip() != str(nv).strip():
                    field_changes.append(FieldChange(f, ov, nv))
            if field_changes:
                changes.append(ComponentChange("bom_row", fn, "modified", field_changes,
                                                old_value=old, new_value=new, low_confidence=low_conf))
            else:
                changes.append(ComponentChange("bom_row", fn, "unchanged", [], old_value=old,
                                                new_value=new, low_confidence=low_conf))
    return changes


def _annotation_key(ann: dict) -> str:
    return f"{ann.get('kind', 'other')}:{ann.get('ref_id', 'none')}"


def diff_annotations(old_anns: list, new_anns: list) -> list:
    changes = []
    old_by_key, new_by_key = {}, {}
    old_unlabelled, new_unlabelled = [], []

    for a in old_anns:
        (old_unlabelled if a.get("ref_id", "none") == "none" else old_by_key.setdefault(_annotation_key(a), a))
    for a in new_anns:
        (new_unlabelled if a.get("ref_id", "none") == "none" else new_by_key.setdefault(_annotation_key(a), a))

    all_keys = set(old_by_key) | set(new_by_key)
    for key in sorted(all_keys):
        old, new = old_by_key.get(key), new_by_key.get(key)
        if old and not new:
            changes.append(ComponentChange("annotation", key, "removed", [], old_value=old))
        elif new and not old:
            changes.append(ComponentChange("annotation", key, "added", [], new_value=new))
        elif old["text"].strip() != new["text"].strip() or old.get("color") != new.get("color"):
            fc = [FieldChange("text", old["text"], new["text"])]
            if old.get("color") != new.get("color"):
                fc.append(FieldChange("color", old.get("color"), new.get("color")))
            changes.append(ComponentChange("annotation", key, "modified", fc, old_value=old, new_value=new))

    matched_new = set()
    for o in old_unlabelled:
        best, best_score, best_idx = None, 0.0, None
        for j, n in enumerate(new_unlabelled):
            if j in matched_new or n.get("kind") != o.get("kind"):
                continue
            score = SequenceMatcher(None, o["text"].lower(), n["text"].lower()).ratio()
            if score > best_score:
                best, best_score, best_idx = n, score, j
        if best and best_score >= 0.5:
            matched_new.add(best_idx)
            if not _text_similar(o["text"], best["text"]):
                changes.append(ComponentChange(
                    "annotation", f"{o.get('kind')}:unlabelled", "modified",
                    [FieldChange("text", o["text"], best["text"])],
                    old_value=o, new_value=best))
        else:
            changes.append(ComponentChange("annotation", f"{o.get('kind')}:unlabelled", "removed", [], old_value=o))
    for j, n in enumerate(new_unlabelled):
        if j not in matched_new:
            changes.append(ComponentChange("annotation", f"{n.get('kind')}:unlabelled", "added", [], new_value=n))

    return changes


def diff_drawings(old_extract: dict, new_extract: dict) -> dict:
    return {
        "bom_changes": diff_bom(old_extract["bom_rows"], new_extract["bom_rows"]),
        "annotation_changes": diff_annotations(old_extract["annotations"], new_extract["annotations"]),
    }
