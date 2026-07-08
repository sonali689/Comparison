"""
app.py — Drawing Revision Auditor

Upload two revisions of the same drawing, state the purpose of the
revision, get an audit of whether that purpose was actually met, plus
every other change found. Everything runs locally -- no cloud API calls
anywhere.
"""

import os
import tempfile
import streamlit as st

import config
from pipeline import run_purpose_check

st.set_page_config(page_title="Drawing Revision Auditor", layout="wide")
st.title("Drawing Revision Auditor")
st.caption(
    "Upload the old and new revision of a drawing, state the purpose of the "
    "revision, and audit whether it was actually met -- fully local, no cloud API calls."
)

if config.EXTRACTION_BACKEND == "ollama" or config.RECONCILE_BACKEND == "ollama":
    from core.ollama_client import is_available as ollama_available
    if not ollama_available():
        st.warning(
            f"Can't reach Ollama at {config.OLLAMA_HOST}. Start it with `ollama serve`, "
            f"pull `{config.OLLAMA_VISION_MODEL}` / `{config.OLLAMA_TEXT_MODEL}`, or switch "
            "EXTRACTION_BACKEND/RECONCILE_BACKEND to 'ocr'/'rules' in config.py."
        )

st.sidebar.markdown(f"**Extraction backend:** `{config.EXTRACTION_BACKEND}`")
st.sidebar.markdown(f"**Reconciliation backend:** `{config.RECONCILE_BACKEND}`")
st.sidebar.caption("Change these in config.py")

col1, col2 = st.columns(2)
with col1:
    old_file = st.file_uploader("Previous revision (PDF)", type="pdf", key="old_rev")
with col2:
    new_file = st.file_uploader("New revision (PDF)", type="pdf", key="new_rev")

drawing_number = st.text_input(
    "Drawing number (optional, only used by the 'ocr' backend to load a saved "
    "grid calibration -- see calibrations/)",
    placeholder="X670001400A",
)

purpose_text = st.text_area(
    "Stated purpose of this revision (one item per line)",
    placeholder="Remove FN22\nChange assembly sewing steps\nChange seam table",
    height=120,
)

ready = bool(old_file and new_file and purpose_text.strip())
if st.button("Run audit", type="primary", disabled=not ready):
    purpose_items = [p.strip() for p in purpose_text.splitlines() if p.strip()]

    with tempfile.TemporaryDirectory() as tmp:
        old_path = os.path.join(tmp, "old.pdf")
        new_path = os.path.join(tmp, "new.pdf")
        with open(old_path, "wb") as f:
            f.write(old_file.read())
        with open(new_path, "wb") as f:
            f.write(new_file.read())

        with st.spinner("Reading both drawings and comparing every component..."):
            result = run_purpose_check(old_path, new_path, purpose_items,
                                        drawing_number=drawing_number.strip() or None)

    report, diff = result["report"], result["diff"]

    st.subheader("Was the purpose met?")
    for v in report["purpose_verdicts"]:
        icon = {"met": "✅", "partially_met": "⚠️", "not_met": "❌"}.get(v["verdict"], "❓")
        with st.expander(f"{icon} {v['purpose']} — {v['verdict'].replace('_', ' ').upper()}"):
            st.write(v["explanation"])
            for e in v.get("evidence", []):
                st.markdown(f"- {e}")

    st.subheader("⚠️ Unexpected changes")
    if report["unexpected_changes"]:
        for u in report["unexpected_changes"]:
            st.markdown(f"- **{u['description']}** — {u['risk_note']}")
    else:
        st.success("None found.")

    st.subheader("All BOM / FN changes")
    bom_changes = [c for c in diff["bom_changes"] if c.change_type != "unchanged"]
    if bom_changes:
        rows = []
        for c in bom_changes:
            flag = " ⚠️ low OCR confidence" if c.low_confidence else ""
            if c.change_type == "modified":
                for fc in c.changes:
                    rows.append({"FN": c.key + flag, "Change": c.change_type, "Field": fc.field,
                                 "Old": fc.old, "New": fc.new})
            else:
                rows.append({"FN": c.key + flag, "Change": c.change_type, "Field": "-", "Old": "-", "New": "-"})
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No BOM/FN table changes detected.")

    st.subheader("All other annotation / dimension / text changes")
    ann_changes = diff["annotation_changes"]
    if ann_changes:
        rows = []
        for c in ann_changes:
            if c.change_type == "modified":
                for fc in c.changes:
                    rows.append({"Ref": c.key, "Change": c.change_type, "Field": fc.field, "Old": fc.old, "New": fc.new})
            else:
                rows.append({"Ref": c.key, "Change": c.change_type, "Field": "text",
                              "Old": (c.old_value or {}).get("text", "-"),
                              "New": (c.new_value or {}).get("text", "-")})
        st.dataframe(rows, use_container_width=True)
    else:
        st.info("No annotation/text changes detected.")
