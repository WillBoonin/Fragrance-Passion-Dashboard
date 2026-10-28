"""Save and load the user's collection so it survives restarts.

The collection is always kept in the page URL (?collection=31861,33519) so each visitor
of a hosted app has their own, and can bookmark or share it. When running locally it is
also written to my_collection.json. That file is skipped on Streamlit Community Cloud,
where it would be shared by every visitor.
"""

import json
from pathlib import Path

import streamlit as st

COLLECTION_FILE = Path(__file__).parent / "my_collection.json"
QUERY_PARAM = "collection"
# Streamlit Community Cloud checks apps out under /mount/src.
RUNNING_ON_CLOUD = Path(__file__).resolve().as_posix().startswith("/mount/src")


def _load_from_file() -> list[str]:
    try:
        return json.loads(COLLECTION_FILE.read_text())["ids"]
    except (FileNotFoundError, json.JSONDecodeError, KeyError, TypeError):
        return []


def load_collection_ids() -> list[str]:
    """Fragrance IDs from the page URL, falling back to the local file."""
    from_url = st.query_params.get(QUERY_PARAM, "")
    if from_url:
        return [i for i in from_url.split(",") if i]
    return [] if RUNNING_ON_CLOUD else _load_from_file()


def save_collection_ids(ids: list[str]) -> None:
    if ids:
        st.query_params[QUERY_PARAM] = ",".join(ids)
    else:
        st.query_params.pop(QUERY_PARAM, None)
    if not RUNNING_ON_CLOUD:
        COLLECTION_FILE.write_text(json.dumps({"ids": ids}, indent=2))
