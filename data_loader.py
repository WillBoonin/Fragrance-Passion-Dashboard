"""Load, clean and join the two Fragrantica CSVs into one tidy DataFrame."""

import re
from pathlib import Path

import pandas as pd
import streamlit as st

from seasons import infer_occasions, infer_seasons

DATA_DIR = Path(__file__).parent / "FragranceData"
CLEANED_CSV = DATA_DIR / "fra_cleaned.csv"
NAMES_CSV = DATA_DIR / "fra_names.csv"  # built from fra_perfumes.csv by prepare_data.py

GENDER_LABELS = {"women": "Feminine", "men": "Masculine", "unisex": "Unisex"}
GENDER_SUFFIX = re.compile(r"\s*for (women and men|women|men)\s*$")


def split_list(text) -> list[str]:
    """Split a comma-separated string into a list of stripped, non-empty items."""
    if not isinstance(text, str):
        return []
    return [item.strip() for item in text.split(",") if item.strip()]


def slug_to_title(slug: str) -> str:
    """'amber-oud-bleu-edition' -> 'Amber Oud Bleu Edition'."""
    return " ".join(word.capitalize() for word in str(slug).split("-"))


def _normalize(text: str) -> str:
    return re.sub(r"[^a-z0-9]", "", text.lower())


def split_name_and_brand(full_name, brand_slug: str) -> tuple[str | None, str | None]:
    """Split 'Eclipse 1996 Oriflamefor women' into ('Eclipse 1996', 'Oriflame').

    The perfumes file glues name + brand + gender together, so we strip the gender
    suffix and then find the trailing words that match the brand slug.
    Returns (None, None) if the brand can't be located.
    """
    if not isinstance(full_name, str):
        return None, None
    name = GENDER_SUFFIX.sub("", full_name).strip()
    words = name.split(" ")
    target = _normalize(brand_slug)
    for i in range(len(words) - 1, 0, -1):
        if _normalize(" ".join(words[i:])) == target:
            return " ".join(words[:i]), " ".join(words[i:])
    return None, None


def _read_cleaned() -> pd.DataFrame:
    df = pd.read_csv(CLEANED_CSV, sep=";", encoding="cp1252")
    df["rating"] = df["Rating Value"].str.replace(",", ".").astype(float)
    df["rating_count"] = df["Rating Count"].astype(int)
    df["year"] = df["Year"].astype("Int64")
    for layer in ["Top", "Middle", "Base"]:
        df[f"{layer.lower()}_notes"] = df[layer].map(split_list)
    df["notes"] = df.apply(
        lambda r: list(dict.fromkeys(r["top_notes"] + r["middle_notes"] + r["base_notes"])),
        axis=1,
    )
    accord_cols = [f"mainaccord{i}" for i in range(1, 6)]
    df["accords"] = df[accord_cols].apply(
        lambda r: [a.strip() for a in r if isinstance(a, str) and a.strip()], axis=1
    )
    df["perfumers"] = df[["Perfumer1", "Perfumer2"]].apply(
        lambda r: [p.title() for p in r if isinstance(p, str) and p.lower() != "unknown"],
        axis=1,
    )
    return df


def _read_display_names() -> pd.DataFrame:
    """Properly-cased 'name + brand + gender' strings keyed by lowercase url."""
    return pd.read_csv(NAMES_CSV)


@st.cache_data(show_spinner="Loading fragrance data…")
def load_fragrances() -> pd.DataFrame:
    """Return one row per fragrance with clean, typed columns ready for the app."""
    df = _read_cleaned()
    extra = _read_display_names()
    df["url_key"] = df["url"].str.lower().str.strip()
    df = df.merge(extra, on="url_key", how="left")

    split = [split_name_and_brand(n, b) for n, b in zip(df["Name"], df["Brand"])]
    df["name"] = [s[0] if s[0] else slug_to_title(slug) for s, slug in zip(split, df["Perfume"])]
    df["brand"] = [s[1] if s[1] else slug_to_title(slug) for s, slug in zip(split, df["Brand"])]
    df["fragrance_id"] = df["url"].str.extract(r"-(\d+)\.html$")[0]  # Fragrantica's numeric ID
    df["gender"] = df["Gender"].map(GENDER_LABELS)
    df["seasons"] = df["accords"].map(infer_seasons)
    df["occasions"] = df["accords"].map(infer_occasions)
    df["label"] = df["name"] + " – " + df["brand"] + df["year"].map(
        lambda y: f" ({y})" if pd.notna(y) else ""
    )
    # A handful of name+brand+year duplicates exist; keep labels unique for the search box.
    dup = df["label"].duplicated(keep=False)
    df.loc[dup, "label"] = df.loc[dup, "label"] + " #" + (df[dup].groupby("label").cumcount() + 1).astype(str)

    columns = [
        "fragrance_id", "url", "label", "name", "brand", "Country", "gender", "year", "rating", "rating_count",
        "top_notes", "middle_notes", "base_notes", "notes", "accords", "perfumers",
        "seasons", "occasions",
    ]
    return (
        df[columns]
        .rename(columns={"Country": "country"})
        .reset_index(drop=True)
    )
