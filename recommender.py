"""Recommendation logic: feature vectors, scoring, filtering and explanations."""

from dataclasses import dataclass, field

import numpy as np
import pandas as pd
from scipy.sparse import hstack
from sklearn.feature_extraction import DictVectorizer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.preprocessing import normalize

from seasons import ACCORD_RANK_WEIGHTS

# Relative weight of accords vs notes in the combined vector.
NOTE_WEIGHT = 1.0
ACCORD_WEIGHT = 1.2

# Final score blend (re-normalised when a component is unavailable).
SIMILARITY_WEIGHT = 0.6
PREFERENCE_WEIGHT = 0.2
RATING_WEIGHT = 0.2

# Bayesian rating: shrink ratings with few votes toward the dataset mean.
RATING_PRIOR_VOTES = 200


@dataclass
class Preferences:
    mode: str = "similar"  # "similar" or "different"
    preferred_notes: list[str] = field(default_factory=list)
    avoid_notes: list[str] = field(default_factory=list)
    preferred_accords: list[str] = field(default_factory=list)
    seasons: list[str] = field(default_factory=list)
    occasions: list[str] = field(default_factory=list)
    genders: list[str] = field(default_factory=list)  # empty = no preference
    min_rating: float = 0.0
    min_rating_count: int = 0


def build_feature_matrix(df: pd.DataFrame):
    """Rows = fragrances. TF-IDF over notes (rare notes count more) + rank-weighted accords.

    Rows are L2-normalised so a dot product is cosine similarity.
    """
    note_vectorizer = TfidfVectorizer(analyzer=lambda notes: notes, sublinear_tf=True)
    note_matrix = note_vectorizer.fit_transform(df["notes"])

    accord_dicts = [
        {accord: w for accord, w in zip(accords, ACCORD_RANK_WEIGHTS)} for accords in df["accords"]
    ]
    accord_matrix = normalize(DictVectorizer().fit_transform(accord_dicts))

    combined = hstack([note_matrix * NOTE_WEIGHT, accord_matrix * ACCORD_WEIGHT]).tocsr()
    return normalize(combined), note_vectorizer


def bayesian_rating_score(df: pd.DataFrame) -> pd.Series:
    """Rating adjusted for vote count, expressed as a 0–1 percentile."""
    mean = (df["rating"] * df["rating_count"]).sum() / df["rating_count"].sum()
    v = df["rating_count"]
    adjusted = (v * df["rating"] + RATING_PRIOR_VOTES * mean) / (v + RATING_PRIOR_VOTES)
    return adjusted.rank(pct=True)


def preference_score(df: pd.DataFrame, prefs: Preferences) -> pd.Series:
    """Share of the user's preferred notes and accords each fragrance contains."""
    wanted_notes, wanted_accords = set(prefs.preferred_notes), set(prefs.preferred_accords)
    total = len(wanted_notes) + len(wanted_accords)
    if total == 0:
        return pd.Series(0.0, index=df.index)
    hits = df["notes"].map(lambda n: len(wanted_notes.intersection(n))) + df["accords"].map(
        lambda a: len(wanted_accords.intersection(a))
    )
    return hits / total


def apply_filters(df: pd.DataFrame, prefs: Preferences, exclude_idx) -> pd.Series:
    """Boolean mask of fragrances that pass every hard filter."""
    mask = ~df.index.isin(exclude_idx)
    mask &= df["rating"] >= prefs.min_rating
    mask &= df["rating_count"] >= prefs.min_rating_count
    if prefs.genders:
        mask &= df["gender"].isin(prefs.genders)
    if prefs.avoid_notes:
        avoid = set(prefs.avoid_notes)
        mask &= ~df["notes"].map(lambda n: bool(avoid.intersection(n)))
    if prefs.seasons:
        wanted = set(prefs.seasons)
        mask &= df["seasons"].map(lambda s: bool(wanted.intersection(s)))
    if prefs.occasions:
        wanted = set(prefs.occasions)
        mask &= df["occasions"].map(lambda o: bool(wanted.intersection(o)))
    return pd.Series(mask, index=df.index)


def explain(row: pd.Series, match: pd.Series | None, prefs: Preferences, note_idf: dict) -> str:
    """Build a one-line 'why this was recommended'."""
    parts = []
    if match is not None and prefs.mode == "similar":
        shared_notes = sorted(set(row["notes"]) & set(match["notes"]), key=lambda n: -note_idf.get(n, 0))
        shared_accords = [a for a in row["accords"] if a in match["accords"]]
        shared = shared_notes[:3] or shared_accords[:3]
        if shared:
            parts.append(f"Shares {', '.join(shared)} with {match['name']}")
        else:
            parts.append(f"Closest to {match['name']}")
    elif match is not None:
        new_accords = [a for a in row["accords"] if a not in match["accords"]][:3]
        if new_accords:
            parts.append(f"A change of pace: leans {', '.join(new_accords)}")
    liked = [n for n in prefs.preferred_notes if n in row["notes"]]
    liked += [a for a in prefs.preferred_accords if a in row["accords"]]
    if liked:
        parts.append(f"has your picks: {', '.join(liked[:4])}")
    parts.append(f"rated {row['rating']:.2f} by {row['rating_count']:,} people")
    text = "; ".join(parts)
    return text[0].upper() + text[1:]


def recommend(
    df: pd.DataFrame,
    features,
    note_idf: dict,
    collection_idx: list[int],
    prefs: Preferences,
    top_n: int = 10,
) -> tuple[pd.DataFrame, list[str]]:
    """Return the top-N fragrances (with a 'score' column) and an explanation for each."""
    components, weights = {}, {}

    best_match = pd.Series(-1, index=df.index)
    if collection_idx:
        sims = (features @ features[collection_idx].T).toarray()  # (n_fragrances, n_owned)
        best_sim = sims.max(axis=1)
        best_match = pd.Series(np.asarray(collection_idx)[sims.argmax(axis=1)], index=df.index)
        components["similarity"] = best_sim if prefs.mode == "similar" else 1 - best_sim
        weights["similarity"] = SIMILARITY_WEIGHT

    if prefs.preferred_notes or prefs.preferred_accords:
        components["preference"] = preference_score(df, prefs).to_numpy()
        weights["preference"] = PREFERENCE_WEIGHT

    components["rating"] = bayesian_rating_score(df).to_numpy()
    weights["rating"] = RATING_WEIGHT

    total_weight = sum(weights.values())
    score = sum(components[k] * weights[k] for k in components) / total_weight

    mask = apply_filters(df, prefs, collection_idx)
    candidates = df.assign(score=score)[mask]
    top = candidates.nlargest(top_n, "score")

    reasons = []
    for idx, row in top.iterrows():
        match = df.loc[best_match[idx]] if best_match[idx] >= 0 else None
        reasons.append(explain(row, match, prefs, note_idf))
    return top, reasons
