"""Personal fragrance recommendation dashboard. Run with: streamlit run app.py"""

import altair as alt
import pandas as pd
import streamlit as st

from collection_store import load_collection_ids, save_collection_ids
from data_loader import load_fragrances
from recommender import Preferences, build_feature_matrix, recommend
from seasons import OCCASIONS, SEASONS

st.set_page_config(page_title="Fragrance Finder", page_icon="🌸", layout="wide")


BAR_COLOR = "#2a78d6"
GENDER_CHOICES = {
    "No preference": [],
    "Feminine": ["Feminine"],
    "Masculine": ["Masculine"],
    "Unisex": ["Unisex"],
}


@st.cache_resource(show_spinner="Building similarity vectors…")
def load_features(_df: pd.DataFrame):
    """Feature matrix + note rarity (IDF) lookup, built once per session."""
    features, note_vectorizer = build_feature_matrix(_df)
    note_idf = dict(zip(note_vectorizer.get_feature_names_out(), note_vectorizer.idf_))
    return features, note_idf


def values_by_popularity(series: pd.Series) -> list[str]:
    """Unique items of a list column, most common first."""
    return series.explode().dropna().value_counts().index.tolist()


def render_fragrance_card(row: pd.Series, reason: str | None = None) -> None:
    """Show every field the dataset has for one fragrance."""
    with st.container(border=True):
        st.markdown(f"### {row['name']}")
        year = f" · {row['year']}" if pd.notna(row["year"]) else ""
        st.caption(f"**{row['brand']}** ({row['country']}){year} · {row['gender']}")
        if reason:
            st.info(reason, icon="💡")
        st.markdown(f"⭐ **{row['rating']:.2f}** / 5 from {row['rating_count']:,} ratings")
        st.markdown("**Main accords:** " + ", ".join(row["accords"]))
        st.markdown("**Top:** " + (", ".join(row["top_notes"]) or "—"))
        st.markdown("**Heart:** " + (", ".join(row["middle_notes"]) or "—"))
        st.markdown("**Base:** " + (", ".join(row["base_notes"]) or "—"))
        st.markdown("**Perfumer:** " + (", ".join(row["perfumers"]) or "Unknown"))
        st.markdown(
            f"**Seasons (estimated):** {', '.join(row['seasons'])} · "
            f"**Best for:** {', '.join(row['occasions'])}"
        )
        st.markdown(f"[View on Fragrantica]({row['url']})")


def render_cards(rows: pd.DataFrame, reasons: list[str] | None = None, columns: int = 2) -> None:
    cols = st.columns(columns)
    for i, (_, row) in enumerate(rows.iterrows()):
        with cols[i % columns]:
            render_fragrance_card(row, reasons[i] if reasons else None)


def save_collection(df: pd.DataFrame) -> None:
    """Persist the current selection as Fragrantica IDs (stable, unlike labels)."""
    selected = st.session_state["collection"]
    save_collection_ids(df.loc[df["label"].isin(selected), "fragrance_id"].tolist())


def collection_tab(df: pd.DataFrame) -> list[str]:
    """Search and pick owned/liked fragrances. Returns the selected labels."""
    st.subheader("My Collection")
    st.write("Type a name or brand to search, then pick the fragrances you own or love.")
    if "collection" not in st.session_state:
        saved_ids = load_collection_ids()
        st.session_state["collection"] = df.loc[df["fragrance_id"].isin(saved_ids), "label"].tolist()
        save_collection(df)  # keep the page URL in sync with what was restored
    selected = st.multiselect(
        "Search fragrances",
        options=df["label"].tolist(),
        key="collection",
        placeholder="e.g. Baccarat Rouge, Dior, Santal 33…",
        on_change=save_collection,
        args=(df,),
    )
    st.caption("Your collection is saved in this page's link: bookmark it to come back to it later.")
    if not selected:
        st.info("Your collection is empty. Add a few fragrances to get recommendations.")
        return []
    render_cards(df[df["label"].isin(selected)])
    return selected


def preferences_tab(df: pd.DataFrame) -> Preferences:
    """Short questionnaire. Returns the user's preferences."""
    st.subheader("Preferences")
    mode = st.radio(
        "Do you want something similar to your collection, or something different?",
        ["Similar to my collection", "Something different"],
        horizontal=True,
    )
    st.caption("💸 Price filter unavailable: the dataset has no price information.")

    notes = values_by_popularity(df["notes"])
    accords = values_by_popularity(df["accords"])
    col1, col2 = st.columns(2)
    with col1:
        preferred_notes = st.multiselect("Notes you love", notes, placeholder="e.g. vanilla, bergamot")
        preferred_accords = st.multiselect("Accords you like", accords, placeholder="e.g. woody, citrus")
        seasons = st.multiselect(
            "Season", SEASONS, placeholder="Any season",
            help="Estimated from each fragrance's main accords (the data has no season column).",
        )
        gender = st.radio("Gender label", list(GENDER_CHOICES), horizontal=True)
    with col2:
        avoid_notes = st.multiselect("Notes to avoid", notes, placeholder="e.g. oud, patchouli")
        occasions = st.multiselect(
            "Occasion", OCCASIONS, placeholder="Any occasion",
            help="Day = fresh, clean, airy accords. Night = warm, sweet, rich accords. Estimated.",
        )
        min_rating = st.slider("Minimum rating", 1.0, 5.0, 3.5, 0.1)
        min_rating_count = st.number_input(
            "Minimum number of ratings", min_value=0, max_value=30000, value=50, step=25
        )

    return Preferences(
        mode="similar" if mode.startswith("Similar") else "different",
        preferred_notes=preferred_notes,
        avoid_notes=avoid_notes,
        preferred_accords=preferred_accords,
        seasons=seasons,
        occasions=occasions,
        genders=GENDER_CHOICES[gender],
        min_rating=min_rating,
        min_rating_count=int(min_rating_count),
    )


def recommendations_tab(df: pd.DataFrame, collection: list[str], prefs: Preferences) -> None:
    st.subheader("Recommendations")
    if not collection and not (prefs.preferred_notes or prefs.preferred_accords):
        st.info("Add fragrances in **My Collection** or pick notes/accords in **Preferences** to get started.")
        return
    features, note_idf = load_features(df)
    collection_idx = df.index[df["label"].isin(collection)].tolist()
    top, reasons = recommend(df, features, note_idf, collection_idx, prefs)
    if top.empty:
        st.warning("Nothing matches these filters. Try loosening the rating, season or avoided notes.")
        return
    basis = "your collection" if collection else "your preferred notes and accords"
    st.caption(f"Top {len(top)} matches based on {basis}, filters and ratings.")
    render_cards(top, reasons)


def bar_chart(data: pd.DataFrame, x: str, y: str, x_title: str, tooltip: list) -> alt.Chart:
    """Horizontal single-series bar chart, sorted by value."""
    return (
        alt.Chart(data)
        .mark_bar(color=BAR_COLOR, cornerRadiusEnd=4, size=14)
        .encode(
            x=alt.X(f"{x}:Q", title=x_title, axis=alt.Axis(grid=True, gridOpacity=0.3)),
            y=alt.Y(f"{y}:N", sort="-x", title=None),
            tooltip=tooltip,
        )
        .properties(height=alt.Step(22))
    )


def explore_tab(df: pd.DataFrame) -> None:
    st.subheader("Explore")
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Most common main accords**")
        accord_counts = df["accords"].explode().value_counts().head(15).rename_axis("accord")
        accord_counts = accord_counts.reset_index(name="fragrances")
        st.altair_chart(
            bar_chart(accord_counts, "fragrances", "accord", "Number of fragrances",
                      ["accord", alt.Tooltip("fragrances", format=",")]),
            width="stretch",
        )
        with st.expander("Show as table"):
            st.dataframe(accord_counts, hide_index=True)

    with col2:
        min_count = st.slider("Top-rated brands: minimum fragrances per brand", 5, 100, 20, 5)
        brands = (
            df.assign(weighted=df["rating"] * df["rating_count"])
            .groupby("brand")
            .agg(fragrances=("name", "size"), weighted=("weighted", "sum"), votes=("rating_count", "sum"))
            .query("fragrances >= @min_count")
        )
        brands["avg_rating"] = (brands["weighted"] / brands["votes"]).round(2)
        brands = brands.nlargest(15, "avg_rating").reset_index()[["brand", "avg_rating", "fragrances", "votes"]]
        st.markdown("**Top-rated brands** (vote-weighted average rating)")
        st.altair_chart(
            alt.Chart(brands)
            .mark_circle(color=BAR_COLOR, size=90, opacity=1)
            .encode(
                x=alt.X("avg_rating:Q", title="Average rating", scale=alt.Scale(zero=False),
                        axis=alt.Axis(grid=True, gridOpacity=0.3)),
                y=alt.Y("brand:N", sort="-x", title=None),
                tooltip=["brand", "avg_rating", "fragrances", alt.Tooltip("votes", format=",")],
            )
            .properties(height=alt.Step(22)),
            width="stretch",
        )
        with st.expander("Show as table"):
            st.dataframe(brands, hide_index=True)

    st.divider()
    st.markdown("**Notes in a fragrance**")
    choice = st.selectbox("Pick a fragrance", df["label"].tolist(), index=None,
                          placeholder="Search by name or brand…")
    if choice:
        row = df[df["label"] == choice].iloc[0]
        layers = st.columns(4)
        for column, (title, key) in zip(
            layers, [("Top", "top_notes"), ("Heart", "middle_notes"), ("Base", "base_notes"), ("Main accords", "accords")]
        ):
            with column:
                st.markdown(f"**{title}**")
                st.markdown("\n".join(f"- {item}" for item in row[key]) or "—")


def main() -> None:
    st.title("🌸 Fragrance Finder")
    df = load_fragrances()
    tab_collection, tab_prefs, tab_recs, tab_explore = st.tabs(
        ["My Collection", "Preferences", "Recommendations", "Explore"]
    )
    with tab_collection:
        collection = collection_tab(df)
    with tab_prefs:
        prefs = preferences_tab(df)
    with tab_recs:
        recommendations_tab(df, collection, prefs)
    with tab_explore:
        explore_tab(df)


main()
