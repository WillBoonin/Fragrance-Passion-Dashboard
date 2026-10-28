# Fragrance Recommendation Dashboard

Personal Streamlit app (Python + pandas + scikit-learn) that recommends fragrances
based on a user's collection and preferences. Run with `streamlit run app.py` using
the project venv (`.venv/`).

The CSVs are large: inspect them with pandas only (shape, head, value_counts),
never read or print whole files.

## Code layout
- `data_loader.py`: reads, cleans and joins the CSVs (`load_fragrances()`, cached).
  Output has list columns `notes`, `top_notes`, `middle_notes`, `base_notes`, `accords`,
  `perfumers`, `seasons`, `occasions`, plus a unique `label` ("Name – Brand (Year)").
- `seasons.py`: keyword maps from accords to season/occasion (estimates).
- `recommender.py`: TF-IDF notes + rank-weighted accords → cosine similarity, blended
  with preference match and a Bayesian (vote-count-adjusted) rating. Weights are constants
  at the top of the file.
- `collection_store.py`: the collection is saved as Fragrantica numeric IDs (from the url)
  in the page URL (`?collection=31861,33519`), and also in `my_collection.json` when running
  locally. The file is skipped on Streamlit Community Cloud (detected via `/mount/src`).
- `prepare_data.py`: one-off that builds `FragranceData/fra_names.csv` (display names)
  from the 32MB `fra_perfumes.csv`, which is git-ignored so the repo stays small.
- Deployed via Streamlit Community Cloud from a GitHub repo (main file `app.py`).
- `app.py`: Streamlit UI with 4 tabs (Collection, Preferences, Recommendations, Explore).
- Headless check: `streamlit.testing.v1.AppTest.from_file("app.py", default_timeout=60)`.

## Data files (in `FragranceData/`, not `data/`)

### `fra_cleaned.csv`: primary base (24,063 rows, one per perfume)
- **Format:** semicolon-separated, **cp1252/latin-1 encoded** (UTF-8 fails), CRLF,
  decimal comma in `Rating Value` (e.g. `"3,97"`).
- Columns:
  - `url`: Fragrantica URL, unique, all lowercase
  - `Perfume`, `Brand`: lowercase hyphenated slugs (`amber-oud-bleu-edition`, `al-haramain-perfumes`)
  - `Country`: brand country
  - `Gender`: `women` / `men` / `unisex`
  - `Rating Value`: string with comma decimal, range 1.42–4.93 (mean 3.96)
  - `Rating Count`: int, min 26, median 127, max ~30k
  - `Year`: float, 2,037 nulls, range 1781–2024
  - `Top`, `Middle`, `Base`: comma-separated lowercase notes, no nulls. 1,671 unique notes
    (most common: musk, bergamot, sandalwood, jasmine, amber, patchouli, vanilla)
  - `Perfumer1`: `"unknown"` for ~51% of rows. `Perfumer2`: mostly null
  - `mainaccord1`..`mainaccord5`: ordered accords (1 = strongest), a few nulls in 2–5.
    84 unique accords (woody, citrus, sweet, fresh, amber, vanilla, white floral, …)
- 217 duplicate (Perfume, Brand) pairs exist (different URLs). Keep `url` as the ID.

### `fra_perfumes.csv`: wider and messier (70,103 rows)
- Comma-separated, UTF-8. Columns: `Name`, `Gender`, `Rating Value`, `Rating Count`
  (string), `Main Accords` (stringified Python list), `Perfumers` (stringified list,
  ~61% empty `[]`), `Description` (free text), `url`.
- `Name` is name + brand + gender glued together, e.g. `"Eclipse 1996 Oriflamefor women"`.
- No structured notes: notes are only in `Description` prose ("Top notes are …").
- ~6k rows have no rating.

### How they relate
- Shared key: `url`. Matches **after lowercasing** (perfumes URLs keep original case).
  All 24,063 cleaned rows have a match in perfumes.
- Use `fra_cleaned.csv` as the base (structured notes, accords, year, perfumer, country).
  From `fra_perfumes.csv`, take only the properly-cased display name, pre-extracted to
  `fra_names.csv`, which is what the app reads. To split it, strip the trailing
  `for women` / `for men` / `for women and men`, then strip the brand.

## Not in the data
- **Price:** no price column in either file. The price filter is skipped.
- **Season / occasion:** no column. These are inferred from accords and notes with a
  hand-written keyword map (e.g. fresh/citrus/aquatic → summer, warm spicy/amber/vanilla
  → winter).
- No longevity, sillage, or images.

## Cleaning applied in `data_loader.py`
- Read with `sep=";"` and `encoding="cp1252"`. Convert `Rating Value` to float.
- Split Top/Middle/Base on `,` and strip. Collect accords into a list (drop NaN).
- Treat `Perfumer1 == "unknown"` as missing. Join `Perfumer2` when present.
- Use title-cased slugs as fallback display names when the perfumes join gives nothing.
