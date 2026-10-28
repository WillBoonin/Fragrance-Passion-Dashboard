"""One-off: extract the display names from the large fra_perfumes.csv into a small file.

fra_perfumes.csv (32MB) is too big for GitHub's web upload and the app only needs its
properly-cased `Name`. Run once with: python prepare_data.py
"""

import pandas as pd

from data_loader import DATA_DIR, NAMES_CSV

perfumes = pd.read_csv(DATA_DIR / "fra_perfumes.csv", usecols=["Name", "url"])
perfumes["url_key"] = perfumes["url"].str.lower().str.strip()
cleaned_urls = pd.read_csv(DATA_DIR / "fra_cleaned.csv", sep=";", encoding="cp1252", usecols=["url"])

names = perfumes[perfumes["url_key"].isin(cleaned_urls["url"].str.lower().str.strip())]
names = names.drop_duplicates("url_key")[["url_key", "Name"]]
names.to_csv(NAMES_CSV, index=False)
print(f"Wrote {len(names):,} names to {NAMES_CSV}")
