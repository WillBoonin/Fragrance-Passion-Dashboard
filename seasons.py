"""Infer season and occasion from main accords (the dataset has no such columns).

Each accord votes for the seasons/occasions it typically suits; stronger accords
(mainaccord1) count more than weaker ones (mainaccord5). These are rough heuristics.
"""

SEASONS = ["Spring", "Summer", "Fall", "Winter"]
OCCASIONS = ["Day", "Night"]

ACCORD_SEASONS = {
    "Spring": {"floral", "white floral", "yellow floral", "rose", "green", "fruity", "powdery",
               "iris", "violet", "aldehydic", "soapy", "fresh", "tuberose", "lactonic"},
    "Summer": {"citrus", "fresh", "aquatic", "marine", "ozonic", "green", "fresh spicy",
               "tropical", "coconut", "salty", "herbal", "aromatic", "mineral"},
    "Fall": {"woody", "leather", "tobacco", "earthy", "mossy", "patchouli", "warm spicy",
             "spicy", "smoky", "oud", "cinnamon", "coffee", "nutty", "rum", "whiskey", "wine"},
    "Winter": {"amber", "vanilla", "sweet", "gourmand", "balsamic", "warm spicy", "oud",
               "cacao", "chocolate", "caramel", "honey", "animalic", "oriental", "tobacco",
               "smoky", "leather", "cinnamon", "almond", "creamy"},
}

ACCORD_OCCASIONS = {
    "Day": {"citrus", "fresh", "aquatic", "marine", "green", "ozonic", "soapy", "aromatic",
            "fresh spicy", "white floral", "floral", "powdery", "herbal", "lavender", "fruity"},
    "Night": {"amber", "vanilla", "oud", "leather", "tobacco", "sweet", "warm spicy", "animalic",
              "gourmand", "smoky", "balsamic", "oriental", "cacao", "rum", "whiskey", "honey"},
}

ACCORD_RANK_WEIGHTS = [1.0, 0.8, 0.6, 0.45, 0.3]
TAG_THRESHOLD = 0.6  # a label is assigned if it scores >= 60% of the top label


def _tags_from_accords(accords: list[str], mapping: dict[str, set[str]]) -> list[str]:
    scores = {label: 0.0 for label in mapping}
    for accord, weight in zip(accords, ACCORD_RANK_WEIGHTS):
        for label, keywords in mapping.items():
            if accord in keywords:
                scores[label] += weight
    best = max(scores.values())
    if best == 0:
        return list(mapping)  # no signal: treat as versatile
    return [label for label, score in scores.items() if score >= TAG_THRESHOLD * best]


def infer_seasons(accords: list[str]) -> list[str]:
    return _tags_from_accords(accords, ACCORD_SEASONS)


def infer_occasions(accords: list[str]) -> list[str]:
    return _tags_from_accords(accords, ACCORD_OCCASIONS)
