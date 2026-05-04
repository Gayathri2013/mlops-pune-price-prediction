"""Stage 2 of the DVC pipeline — cleaned CSV → feature matrix + helper artifacts.

Extracts the *deterministic* feature-engineering work from M2_Lab2 (NLP
Feature Engineering & Target Encoding). All word clouds, plots, and EDA
visualizations have been removed — this script only writes the artifacts
that downstream stages (and the FastAPI service) consume.

Input  : <PROJECT_ROOT>/data_cleaned.csv     (output of mlops.clean_data)

Outputs (all written to project root or model/):
  * model_features.csv                       — combined feature matrix (X)
  * model_target.npy                         — target vector (y, in ₹L)
  * model/count_vectorizer.pkl               — fitted bigram CountVectorizer
  * model/sub_area_price_map.pkl             — sub-area → mean-price dict
  * model/amenities_score_price_map.pkl      — amenities-score → mean-price dict
  * model/feature_cols.pkl                   — list of structural+engineered cols
  * model/all_feature_names.pkl              — full feature list (incl. bigrams)

Determinism: Lab 2's logic is deterministic — POS tagging, CountVectorizer,
groupby-mean, np.sum, and column concatenation all produce identical bytes
across runs. Re-running this script overwrites the pkls with bit-identical
versions, which is exactly what DVC needs to skip the stage when nothing
upstream has changed.

Run from the project root:
    python -m mlops.build_features
"""

from __future__ import annotations

import pickle
import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
CLEAN_CSV    = PROJECT_ROOT / "data_cleaned.csv"
MODEL_DIR    = PROJECT_ROOT / "model"
FEATURES_CSV = PROJECT_ROOT / "model_features.csv"
TARGET_NPY   = PROJECT_ROOT / "model_target.npy"

# ── Constants from Lab 2 ─────────────────────────────────────────────────────
REPLACE_BY_SPACE = re.compile(r"[/(){}\[\]\|@,;!]")
BAD_SYMBOLS      = re.compile(r"[^0-9a-z #+_]")

AMENITY_COLS = [
    "ClubHouse Cleaned", "School Cleaned", "Hospital Cleaned",
    "Mall Cleaned", "Park Cleaned", "Pool Cleaned", "Gym Cleaned",
]

# Order of structural+engineered features — MUST match Lab 2 §7 exactly,
# because the trained Voting Regressor was fitted with this column order.
STRUCTURAL_FEATURE_COLS = [
    "Property Type Cleaned", "Area Cleaned",
    "ClubHouse Cleaned", "School Cleaned", "Hospital Cleaned",
    "Mall Cleaned", "Park Cleaned", "Pool Cleaned", "Gym Cleaned",
    "Price by sub-area", "Amenities score", "Price by Amenities score",
    "Noun_Counts", "Verb_Counts", "Adjective_Counts",
]


# ── NLTK setup (lazy — only download what's missing) ─────────────────────────
def _ensure_nltk():
    """Download the NLTK resources Lab 2 uses, idempotently."""
    import nltk
    for pkg in ("punkt", "punkt_tab", "averaged_perceptron_tagger",
                "averaged_perceptron_tagger_eng", "stopwords"):
        nltk.download(pkg, quiet=True)


# ── Text preprocessing (Lab 2 §3) ────────────────────────────────────────────
def _build_text_prepare_fn(stopwords_set: set):
    """Return the same `text_prepare` closure Lab 2 uses for description cleaning."""

    def text_prepare(text) -> str:
        text = str(text).lower()
        text = REPLACE_BY_SPACE.sub(" ", text)
        text = BAD_SYMBOLS.sub("", text)
        return " ".join(w for w in text.split() if w not in stopwords_set and len(w) > 2)

    return text_prepare


# ── POS counts (Lab 2 §4) ────────────────────────────────────────────────────
def _extract_pos_counts(text: str) -> tuple[int, int, int]:
    """(noun_count, verb_count, adj_count) — Lab 2's exact tag sets."""
    from nltk import pos_tag
    from nltk.tokenize import word_tokenize

    if pd.isna(text) or text == "":
        return 0, 0, 0
    tagged = pos_tag(word_tokenize(str(text)))
    nouns = sum(1 for _, t in tagged if t in ("NN", "NNS", "NNP"))
    verbs = sum(1 for _, t in tagged if t.startswith("VB"))
    adjs  = sum(1 for _, t in tagged if t in ("JJ", "JJR", "JJS"))
    return nouns, verbs, adjs


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> dict[str, Path]:
    print("=" * 70)
    print(" Stage 2: build_features — data_cleaned.csv → feature matrix")
    print("=" * 70)

    if not CLEAN_CSV.exists():
        raise FileNotFoundError(
            f"{CLEAN_CSV.name} not found. Run `python -m mlops.clean_data` first "
            "(or `dvc repro` to run the full pipeline)."
        )

    MODEL_DIR.mkdir(parents=True, exist_ok=True)

    # 1. NLTK resources + stopwords.
    print("\n[1/6] Loading NLTK resources")
    _ensure_nltk()
    from nltk.corpus import stopwords
    stop_words = set(stopwords.words("english"))

    # 2. Load cleaned data.
    print(f"[2/6] Loading {CLEAN_CSV.name}")
    df = pd.read_csv(CLEAN_CSV)
    print(f"      Loaded: {df.shape}")

    # 3. Description preprocessing + POS counts.
    print("[3/6] Preprocessing descriptions and extracting POS counts")
    text_prepare = _build_text_prepare_fn(stop_words)
    df["Description Processed"] = df["Description Cleaned"].apply(text_prepare)

    pos = df["Description Processed"].apply(_extract_pos_counts)
    df["Noun_Counts"]      = pos.apply(lambda x: x[0])
    df["Verb_Counts"]      = pos.apply(lambda x: x[1])
    df["Adjective_Counts"] = pos.apply(lambda x: x[2])

    # 4. CountVectorizer — top 10 bigrams (Lab 2 §5).
    print("[4/6] Fitting CountVectorizer (bigrams, top 10) on descriptions")
    from sklearn.feature_extraction.text import CountVectorizer
    cv = CountVectorizer(ngram_range=(2, 2), max_features=10, stop_words="english")
    cv.fit(df["Description Processed"])
    text_matrix = cv.transform(df["Description Processed"])
    text_df = pd.DataFrame(text_matrix.toarray(),
                           columns=cv.get_feature_names_out(),
                           index=df.index)
    print(f"      Bigrams selected: {cv.get_feature_names_out().tolist()}")

    with open(MODEL_DIR / "count_vectorizer.pkl", "wb") as fh:
        pickle.dump(cv, fh)

    # 5. Target encoding — sub-area mean price + amenities score (Lab 2 §6).
    print("[5/6] Target-encoding sub-area and amenities score")
    sub_area_price_map = df.groupby("Sub-Area Cleaned")["Price Cleaned"].mean().to_dict()
    df["Price by sub-area"] = df["Sub-Area Cleaned"].map(sub_area_price_map)

    df["Amenities score"] = df[AMENITY_COLS].sum(axis=1)
    amenities_price_map = df.groupby("Amenities score")["Price Cleaned"].mean().to_dict()
    df["Price by Amenities score"] = df["Amenities score"].map(amenities_price_map)

    with open(MODEL_DIR / "sub_area_price_map.pkl", "wb") as fh:
        pickle.dump(sub_area_price_map, fh)
    with open(MODEL_DIR / "amenities_score_price_map.pkl", "wb") as fh:
        pickle.dump(amenities_price_map, fh)

    # 6. Assemble feature matrix — column order matches the trained model.
    print("[6/6] Assembling feature matrix")
    X_structural = df[STRUCTURAL_FEATURE_COLS].fillna(0)
    X_combined = pd.concat(
        [X_structural.reset_index(drop=True), text_df.reset_index(drop=True)],
        axis=1,
    )
    y = df["Price Cleaned"].values

    # 7. Persist.
    X_combined.to_csv(FEATURES_CSV, index=False)
    np.save(TARGET_NPY, y)
    with open(MODEL_DIR / "feature_cols.pkl", "wb") as fh:
        pickle.dump(STRUCTURAL_FEATURE_COLS, fh)
    with open(MODEL_DIR / "all_feature_names.pkl", "wb") as fh:
        pickle.dump(X_combined.columns.tolist(), fh)

    paths = {
        "features":             FEATURES_CSV,
        "target":               TARGET_NPY,
        "count_vectorizer":     MODEL_DIR / "count_vectorizer.pkl",
        "sub_area_price_map":   MODEL_DIR / "sub_area_price_map.pkl",
        "amenities_price_map":  MODEL_DIR / "amenities_score_price_map.pkl",
        "feature_cols":         MODEL_DIR / "feature_cols.pkl",
        "all_feature_names":    MODEL_DIR / "all_feature_names.pkl",
    }

    print(f"\n✅ Feature matrix: {X_combined.shape[0]} rows × {X_combined.shape[1]} features")
    print("   Artifacts written:")
    for name, p in paths.items():
        size_kb = p.stat().st_size / 1024
        print(f"      {p.relative_to(PROJECT_ROOT).as_posix():<45s} ({size_kb:>5.1f} KB)")

    return paths


if __name__ == "__main__":
    main()
