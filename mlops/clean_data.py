"""Stage 1 of the DVC pipeline — raw Excel → cleaned CSV.

Extracts the *deterministic* transformations from M2_Lab1 (Data Cleaning &
EDA). All plotting, EDA, and exploratory `display()` calls have been stripped
— this script only does what's needed to produce `data_cleaned.csv`.

Input  : <PROJECT_ROOT>/Pune Real Estate Data.xlsx   (raw, 200 rows × 18 cols)
Output : <PROJECT_ROOT>/data_cleaned.csv             (≈197 rows × 17 cols)

The output is the EXACT dataset Lab 2 reads. Re-running this script and then
`mlops.build_features` reproduces every Lab 2 artifact byte-for-byte.

Run from the project root:
    python -m mlops.clean_data

Notes:
  * Output row count drops slightly because rows with missing price are
    dropped (no target → can't train).
  * Outlier handling is percentile clipping at 5%/95% for Area and Price
    (matches Lab 1 §6.2). Conservative — retains all rows, only the
    extreme values are pulled in.
"""

from __future__ import annotations

import re
from pathlib import Path

import numpy as np
import pandas as pd

# ── Paths ────────────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent.parent
RAW_XLSX     = PROJECT_ROOT / "Pune Real Estate Data.xlsx"
CLEAN_CSV    = PROJECT_ROOT / "data_cleaned.csv"

# ── Regex (pre-compiled, used in multiple cleaners) ──────────────────────────
NUMBERS_PATTERN = re.compile(r"[-+]?(\d*\.\d+|\d+)")


# ── Cleaners — one function per messy column ─────────────────────────────────
def _clean_property_type(val) -> float:
    """Extract bedroom count from strings like '2 BHK', '3-BHK', '1 RK'."""
    matches = NUMBERS_PATTERN.findall(str(val))
    return float(matches[0]) if matches else 0.0


def _clean_area(val) -> float:
    """Extract numeric area. Ranges like '800 - 1200' return their average."""
    nums = NUMBERS_PATTERN.findall(str(val))
    if len(nums) == 1:
        return float(nums[0])
    if len(nums) == 2:
        return (float(nums[0]) + float(nums[1])) / 2
    if len(nums) > 2:
        return float(nums[0])
    return np.nan


def _clean_price(val) -> float:
    """Extract numeric price (in lakhs ₹) from text-mixed field."""
    nums = NUMBERS_PATTERN.findall(str(val))
    return float(nums[0]) if nums else np.nan


def _split_location(df: pd.DataFrame) -> pd.DataFrame:
    """Split 'Pune, Maharashtra, India' → city / state / country (lowercased)."""
    df["City"]    = df["Location"].apply(lambda x: x.split(",")[0].lower().strip())
    df["State"]   = df["Location"].apply(lambda x: x.split(",")[1].lower().strip())
    df["Country"] = df["Location"].apply(lambda x: x.split(",")[2].lower().strip())
    return df


def _encode_binary_amenities(df: pd.DataFrame) -> pd.DataFrame:
    """Encode the 7 Yes/No amenity columns as 0/1 ints with the Lab 1 names."""
    binary_cols = {
        "ClubHouse":                       "ClubHouse Cleaned",
        "School / University in Township ": "School Cleaned",   # trailing space matches raw col name
        "Hospital in TownShip":            "Hospital Cleaned",
        "Mall in TownShip":                "Mall Cleaned",
        "Park / Jogging track":            "Park Cleaned",
        "Swimming Pool":                   "Pool Cleaned",
        "Gym":                             "Gym Cleaned",
    }
    for raw_col, clean_col in binary_cols.items():
        normalized = df[raw_col].apply(lambda x: str(x).lower().strip())
        df[clean_col] = normalized.map({"yes": 1, "no": 0}).fillna(0).astype(int)
    return df


def _clip_outliers(df: pd.DataFrame, col: str, low_pct: int = 5, hi_pct: int = 95) -> pd.DataFrame:
    """Clip a numeric column at the given percentiles (matches Lab 1 §6.2)."""
    lower = df[col].quantile(low_pct / 100)
    upper = df[col].quantile(hi_pct / 100)
    n_clipped = ((df[col] < lower) | (df[col] > upper)).sum()
    df[col] = df[col].clip(lower=lower, upper=upper)
    print(f"      {col}: clipped {n_clipped} values → [{lower:.0f}, {upper:.0f}]")
    return df


# ── Main ─────────────────────────────────────────────────────────────────────
def main() -> Path:
    print("=" * 70)
    print(" Stage 1: clean_data — raw Excel → data_cleaned.csv")
    print("=" * 70)

    if not RAW_XLSX.exists():
        raise FileNotFoundError(
            f"Raw dataset not found: {RAW_XLSX}\n"
            "Place 'Pune Real Estate Data.xlsx' at the project root, or "
            "`dvc pull` if it's tracked on a DVC remote."
        )

    # 1. Load.
    print(f"\n[1/6] Loading {RAW_XLSX.name}")
    df = pd.read_excel(RAW_XLSX)
    print(f"      Raw shape: {df.shape}")

    # 2. Location → city / state / country.
    print("[2/6] Splitting Location into city / state / country")
    df = _split_location(df)

    # 3. Lowercase + strip the categorical text columns.
    print("[3/6] Cleaning text columns (lowercase + strip)")
    df["Sub-Area Cleaned"]         = df["Sub-Area"].apply(lambda x: str(x).lower().strip())
    df["Company Name Cleaned"]     = df["Company Name"].apply(lambda x: str(x).lower().strip())
    df["TownShip Cleaned"]         = df["TownShip Name/ Society Name"].apply(lambda x: str(x).lower().strip())
    df["Description Cleaned"]      = df["Description"].apply(lambda x: str(x).lower().strip())

    # 4. Numeric extractions via regex.
    print("[4/6] Extracting numeric fields via regex (property type, area, price)")
    df["Property Type Cleaned"] = df["Propert Type"].apply(_clean_property_type)  # NB: source typo
    df["Area Cleaned"]          = df["Property Area in Sq. Ft."].apply(_clean_area)
    df["Price Cleaned"]         = df["Price in lakhs"].apply(_clean_price)

    # 5. Binary amenity encoding.
    print("[5/6] Encoding 7 amenity columns as 0/1")
    df = _encode_binary_amenities(df)

    # 6. Assemble + drop NaN-price + clip outliers.
    print("[6/6] Assembling final dataframe")
    keep = [
        "City", "State", "Country",
        "Property Type Cleaned", "Sub-Area Cleaned",
        "Company Name Cleaned", "TownShip Cleaned",
        "Description Cleaned",
        "ClubHouse Cleaned", "School Cleaned", "Hospital Cleaned",
        "Mall Cleaned", "Park Cleaned", "Pool Cleaned", "Gym Cleaned",
        "Area Cleaned", "Price Cleaned",
    ]
    out = df[keep].copy()

    before = len(out)
    out = out.dropna(subset=["Price Cleaned"])
    print(f"      Dropped {before - len(out)} rows with missing Price")

    print("      Clipping outliers at 5/95 percentiles:")
    out = _clip_outliers(out, "Area Cleaned")
    out = _clip_outliers(out, "Price Cleaned")

    # 7. Save.
    out.to_csv(CLEAN_CSV, index=False)
    print(f"\n✅ Wrote {CLEAN_CSV.name}: {out.shape[0]} rows × {out.shape[1]} columns")
    return CLEAN_CSV


if __name__ == "__main__":
    main()
