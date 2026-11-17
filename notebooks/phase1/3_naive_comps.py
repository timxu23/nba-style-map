"""
Phase 1.4 — the deliberate failure, then the diagnosis.

Three acts:
  A. Naive Euclidean comps on the RAW (un-standardized) matrix -> bad comps,
     because big-magnitude columns dominate the distance.
  B. Same tool on the STANDARDIZED matrix -> better, but still imperfect.
  C. Diagnose the *remaining* badness: correlation matrix of the rich features,
     surfacing redundant clusters that double-count. This motivates PCA.

Run:  python3 notebooks/phase1/3_naive_comps.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import numpy as np
import pandas as pd

from src.cache import PROJECT_ROOT
from src import config, comps

PROCESSED = PROJECT_ROOT / "data" / "processed"
pd.set_option("display.width", 200)

rich_raw = pd.read_parquet(PROCESSED / f"rich_raw_{config.SEASON}.parquet")
rich_z = pd.read_parquet(PROCESSED / f"rich_z_{config.SEASON}.parquet")

# Players we know well, spanning archetypes.
QUERIES = ["Luka Don", "Nikola Joki", "Rudy Gobert", "Stephen Curry"]


def show_comps(title, df):
    print("\n" + "=" * 70)
    print(title)
    print("=" * 70)
    for name in QUERIES:
        res = comps.nearest(name, df, config.RICH_FEATURES, k=5)
        who = df.loc[df["PLAYER_NAME"].str.contains(name, case=False), "PLAYER_NAME"].iloc[0]
        print(f"\n{who} ->")
        for _, r in res.iterrows():
            print(f"    {r['PLAYER_NAME']:24s} {r['distance']:.2f}")


# --- Act A: raw. Look at which feature has the largest spread ----------------
print("Feature magnitude spread (raw) — the biggest std dominates distance:")
spread = rich_raw[config.RICH_FEATURES].std().sort_values(ascending=False)
print(spread.head(5).round(3).to_string())
print("  ...")
print(spread.tail(3).round(4).to_string())

show_comps("ACT A — NAIVE comps on RAW features (expect junk)", rich_raw)

# --- Act B: standardized ----------------------------------------------------
show_comps("ACT B — comps on STANDARDIZED features (expect better)", rich_z)

# --- Act C: diagnose remaining badness via correlation ----------------------
print("\n" + "=" * 70)
print("ACT C — correlation structure: which features double-count?")
print("=" * 70)
corr = rich_z[config.RICH_FEATURES].corr()

# List the most strongly correlated feature pairs (|r| high), redundancy = weight.
pairs = (corr.where(~np.eye(len(corr), dtype=bool))
         .stack()
         .rename("r")
         .reset_index())
pairs["absr"] = pairs["r"].abs()
pairs = pairs[pairs["level_0"] < pairs["level_1"]]  # de-dupe symmetric pairs
top = pairs.sort_values("absr", ascending=False).head(12)
print("\nMost correlated feature pairs (|r| high => redundant, double-counted):")
for _, r in top.iterrows():
    print(f"    {r['level_0']:20s} <-> {r['level_1']:20s}  r = {r['r']:+.2f}")

print("\nInterpretation: clusters like these mean Euclidean distance is secretly")
print("weighted toward whatever concept has the most columns. PCA (Phase 2) fixes")
print("this by rotating to uncorrelated axes ordered by variance.")
