"""
Phase 1.3 — build and standardize the two feature matrices.

Reads the universe (Phase 0) + config feature lists, runs the shared feature
code path (src/features.py), z-scores, and saves four artifacts to
data/processed/:

  rich_raw_<season>.parquet      identity + RICH_FEATURES, un-standardized
  rich_z_<season>.parquet        identity + RICH_FEATURES, z-scored
  shared_raw_<season>.parquet    identity + SHARED_FEATURES, un-standardized
  shared_z_<season>.parquet      identity + SHARED_FEATURES, z-scored

We keep BOTH raw and z-scored: Phase 1.4 (the deliberate failure) needs the raw
matrix to show how bad un-standardized distance is. The saved z-score mean/std
also go to disk — Phase 5 needs consistent scaling params.

Run:  python3 notebooks/phase1/2_build_matrices.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd

from src.cache import PROJECT_ROOT
from src import config, features

PROCESSED = PROJECT_ROOT / "data" / "processed"


def save(df: pd.DataFrame, name: str) -> None:
    out = PROCESSED / f"{name}_{config.SEASON}.parquet"
    df.to_parquet(out)
    print(f"  saved {out.relative_to(PROJECT_ROOT)}  ({df.shape[0]}x{df.shape[1]})")


# --- Load the universe and build the rich table -----------------------------
universe = pd.read_parquet(PROCESSED / f"universe_{config.SEASON}.parquet")
print(f"Universe: {len(universe)} players (>= {config.MIN_MINUTES} min)")

rich = features.build_rich_table(universe=universe)
print(f"Rich table: {rich.shape[0]} players x {len(config.RICH_FEATURES)} features")

# Coverage sanity: universe players missing from the feature table (should be 0).
missing = set(universe["PLAYER_ID"]) - set(rich["PLAYER_ID"])
print(f"Universe players missing from feature table: {len(missing)}")

# --- Shared matrix is a strict column subset of the rich table ---------------
shared = rich[config.ID_COLUMNS + config.SHARED_FEATURES].copy()

# --- Standardize ------------------------------------------------------------
rich_z, rich_mean, rich_std = features.standardize(rich, config.RICH_FEATURES)
shared_z, shared_mean, shared_std = features.standardize(shared, config.SHARED_FEATURES)

# Re-attach identity columns to the z-scored frames for convenience.
ids = rich[config.ID_COLUMNS]
rich_z = ids.merge(rich_z, on="PLAYER_ID")
shared_z = ids.merge(shared_z, on="PLAYER_ID")

# --- Save -------------------------------------------------------------------
print("\nSaving processed artifacts:")
save(rich, "rich_raw")
save(rich_z, "rich_z")
save(shared, "shared_raw")
save(shared_z, "shared_z")

# Scaling params (Phase 5 will need these to be explicit and saved).
pd.DataFrame({"mean": rich_mean, "std": rich_std}).to_parquet(
    PROCESSED / f"rich_scaler_{config.SEASON}.parquet")
pd.DataFrame({"mean": shared_mean, "std": shared_std}).to_parquet(
    PROCESSED / f"shared_scaler_{config.SEASON}.parquet")
print("  saved rich/shared scaler params (mean, std)")

# --- Quick sanity checks ----------------------------------------------------
print("\n--- sanity: z-scored columns should be ~mean 0, ~std 1 ---")
print(rich_z[config.RICH_FEATURES].mean().abs().max(), "= max |mean| (want ~0)")
print(rich_z[config.RICH_FEATURES].std(ddof=0).round(3).min(), "= min std (want ~1)")

print("\n--- sanity: play-type frequencies for a known player ---")
luka = rich[rich["PLAYER_NAME"].str.contains("Luka", case=False, na=False)]
if len(luka):
    show = ["PLAYER_NAME", "USG_PCT", "PT_ISO", "PT_PRBH", "PT_TRANSITION",
            "Z_ABOVE3", "PASSES_MADE_P100"]
    print(luka[show].to_string(index=False))

print("\nDone. Next: Phase 1.4 — the naive nearest-neighbor comp tool.")
