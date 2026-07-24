"""
Phase 0.4 — define the player universe.

Pulls one league-wide table of season totals THROUGH THE CACHE (never the API
directly), filters to the minimum-minutes threshold from config, and saves the
resulting universe to data/processed/.

Note the separation of concerns:
  - src/config.py  = the deliberate decisions (season, threshold)
  - src/cache.py   = how we get raw data
  - this script    = analysis: reads cache + config, writes a processed artifact

Run:  python3 notebooks/phase0/3_build_universe.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nba_api.stats.endpoints import leaguedashplayerstats
from src.cache import fetch_or_load, PROJECT_ROOT
from src import config

# One row per player: season TOTALS (so MIN and GP are cumulative, not per-game).
params = {
    "season": config.SEASON,
    "season_type_all_star": config.SEASON_TYPE,
    "per_mode_detailed": "Totals",
}

df = fetch_or_load(leaguedashplayerstats.LeagueDashPlayerStats, params)
print(f"\nLeague table: {df.shape[0]} players, {df.shape[1]} columns")

# Apply the universe filter.
universe = df[df["MIN"] >= config.MIN_MINUTES].copy()
universe = universe.sort_values("MIN", ascending=False).reset_index(drop=True)

print(f"Universe: {universe.shape[0]} players with >= {config.MIN_MINUTES} total minutes "
      f"(cut {df.shape[0] - universe.shape[0]})")

# Keep the identity + minutes columns we'll key on later.
keep = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION", "GP", "MIN"]
universe = universe[keep]

# Save the processed artifact (this one we DO keep, in data/processed/).
out = PROJECT_ROOT / "data" / "processed" / f"universe_{config.SEASON}.parquet"
out.parent.mkdir(parents=True, exist_ok=True)
universe.to_parquet(out)
print(f"Saved -> {out.relative_to(PROJECT_ROOT)}")

print("\nMinutes leaders (top 5):")
print(universe.head(5).to_string(index=False))
print("\nJust above the cut line (last 5):")
print(universe.tail(5).to_string(index=False))
