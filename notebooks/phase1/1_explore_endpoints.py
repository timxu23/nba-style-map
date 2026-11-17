"""
Phase 1.1a — endpoint reconnaissance.

Before we can choose a feature set, we need to see what the endpoints ACTUALLY
return for our season. "The feature set defines the space," so we ground the
choice in real columns instead of guessing from docs.

Everything goes THROUGH THE CACHE (Phase 0 hard rule). This is analysis code:
it reads config + cache, touches the network only on a miss.

Run:  python3 notebooks/phase1/1_explore_endpoints.py
"""
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

import pandas as pd
from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayershotlocations,
    leaguedashptstats,
)
from src.cache import fetch_or_load
from src import config

pd.set_option("display.max_columns", 200)
pd.set_option("display.width", 200)

SEASON = config.SEASON
STYPE = config.SEASON_TYPE


def show(title, df, n=3):
    print("\n" + "=" * 78)
    print(title)
    print("=" * 78)
    print(f"shape: {df.shape}")
    print("columns:", list(df.columns))
    print(df.head(n).to_string())


# --- LeagueDashPlayerStats: three measure types --------------------------------
# per_mode_detailed="Per100Possessions" removes minutes as a hidden variable
# for the counting-style columns; the rate/pct columns are per-mode-invariant.
common = dict(season=SEASON, season_type_all_star=STYPE)

adv = fetch_or_load(
    leaguedashplayerstats.LeagueDashPlayerStats,
    dict(common, per_mode_detailed="Per100Possessions",
         measure_type_detailed_defense="Advanced"),
)
show("LeagueDashPlayerStats — Advanced (USG%, AST%, TOV%, REB%, PACE...)", adv)

base = fetch_or_load(
    leaguedashplayerstats.LeagueDashPlayerStats,
    dict(common, per_mode_detailed="Per100Possessions",
         measure_type_detailed_defense="Base"),
)
show("LeagueDashPlayerStats — Base (FGA, FG3A, FTA -> 3PAr, FTr)", base)

scoring = fetch_or_load(
    leaguedashplayerstats.LeagueDashPlayerStats,
    dict(common, per_mode_detailed="Per100Possessions",
         measure_type_detailed_defense="Scoring"),
)
show("LeagueDashPlayerStats — Scoring (PCT_AST_FGM = self-creation, shot mix)", scoring)

# --- Shot-zone distribution ----------------------------------------------------
# distance_range="By Zone" gives named zones (Restricted Area, Paint non-RA,
# Mid-Range, corner 3s, above-break 3). NOTE: this endpoint returns MULTI-INDEX
# columns — we print the structure so we can decide how to flatten it.
shotloc = fetch_or_load(
    leaguedashplayershotlocations.LeagueDashPlayerShotLocations,
    dict(common, per_mode_detailed="Totals", distance_range="By Zone"),
)
show("LeagueDashPlayerShotLocations — By Zone (FGA share per zone)", shotloc)
print("\n(column tuples if MultiIndex):")
print(list(shotloc.columns))

# --- Passing tendencies --------------------------------------------------------
passing = fetch_or_load(
    leaguedashptstats.LeagueDashPtStats,
    dict(common, per_mode_simple="PerGame", player_or_team="Player",
         pt_measure_type="Passing"),
)
show("LeagueDashPtStats — Passing (passes made, potential assists...)", passing)

print("\n\nDone. Next: pick the rich feature columns from what we just saw.")
