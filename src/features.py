"""
Phase 1 — turn raw endpoint tables into a player-by-feature matrix.

This is THE feature code path. Two rules it exists to enforce:

  1. One code path, reused. The shared (NBA ∩ NCAA) box rates are produced by
     `shared_rates_from_box`, a pure function of named box inputs. Phase 5 feeds
     it college box totals; here we feed it NBA tables. Same function, same
     column names, so an NBA player and a prospect are built identically.

  2. Absence is information. A player missing from a Synergy play-type table did
     not run that play type — their frequency is 0.0, filled on merge, never
     imputed.

All raw data arrives THROUGH THE CACHE (Phase 0 hard rule); nothing here calls
the API directly.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from nba_api.stats.endpoints import (
    leaguedashplayerstats,
    leaguedashplayershotlocations,
    leaguedashptstats,
    synergyplaytypes,
)

from src.cache import fetch_or_load
from src import config


# ---------------------------------------------------------------------------
# Raw pulls (cached). One function per endpoint so the cache keys are stable.
# ---------------------------------------------------------------------------
def _base_params() -> dict:
    return dict(season=config.SEASON, season_type_all_star=config.SEASON_TYPE)


def load_player_stats(measure: str) -> pd.DataFrame:
    """LeagueDashPlayerStats for a given measure type (Base/Advanced/Scoring)."""
    return fetch_or_load(
        leaguedashplayerstats.LeagueDashPlayerStats,
        dict(_base_params(),
             per_mode_detailed="Per100Possessions",
             measure_type_detailed_defense=measure),
    )


def load_shot_locations() -> pd.DataFrame:
    """Shot-zone FGM/FGA by named zone. Returns MultiIndex columns."""
    return fetch_or_load(
        leaguedashplayershotlocations.LeagueDashPlayerShotLocations,
        dict(_base_params(), per_mode_detailed="Totals", distance_range="By Zone"),
    )


def load_passing() -> pd.DataFrame:
    """Passing tracking (PerGame — normalized to per-100 downstream)."""
    return fetch_or_load(
        leaguedashptstats.LeagueDashPtStats,
        dict(_base_params(), per_mode_simple="PerGame",
             player_or_team="Player", pt_measure_type="Passing"),
    )


def load_play_type(play_type: str) -> pd.DataFrame:
    """One Synergy offensive play type. POSS_PCT is the frequency we want."""
    return fetch_or_load(
        synergyplaytypes.SynergyPlayTypes,
        dict(_base_params(), per_mode_simple="PerGame",
             player_or_team_abbreviation="P", type_grouping_nullable="offensive",
             play_type_nullable=play_type),
    )


# ---------------------------------------------------------------------------
# Feature blocks. Each returns a DataFrame indexed by PLAYER_ID.
# ---------------------------------------------------------------------------
def shared_rates_from_box(box: pd.DataFrame) -> pd.DataFrame:
    """
    The SHARED (NBA ∩ NCAA) box rates — the reusable core.

    `box` must be indexed by PLAYER_ID and expose these columns (any source,
    NBA or college): USG_PCT, AST_PCT, OREB_PCT, DREB_PCT, TM_TOV_PCT, FGA,
    FG3A, FTA. Phase 5 will call this with college totals; the logic below must
    stay source-agnostic.
    """
    fga = box["FGA"].replace(0, np.nan)  # guard: no shots -> undefined ratio
    out = pd.DataFrame(index=box.index)
    out["USG_PCT"] = box["USG_PCT"]
    out["AST_PCT"] = box["AST_PCT"]
    out["OREB_PCT"] = box["OREB_PCT"]
    out["DREB_PCT"] = box["DREB_PCT"]
    out["TM_TOV_PCT"] = box["TM_TOV_PCT"]
    out["FTR"] = box["FTA"] / fga
    out["FG3AR"] = box["FG3A"] / fga
    return out.fillna(0.0)


def _nba_box(advanced: pd.DataFrame, base: pd.DataFrame) -> pd.DataFrame:
    """Assemble the NBA inputs that `shared_rates_from_box` expects."""
    a = advanced.set_index("PLAYER_ID")[
        ["USG_PCT", "AST_PCT", "OREB_PCT", "DREB_PCT", "TM_TOV_PCT"]
    ]
    b = base.set_index("PLAYER_ID")[["FGA", "FG3A", "FTA"]]
    return a.join(b, how="inner")


def zone_shares(shotloc: pd.DataFrame) -> pd.DataFrame:
    """FGA share by shot zone. Five zones summing to ~1 (backcourt dropped)."""
    def col(zone):  # MultiIndex accessor: (zone, 'FGA')
        return shotloc[(zone, "FGA")].astype(float)

    pid = shotloc[("", "PLAYER_ID")].values
    fga = pd.DataFrame({
        "Z_RA": col("Restricted Area"),
        "Z_PAINT": col("In The Paint (Non-RA)"),
        "Z_MID": col("Mid-Range"),
        "Z_CORNER3": col("Left Corner 3") + col("Right Corner 3"),
        "Z_ABOVE3": col("Above the Break 3"),
    })
    fga.index = pid
    fga.index.name = "PLAYER_ID"
    denom = fga.sum(axis=1).replace(0, np.nan)
    return fga.div(denom, axis=0).fillna(0.0)


def scoring_mix(scoring: pd.DataFrame) -> pd.DataFrame:
    """Self-creation + how points are generated."""
    s = scoring.set_index("PLAYER_ID")
    return pd.DataFrame({
        "PCT_AST_FGM": s["PCT_AST_FGM"],       # share of made FG that were assisted
        "PCT_PTS_PAINT": s["PCT_PTS_PAINT"],
        "PCT_PTS_MR": s["PCT_PTS_2PT_MR"],     # mid-range reliance
        "PCT_PTS_FB": s["PCT_PTS_FB"],         # transition/fast-break scoring
    })


def passing_per100(passing: pd.DataFrame, advanced: pd.DataFrame) -> pd.DataFrame:
    """
    Passing volume per-100 possessions + intent ratio.

    The passing endpoint is PerGame only, so we convert volumes to per-100 using
    each player's possessions/game (POSS total / GP from the Advanced table).
    AST_TO_PASS_PCT is already a ratio (assists per pass) — left as-is.
    """
    p = passing.set_index("PLAYER_ID")
    adv = advanced.set_index("PLAYER_ID")
    poss_per_game = adv["POSS"] / adv["GP"]
    scale = 100.0 / poss_per_game
    return pd.DataFrame({
        "PASSES_MADE_P100": p["PASSES_MADE"] * scale,
        "POTENTIAL_AST_P100": p["POTENTIAL_AST"] * scale,
        "AST_TO_PASS_PCT": p["AST_TO_PASS_PCT"],
    })


def _collapse_traded_freq(df: pd.DataFrame) -> pd.Series:
    """
    One frequency per PLAYER_ID, combining a traded player's stints correctly.

    The Synergy tables split a traded player into one row per team. Since
    POSS_PCT = (play-type poss) / (total poss), each stint's total possessions is
    POSS / POSS_PCT, so the combined frequency is:

        sum(POSS_i) / sum(POSS_i / POSS_PCT_i)

    For a single-stint player this returns their POSS_PCT unchanged.
    (Per-game POSS is used as-is; the GP imbalance between stints is a minor
    approximation for the handful of mid-season trades — noted in NOTES.md.)
    """
    d = df[["PLAYER_ID", "POSS", "POSS_PCT"]].copy()
    d = d[d["POSS_PCT"] > 0]                       # guard divide-by-zero
    d["total_poss"] = d["POSS"] / d["POSS_PCT"]
    g = d.groupby("PLAYER_ID")
    return (g["POSS"].sum() / g["total_poss"].sum()).rename(None)


def play_type_freqs() -> pd.DataFrame:
    """Frequency for each of the 10 play types; traded players combined, absence -> 0."""
    cols = {}
    for play_type, name in config.PLAY_TYPES.items():
        cols[name] = _collapse_traded_freq(load_play_type(play_type))
    freqs = pd.DataFrame(cols)          # outer-joined on PLAYER_ID index
    return freqs                        # NaNs filled at assembly time


# ---------------------------------------------------------------------------
# Assembly
# ---------------------------------------------------------------------------
def build_rich_table(universe: pd.DataFrame | None = None) -> pd.DataFrame:
    """
    Build the full player-by-feature table (identity + all RICH_FEATURES),
    restricted to the universe if given. Play-type absences become 0.
    """
    advanced = load_player_stats("Advanced")
    base = load_player_stats("Base")
    scoring = load_player_stats("Scoring")
    shotloc = load_shot_locations()
    passing = load_passing()

    ids = advanced.set_index("PLAYER_ID")[["PLAYER_NAME", "TEAM_ABBREVIATION"]]

    blocks = [
        shared_rates_from_box(_nba_box(advanced, base)),
        zone_shares(shotloc),
        scoring_mix(scoring),
        passing_per100(passing, advanced),
        play_type_freqs(),
    ]
    feats = ids.join(blocks, how="left")

    # Absence in a play-type / zone table means the player never did it -> 0.
    feats[config.RICH_FEATURES] = feats[config.RICH_FEATURES].fillna(0.0)

    feats = feats.reset_index()  # PLAYER_ID back to a column
    if universe is not None:
        feats = feats[feats["PLAYER_ID"].isin(universe["PLAYER_ID"])]

    ordered = config.ID_COLUMNS + config.RICH_FEATURES
    return feats[ordered].sort_values("PLAYER_ID").reset_index(drop=True)


def standardize(df: pd.DataFrame, cols: list[str]) -> tuple[pd.DataFrame, pd.Series, pd.Series]:
    """
    Z-score the given columns (population std). Returns (z-scored frame, mean,
    std). Keep mean/std: Phase 2 needs the mean, Phase 5 needs the same params
    to place prospects on the same scale.
    """
    sub = df[cols].astype(float)
    mean = sub.mean()
    std = sub.std(ddof=0).replace(0, 1.0)  # dead column -> avoid divide-by-zero
    z = (sub - mean) / std
    z.insert(0, "PLAYER_ID", df["PLAYER_ID"].values)
    return z, mean, std
