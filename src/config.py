"""
Project configuration — single source of truth for the choices that define
the dataset. Kept here (not hardcoded in scripts) so every phase reads the
same values, and so decisions are documented in one place.
"""

# --- Phase 0.4: the player universe -----------------------------------------
# The season we analyze first. Format is the NBA API's "YYYY-YY".
SEASON = "2024-25"

# We study the regular season (playoff samples are small and role-distorting).
SEASON_TYPE = "Regular Season"

# Minimum TOTAL minutes played to be "in" the universe.
# Chosen deliberately (see NOTES.md): 1000 min keeps genuine rotation players
# and interesting specialists, while giving enough possessions for play-type
# frequencies to stabilize. Tradeoff accepted: drops deep-bench / small-sample
# players whose style vectors would be too noisy to trust.
MIN_MINUTES = 1000

# ============================================================================
# Phase 1 — Feature spaces
# ============================================================================
# "The feature set defines the space." Two lists, kept parallel (design lock-in):
#   RICH   — every "how they play" signal available for the NBA (play types,
#            tracking, shot zones, scoring mix). Powers the flagship map.
#   SHARED — only what free college box data can ALSO produce, so NCAA prospects
#            can be projected into the SAME space in Phase 5.
#
# All feature columns use our OWN internal names (defined here), produced by
# src/features.py from the raw endpoint tables. Prospect vectors in Phase 5 must
# be built to these same names through the same code path.

# --- The shared (NBA ∩ NCAA) box rates --------------------------------------
# Strictly derivable from a box score, so real college data can reproduce them.
# What the shared space LOSES (1.2 checkpoint): all texture — play types, passing
# tracking, shot zones, self-creation. Anyone with the same 7 rates collapses to
# the same point (a rim-runner and a post scorer become indistinguishable).
SHARED_FEATURES = [
    "USG_PCT",      # usage: share of team possessions a player uses
    "AST_PCT",      # assist rate: share of teammate FGs assisted while on floor
    "OREB_PCT",     # offensive rebound rate
    "DREB_PCT",     # defensive rebound rate
    "TM_TOV_PCT",   # turnover rate (on a percentage scale, e.g. 5.7 = 5.7%)
    "FTR",          # free-throw rate  = FTA / FGA   (derived)
    "FG3AR",        # three-point rate = FG3A / FGA  (derived)
]

# --- Rich-only additions (NBA feasible, NOT college-derivable) --------------
# Shot-zone distribution: FGA share by zone (sums to ~1 across the five).
ZONE_FEATURES = ["Z_RA", "Z_PAINT", "Z_MID", "Z_CORNER3", "Z_ABOVE3"]

# Scoring mix + self-creation (Scoring measure type).
SCORING_FEATURES = ["PCT_AST_FGM", "PCT_PTS_PAINT", "PCT_PTS_MR", "PCT_PTS_FB"]

# Passing tendencies (per-100 possessions where a volume; intent ratio as-is).
PASSING_FEATURES = ["PASSES_MADE_P100", "POTENTIAL_AST_P100", "AST_TO_PASS_PCT"]

# Play-type frequencies (Synergy POSS_PCT). Absence in a table => 0 frequency.
# Maps the API's play_type value -> our internal column name.
PLAY_TYPES = {
    "Isolation":     "PT_ISO",
    "Transition":    "PT_TRANSITION",
    "PRBallHandler": "PT_PRBH",
    "PRRollman":     "PT_PRRM",
    "Postup":        "PT_POSTUP",
    "Spotup":        "PT_SPOTUP",
    "Handoff":       "PT_HANDOFF",
    "Cut":           "PT_CUT",
    "OffScreen":     "PT_OFFSCREEN",
    "OffRebound":    "PT_PUTBACK",
}
PLAYTYPE_FEATURES = list(PLAY_TYPES.values())

# The full rich space = shared box rates + everything NBA-only.
RICH_FEATURES = (
    SHARED_FEATURES
    + ZONE_FEATURES
    + SCORING_FEATURES
    + PASSING_FEATURES
    + PLAYTYPE_FEATURES
)

# Identity columns carried alongside the feature matrix (never fed to models).
ID_COLUMNS = ["PLAYER_ID", "PLAYER_NAME", "TEAM_ABBREVIATION"]
