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

# --- Feature lists (Phase 1) will be added here later -----------------------
# RICH_FEATURES  = [...]   # NBA-only: play types, tracking, passing
# SHARED_FEATURES = [...]  # NBA ∩ NCAA: derivable from free college data
