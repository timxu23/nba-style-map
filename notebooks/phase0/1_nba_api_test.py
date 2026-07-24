"""
Phase 0.2 — first contact with the NBA API.

Exploration script (not reusable logic — lives in notebooks/, not src/).
Purpose: touch the raw API ONCE, before any cache exists, to feel:
  1. the API speaks in integer IDs, not names
  2. one endpoint returns MULTIPLE tables (result sets)
  3. the per-call latency that motivates the cache

Run:  python3 notebooks/phase0/1_nba_api_test.py
"""
import time
from nba_api.stats.static import players
from nba_api.stats.endpoints import commonplayerinfo

# 1. Resolve a name -> player_id. This static lookup is LOCAL (bundled json),
#    no network call. Note the API speaks in integer IDs, not names.
jokic = players.find_players_by_full_name("Nikola Jokic")[0]
print("Static lookup (no network):", jokic)
pid = jokic["id"]

# 2. One real API call. Time it so we feel the latency.
t0 = time.time()
info = commonplayerinfo.CommonPlayerInfo(player_id=pid)
elapsed = time.time() - t0
print(f"\nAPI call took {elapsed:.2f}s")

# 3. The raw response is a set of named 'result sets', each a table.
#    This is the shape lesson: one endpoint returns MULTIPLE tables.
raw = info.get_dict()
print("\nResult set names returned by this ONE endpoint:")
for rs in raw["resultSets"]:
    print("   -", rs["name"], f"({len(rs['rowSet'])} rows x {len(rs['headers'])} cols)")

# 4. Same data as a DataFrame (nba_api gives us this convenience).
df = info.get_data_frames()[0]
print("\nFirst result set as a DataFrame — a few columns:")
print(df[["PERSON_ID", "DISPLAY_FIRST_LAST", "POSITION", "TEAM_NAME", "HEIGHT"]].to_string(index=False))
