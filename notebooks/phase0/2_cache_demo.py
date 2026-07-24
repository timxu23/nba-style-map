"""
Phase 0.3 — prove the cache layer works.

Demonstrates the whole point of fetch_or_load:
  - 1st call  = MISS: sleeps, hits the API, writes parquet + json
  - 2nd call  = HIT : instant, no sleep, no network
  - analysis code (here) never touches nba_api directly, only the cache

Run:  python3 notebooks/phase0/2_cache_demo.py
"""
import sys
import time
from pathlib import Path

# Make `src` importable no matter where we run from (the notebook-import fix).
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from nba_api.stats.endpoints import commonplayerinfo
from src.cache import fetch_or_load, RAW_DIR

params = {"player_id": 203999}  # Nikola Jokić

print("=== First call (expect MISS) ===")
t0 = time.time()
df1 = fetch_or_load(commonplayerinfo.CommonPlayerInfo, params)
print(f"  elapsed: {time.time() - t0:.2f}s\n")

print("=== Second call (expect HIT, instant) ===")
t0 = time.time()
df2 = fetch_or_load(commonplayerinfo.CommonPlayerInfo, params)
print(f"  elapsed: {time.time() - t0:.2f}s\n")

assert df1.equals(df2), "cache returned different data!"
print("Cache round-trips identical data:", df1.equals(df2))

print("\nFiles now in data/raw/:")
for p in sorted(RAW_DIR.iterdir()):
    print("   -", p.name)
