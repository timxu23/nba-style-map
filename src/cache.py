"""
THE HARD RULE (from the roadmap): analysis code never calls the NBA API
directly. It only ever calls `fetch_or_load`. Everything downstream reads
from disk; the network is touched exactly once per unique request, ever.

Why this matters (answer it yourself in NOTES.md too):
  - Reproducibility: a raw pull saved to disk is a fixed input. Re-running
    analysis gives identical results without depending on a live server.
  - Speed: the API is slow and rate-limited. Cached reads are instant.
  - Etiquette: the mandatory sleep lives INSIDE the miss branch, so cached
    reads never pause and you physically cannot hammer the API by re-running.
  - Decoupling: a bug in feature code is fixed by re-running analysis, never
    by re-fetching. raw/ is immutable; processed/ is disposable.
"""
from __future__ import annotations

import hashlib
import json
import time
from datetime import datetime, timezone
from pathlib import Path

import pandas as pd

# Resolve paths relative to THIS file, not the current working directory,
# so imports work identically from a notebook, a script, or a test.
PROJECT_ROOT = Path(__file__).resolve().parents[1]
RAW_DIR = PROJECT_ROOT / "data" / "raw"

# Default pause (seconds) between live API calls. stats.nba.com rate-limits
# aggressively; ~0.6s is a polite floor. Only applied on a cache MISS.
DEFAULT_PAUSE = 0.6


def _slugify(text: str) -> str:
    """Make a string safe and readable as a filename fragment."""
    keep = [c if (c.isalnum() or c in "-_") else "-" for c in str(text)]
    return "".join(keep).strip("-").lower()


def _cache_key(endpoint_name: str, params: dict, df_index: int) -> str:
    """
    Deterministic cache filename stem for a request.

    Combines a HUMAN-READABLE slug (so data/raw/ is browsable) with a short
    HASH of the full params (so distinct requests can never collide onto the
    same file, even after slug sanitization loses information).
    """
    # Readable part: endpoint + a few sorted key=value pairs.
    parts = [_slugify(k) + "-" + _slugify(v) for k, v in sorted(params.items())]
    readable = _slugify(endpoint_name)
    if parts:
        readable += "__" + "__".join(parts)

    # Uniqueness part: hash of the canonical (sorted) params + df index.
    canonical = json.dumps(params, sort_keys=True, default=str) + f"|rs{df_index}"
    digest = hashlib.md5(canonical.encode()).hexdigest()[:8]

    stem = f"{readable}__rs{df_index}__{digest}"
    # Guard against pathologically long names on some filesystems.
    return stem[:180]


def fetch_or_load(
    endpoint_cls,
    params: dict,
    df_index: int = 0,
    *,
    pause: float = DEFAULT_PAUSE,
    cache_dir: Path = RAW_DIR,
    verbose: bool = True,
) -> pd.DataFrame:
    """
    Return one result-set table for an nba_api endpoint, from cache if present.

    Parameters
    ----------
    endpoint_cls : an nba_api endpoint CLASS (e.g. commonplayerinfo.CommonPlayerInfo).
        Passed as the class, not an instance — we only construct it (which is
        what fires the network call) on a cache miss.
    params : dict
        Keyword args for the endpoint (e.g. {"player_id": 203999}). These are
        also what the cache is keyed on.
    df_index : int, default 0
        Which result set to keep. One endpoint can return several tables.
    pause : float
        Seconds to sleep BEFORE a live call. Applied only on a miss.
    cache_dir : Path
        Where parquet + sidecar json live. Defaults to data/raw/.
    verbose : bool
        Print HIT/MISS so you can see the cache working.

    Returns
    -------
    pandas.DataFrame
    """
    cache_dir.mkdir(parents=True, exist_ok=True)
    stem = _cache_key(endpoint_cls.__name__, params, df_index)
    parquet_path = cache_dir / f"{stem}.parquet"
    meta_path = cache_dir / f"{stem}.json"

    # --- HIT: serve from disk, no network, no sleep -----------------------
    if parquet_path.exists():
        if verbose:
            print(f"[cache HIT ] {parquet_path.name}")
        return pd.read_parquet(parquet_path)

    # --- MISS: be polite, then hit the API --------------------------------
    if verbose:
        print(f"[cache MISS] {endpoint_cls.__name__} {params} -> sleeping {pause}s then fetching")
    time.sleep(pause)  # etiquette: only ever sleep when we truly call out

    endpoint = endpoint_cls(**params)          # <-- the network call happens here
    df = endpoint.get_data_frames()[df_index]

    # Persist the raw table + a sidecar documenting what it is / when pulled.
    df.to_parquet(parquet_path)
    meta = {
        "endpoint": endpoint_cls.__name__,
        "params": params,
        "df_index": df_index,
        "n_rows": int(df.shape[0]),
        "n_cols": int(df.shape[1]),
        "columns": list(df.columns),
        "fetched_at_utc": datetime.now(timezone.utc).isoformat(),
    }
    meta_path.write_text(json.dumps(meta, indent=2, default=str))

    return df
