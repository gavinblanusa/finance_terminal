import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import List, Optional, Tuple

try:
    from app.macro_indicators import compute_sahm_series
    from app.openbb_adapter import fetch_macro_data_openbb
except ModuleNotFoundError:
    from macro_indicators import compute_sahm_series
    from openbb_adapter import fetch_macro_data_openbb

_ROOT = Path(__file__).resolve().parent.parent
MACRO_CACHE_DIR = _ROOT / ".macro_cache"


def _ensure_cache_dir():
    MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)


def clear_macro_file_cache() -> None:
    """Delete JSON files under :file:`.macro_cache/` (next fetch hits FRED)."""
    if not MACRO_CACHE_DIR.is_dir():
        return
    for f in MACRO_CACHE_DIR.glob("*.json"):
        try:
            f.unlink()
        except OSError:
            pass


def macro_file_cache_status(metric_ids: List[str]) -> Tuple[Optional[datetime], int, int]:
    """
    For on-disk FRED JSON caches under ``.macro_cache/`` (24h TTL in fetch).
    Return (newest mtime among present files, count present, count missing),
    in the order of ``metric_ids`` (e.g. registry order).
    """
    if not MACRO_CACHE_DIR.is_dir() or not metric_ids:
        return (None, 0, len(metric_ids))
    present = 0
    newest: Optional[datetime] = None
    for mid in metric_ids:
        p = MACRO_CACHE_DIR / f"{mid}.json"
        if p.is_file():
            try:
                mt = datetime.fromtimestamp(p.stat().st_mtime)
            except OSError:
                pass
            else:
                present += 1
                if newest is None or mt > newest:
                    newest = mt
    return (newest, present, len(metric_ids) - present)


def fetch_macro_indicator(metric: str) -> Optional[pd.DataFrame]:
    """
    Fetches a macroeconomic indicator. Uses file-based caching for 24 hours.
    For ``sahm``, data are derived from the unemployment (UNRATE) series.

    Returns a DataFrame with datetime index and a ``value`` column.
    """
    _ensure_cache_dir()
    cache_file = MACRO_CACHE_DIR / f"{metric}.json"
    
    # Check cache freshness (24 hours)
    if cache_file.exists():
        try:
            mtime = datetime.fromtimestamp(cache_file.stat().st_mtime)
            if datetime.now() - mtime < timedelta(hours=24):
                with open(cache_file, "r") as f:
                    data = json.load(f)
                df = pd.DataFrame(data)
                df['date'] = pd.to_datetime(df['date'])
                df = df.set_index('date')
                return df
        except Exception as e:
            print(f"Error reading macro cache for {metric}: {e}")

    # Fetch fresh data
    if metric == "sahm":
        u = fetch_macro_data_openbb("unemployment")
        if u is None or u.empty:
            return None
        df = compute_sahm_series(u)
    else:
        df = fetch_macro_data_openbb(metric)

    if df is not None and not df.empty:
        try:
            # Save to cache
            df_reset = df.reset_index()
            # Convert datetime to string for JSON serialization
            df_reset['date'] = df_reset['date'].dt.strftime('%Y-%m-%d')
            data = df_reset.to_dict(orient='records')
            with open(cache_file, "w") as f:
                json.dump(data, f)
        except Exception as e:
            print(f"Error writing macro cache for {metric}: {e}")
            
    return df
