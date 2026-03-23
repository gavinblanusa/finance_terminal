import os
import json
from pathlib import Path
from datetime import datetime, timedelta
import pandas as pd
from typing import Optional

try:
    from app.openbb_adapter import fetch_macro_data_openbb
except ModuleNotFoundError:
    from openbb_adapter import fetch_macro_data_openbb

_ROOT = Path(__file__).resolve().parent.parent
MACRO_CACHE_DIR = _ROOT / ".macro_cache"

def _ensure_cache_dir():
    MACRO_CACHE_DIR.mkdir(parents=True, exist_ok=True)

def fetch_macro_indicator(metric: str) -> Optional[pd.DataFrame]:
    """
    Fetches a macroeconomic indicator. Uses file-based caching for 24 hours.
    metric: 'gdp', 'cpi', 'unemployment', 'treasury_10y', 'm2', 'pmi', 'retail_sales', 'consumer_sentiment'
    Returns a DataFrame with the structure expected by the UI.
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
