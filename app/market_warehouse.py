"""
Optional OHLCV load from a local market-data-warehouse (Parquet bronze or DuckDB).

See docs/plans/PLAN-mdw-warehouse-adapter.md and README (optional MDW env vars).
Uses split-adjusted close (adj_close) as Close to align with typical yfinance history.
"""

from __future__ import annotations

import concurrent.futures
import os
from datetime import date, timedelta
from pathlib import Path
from typing import Callable, Optional, TypeVar

import pandas as pd

T = TypeVar("T")

ENV_DUCKDB = "GFT_MARKET_WAREHOUSE_DUCKDB"
ENV_BRONZE = "GFT_MARKET_WAREHOUSE_BRONZE"
ENV_TIMEOUT = "GFT_MARKET_WAREHOUSE_TIMEOUT_SEC"


def _timeout_sec() -> float:
    raw = os.environ.get(ENV_TIMEOUT, "5").strip()
    try:
        v = float(raw)
        return max(0.5, min(v, 60.0))
    except ValueError:
        return 5.0


def _run_with_timeout(fn: Callable[[], T], timeout_sec: float) -> T:
    with concurrent.futures.ThreadPoolExecutor(max_workers=1) as pool:
        fut = pool.submit(fn)
        return fut.result(timeout=timeout_sec)


def _resolve_equity_bronze_root(raw: str) -> Path:
    """Resolve user path to the MDW equity bronze directory (…/asset_class=equity)."""
    p = Path(raw).expanduser().resolve()
    if not p.exists():
        return p
    if p.name == "asset_class=equity":
        return p
    candidate = p / "asset_class=equity"
    if candidate.is_dir():
        return candidate
    return p


def _normalize_mdw_frame(df: pd.DataFrame) -> Optional[pd.DataFrame]:
    """Build OHLCV DataFrame matching market_data.fetch_ohlcv output shape."""
    if df is None or df.empty:
        return None
    colmap = {str(c).lower(): c for c in df.columns}
    need = ("open", "high", "low", "volume")
    if not all(k in colmap for k in need):
        return None
    date_col = None
    for key in ("trade_date", "date", "datetime"):
        if key in colmap:
            date_col = colmap[key]
            break
    if date_col is None:
        if isinstance(df.index, pd.DatetimeIndex) or (
            df.index.name and str(df.index.name).lower() in ("trade_date", "date")
        ):
            work = df.copy()
            if not isinstance(work.index, pd.DatetimeIndex):
                work.index = pd.to_datetime(work.index, errors="coerce")
        else:
            return None
    else:
        work = df.copy()
        work[date_col] = pd.to_datetime(work[date_col], errors="coerce")
        work = work.set_index(date_col)

    close_src = None
    if "adj_close" in colmap:
        close_src = colmap["adj_close"]
    elif "close" in colmap:
        close_src = colmap["close"]
    else:
        return None

    out = pd.DataFrame(
        {
            "Open": pd.to_numeric(work[colmap["open"]], errors="coerce"),
            "High": pd.to_numeric(work[colmap["high"]], errors="coerce"),
            "Low": pd.to_numeric(work[colmap["low"]], errors="coerce"),
            "Close": pd.to_numeric(work[close_src], errors="coerce"),
            "Volume": pd.to_numeric(work[colmap["volume"]], errors="coerce"),
        },
        index=work.index,
    )
    out = out.dropna(how="all")
    out.sort_index(inplace=True)
    out = out[out.index.notna()]
    out = out[~out.index.duplicated(keep="last")]
    if out.empty or len(out) < 1:
        return None
    if getattr(out.index, "tz", None) is not None:
        out.index = out.index.tz_localize(None)
    return out


def _filter_date_range(df: pd.DataFrame, start_date: date, end_date: date) -> pd.DataFrame:
    if df is None or df.empty:
        return df
    ts_start = pd.Timestamp(start_date)
    ts_end = pd.Timestamp(end_date) + pd.Timedelta(days=1)
    try:
        mask = (df.index >= ts_start) & (df.index < ts_end)
        return df.loc[mask]
    except Exception:
        return df


def _load_from_bronze_parquet(
    ticker: str,
    start_date: date,
    end_date: date,
    bronze_env: str,
) -> Optional[pd.DataFrame]:
    import pyarrow.parquet as pq  # type: ignore[import-untyped]

    root = _resolve_equity_bronze_root(bronze_env)
    path = root / f"symbol={ticker.upper()}" / "data.parquet"
    if not path.is_file():
        return None
    try:
        table = pq.read_table(path)
        df = table.to_pandas()
    except Exception as e:
        print(f"[Warehouse] Parquet read failed for {ticker} ({path}): {e}")
        return None
    out = _normalize_mdw_frame(df)
    if out is None:
        return None
    return _filter_date_range(out, start_date, end_date)


def _load_from_duckdb(
    ticker: str,
    start_date: date,
    end_date: date,
    db_path: str,
) -> Optional[pd.DataFrame]:
    import duckdb  # type: ignore[import-untyped]

    path = str(Path(db_path).expanduser().resolve())
    try:
        con = duckdb.connect(path, read_only=True)
    except Exception as e:
        print(f"[Warehouse] DuckDB connect failed ({path}): {e}")
        return None
    try:
        df = con.execute(
            """
            SELECT e.trade_date, e.open, e.high, e.low, e.adj_close, e.close, e.volume
            FROM md.equities_daily e
            JOIN md.symbols s ON e.symbol_id = s.symbol_id
            WHERE upper(s.symbol) = ?
              AND e.trade_date >= ?
              AND e.trade_date <= ?
            ORDER BY e.trade_date
            """,
            [ticker.upper(), start_date, end_date],
        ).df()
    except Exception as e:
        print(f"[Warehouse] DuckDB query failed for {ticker}: {e}")
        return None
    finally:
        try:
            con.close()
        except Exception:
            pass

    out = _normalize_mdw_frame(df)
    return out


def try_load_ohlcv_from_warehouse(
    ticker: str,
    start_date: date,
    end_date: date,
) -> Optional[pd.DataFrame]:
    """
    Load OHLCV from MDW-style bronze Parquet and/or DuckDB if env is configured.

    Returns None if env unset, optional deps missing, I/O errors, or empty result.
    """
    bronze = os.environ.get(ENV_BRONZE, "").strip()
    duckdb_path = os.environ.get(ENV_DUCKDB, "").strip()
    if not bronze and not duckdb_path:
        return None

    timeout = _timeout_sec()
    t = ticker.upper().strip()

    def _try_bronze() -> Optional[pd.DataFrame]:
        if not bronze:
            return None
        try:
            return _load_from_bronze_parquet(t, start_date, end_date, bronze)
        except ImportError:
            print(
                "[Warehouse] pyarrow required for bronze Parquet reads "
                "(pip install pyarrow)."
            )
            return None

    def _try_duck() -> Optional[pd.DataFrame]:
        if not duckdb_path:
            return None
        try:
            return _load_from_duckdb(t, start_date, end_date, duckdb_path)
        except ImportError:
            print("[Warehouse] duckdb package not installed; skip DuckDB path.")
            return None

    df: Optional[pd.DataFrame] = None
    try:
        if bronze:
            df = _run_with_timeout(_try_bronze, timeout)
        if df is None or df.empty:
            df = _run_with_timeout(_try_duck, timeout)
    except concurrent.futures.TimeoutError:
        print(f"[Warehouse] Timed out after {timeout}s for {t}")
        return None
    except Exception as e:
        print(f"[Warehouse] Unexpected error for {t}: {e}")
        return None

    if df is None or df.empty:
        return None

    try:
        dmax = pd.Timestamp(df.index.max()).date()
        print(
            f"[Warehouse] Loaded {len(df)} rows for {t} "
            f"(max_bar_date={dmax.isoformat()})"
        )
    except Exception:
        print(f"[Warehouse] Loaded {len(df)} rows for {t}")

    return df


def ohlcv_df_sufficient_for_request(
    df: Optional[pd.DataFrame],
    start_date: date,
    end_date: date,
    period_years: int | str | None,
) -> bool:
    """
    Align with OpenBB success path in fetch_ohlcv: need 200+ rows and calendar span.
    """
    if df is None or df.empty or len(df) < 200:
        return False
    try:
        dmin = pd.Timestamp(df.index.min()).date()
        dmax = pd.Timestamp(df.index.max()).date()
    except Exception:
        return False
    if dmax < start_date or dmin > end_date:
        return False
    # Requested window starts at start_date; allow the first trading bar to land
    # just after a weekend/holiday, while still letting IPO-short panels fall through.
    if dmin > start_date + timedelta(days=5):
        return False
    # Drop obviously stale snapshots (no bar within two weeks of requested end).
    if dmax < end_date - timedelta(days=14):
        return False
    return True
