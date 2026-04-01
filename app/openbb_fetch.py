"""
Shared OpenBB fetch kernel: lazy obb, provider chain, timeouts, structured logging.

Callers in openbb_adapter normalize OpenBB results to app shapes.
"""

from __future__ import annotations

import logging
import os
import time
from concurrent.futures import ThreadPoolExecutor
from concurrent.futures import TimeoutError as FuturesTimeout
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Optional, Tuple

# Project root (app lives in repo/app/)
_ROOT = Path(__file__).resolve().parent.parent

try:
    from dotenv import load_dotenv

    _env_path = _ROOT / ".env"
    if _env_path.exists():
        load_dotenv(_env_path)
except ImportError:
    pass

if os.environ.get("MASSIVE_API_KEY") and not os.environ.get("POLYGON_API_KEY"):
    os.environ["POLYGON_API_KEY"] = os.environ["MASSIVE_API_KEY"]

# OpenBB’s credential loader maps env vars ending in API_KEY to provider keys; FRED_API_KEY → fred_api_key.

USE_OPENBB = os.environ.get("USE_OPENBB", "true").lower() in ("true", "1", "yes")

OPENBB_REQUEST_TIMEOUT_SEC = float(os.environ.get("OPENBB_REQUEST_TIMEOUT_SEC", "30"))

_obb: Any = None

logger = logging.getLogger("gft.openbb")


@dataclass(frozen=True)
class OpenBBFetchResult:
    ok: bool
    data: Any
    provider_used: Optional[str]
    elapsed_ms: int
    error_kind: Optional[str]


def _get_obb():
    """Lazy load OpenBB; returns None when disabled or not installed."""
    global _obb
    if _obb is not None:
        return _obb
    if not USE_OPENBB:
        return None
    try:
        from openbb import obb

        _obb = obb
        return _obb
    except ImportError:
        return None


def run_provider_chain(
    dataset_id: str,
    symbol: Optional[str],
    providers: Tuple[str, ...],
    invoke: Callable[[Any, str], Any],
) -> OpenBBFetchResult:
    """
    Try OpenBB providers in order. invoke(obb, provider) returns the SDK result object
    (with .results and .to_df()) or raises.

    Uses a thread + timeout so a stuck provider does not block forever.
    """
    obb = _get_obb()
    if obb is None:
        logger.info(
            "openbb fetch skip_disabled dataset_id=%s symbol=%s outcome=skip_disabled",
            dataset_id,
            symbol or "",
        )
        return OpenBBFetchResult(False, None, None, 0, "skip_disabled")

    last_kind: Optional[str] = None

    for provider in providers:
        t0 = time.perf_counter()

        def _call() -> Any:
            return invoke(obb, provider)

        try:
            with ThreadPoolExecutor(max_workers=1) as ex:
                fut = ex.submit(_call)
                raw = fut.result(timeout=OPENBB_REQUEST_TIMEOUT_SEC)
        except FuturesTimeout:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            last_kind = "timeout"
            logger.info(
                "openbb fetch dataset_id=%s symbol=%s provider=%s elapsed_ms=%s outcome=timeout",
                dataset_id,
                symbol or "",
                provider,
                elapsed_ms,
            )
            continue
        except Exception as e:
            elapsed_ms = int((time.perf_counter() - t0) * 1000)
            last_kind = "exception"
            logger.debug(
                "openbb fetch dataset_id=%s symbol=%s provider=%s elapsed_ms=%s err=%s",
                dataset_id,
                symbol or "",
                provider,
                elapsed_ms,
                e,
            )
            continue

        elapsed_ms = int((time.perf_counter() - t0) * 1000)

        if raw is None or not getattr(raw, "results", None):
            last_kind = "empty"
            logger.info(
                "openbb fetch dataset_id=%s symbol=%s provider=%s elapsed_ms=%s outcome=empty",
                dataset_id,
                symbol or "",
                provider,
                elapsed_ms,
            )
            continue

        logger.info(
            "openbb fetch dataset_id=%s symbol=%s provider=%s elapsed_ms=%s outcome=ok",
            dataset_id,
            symbol or "",
            provider,
            elapsed_ms,
        )
        return OpenBBFetchResult(True, raw, provider, elapsed_ms, None)

    logger.info(
        "openbb fetch dataset_id=%s symbol=%s provider=%s elapsed_ms=0 outcome=fallback error_kind=%s",
        dataset_id,
        symbol or "",
        "none",
        last_kind or "no_provider",
    )
    return OpenBBFetchResult(False, None, None, 0, last_kind or "no_provider")
