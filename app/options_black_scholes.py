"""
European Black–Scholes option values (OVME-lite math layer).

Continuous yield q default 0. Uses scipy.stats.norm. For theory / comparison
only—not a live quoting or execution price.
"""

from __future__ import annotations

from dataclasses import dataclass

import numpy as np
from scipy.stats import norm


@dataclass
class BlackScholesResult:
    call_price: float
    put_price: float
    d1: float
    d2: float
    spot: float
    strike: float
    time_years: float
    rate: float
    dividend_yield: float
    volatility: float


def black_scholes_european(
    spot: float,
    strike: float,
    time_years: float,
    rate: float,
    volatility: float,
    dividend_yield: float = 0.0,
) -> BlackScholesResult:
    """
    spot, strike > 0; time_years >= 0; volatility >= 0; rate and q annualized decimals.
    At T<=0 returns intrinsic via parity; at sigma<=0 returns discounted intrinsic.
    """
    S = float(spot)
    K = float(strike)
    T = max(0.0, float(time_years))
    r = float(rate)
    q = float(dividend_yield)
    sigma = max(0.0, float(volatility))

    if S <= 0 or K <= 0:
        return BlackScholesResult(0.0, 0.0, 0.0, 0.0, S, K, T, r, q, sigma)

    disc_s = np.exp(-q * T) * S
    disc_k = np.exp(-r * T) * K

    if T <= 1e-12:
        call = max(0.0, disc_s - disc_k)
        put = max(0.0, disc_k - disc_s)
        return BlackScholesResult(float(call), float(put), 0.0, 0.0, S, K, T, r, q, sigma)

    if sigma <= 1e-12:
        fwd = np.exp((r - q) * T) * S
        call = np.exp(-r * T) * max(0.0, fwd - K)
        put = np.exp(-r * T) * max(0.0, K - fwd)
        return BlackScholesResult(float(call), float(put), 0.0, 0.0, S, K, T, r, q, sigma)

    sqrt_t = np.sqrt(T)
    d1 = (np.log(S / K) + (r - q + 0.5 * sigma**2) * T) / (sigma * sqrt_t)
    d2 = d1 - sigma * sqrt_t
    call = float(disc_s * norm.cdf(d1) - disc_k * norm.cdf(d2))
    put = float(disc_k * norm.cdf(-d2) - disc_s * norm.cdf(-d1))
    return BlackScholesResult(call, put, float(d1), float(d2), S, K, T, r, q, sigma)