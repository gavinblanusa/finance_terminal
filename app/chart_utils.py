"""
Helpers for streamlit-lightweight-charts: OHLCV DataFrame to component format.
"""

from typing import List, Tuple, Any, Optional

import pandas as pd


def ohlcv_to_lightweight_charts_data(
    df: pd.DataFrame,
) -> Tuple[List[dict], List[dict]]:
    """
    Convert OHLCV DataFrame to candlestick and volume lists for renderLightweightCharts.

    Expects df with datetime index and columns: Open, High, Low, Close, Volume.
    Returns (candles, volume) where candles are { time, open, high, low, close }
    and volume are { time, value }.
    """
    if df is None or df.empty:
        return [], []

    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return [], []

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index

    times = out.index.strftime("%Y-%m-%d").tolist()
    candles = []
    volume = []
    for i, t in enumerate(times):
        row = out.iloc[i]
        o = float(row["Open"])
        h = float(row["High"])
        l_ = float(row["Low"])
        c = float(row["Close"])
        v = int(row["Volume"]) if pd.notna(row["Volume"]) else 0
        candles.append({"time": t, "open": o, "high": h, "low": l_, "close": c})
        volume.append({"time": t, "value": v})
    return candles, volume


def df_to_technical_chart_data(df: pd.DataFrame) -> dict:
    """
    Convert full technical DataFrame to all series needed for price + RSI charts.

    Expects df with datetime index and at least: Open, High, Low, Close, Volume.
    Optional: RSI_14, SMA_50, SMA_200, BB_Upper, BB_Lower, Signal (BUY, SELL, GOLDEN CROSS, DEATH CROSS).

    Returns dict with: candles, volume, rsi, sma_50, sma_200, bb_upper, bb_lower, markers.
    Missing optional columns yield empty lists.
    """
    result = {
        "candles": [],
        "volume": [],
        "rsi": [],
        "sma_50": [],
        "sma_200": [],
        "bb_upper": [],
        "bb_lower": [],
        "markers": [],
    }
    if df is None or df.empty:
        return result
    required = {"Open", "High", "Low", "Close", "Volume"}
    if not required.issubset(df.columns):
        return result

    out = df.copy()
    if not isinstance(out.index, pd.DatetimeIndex):
        out.index = pd.to_datetime(out.index)
    out.index = out.index.tz_localize(None) if out.index.tz is not None else out.index
    times = out.index.strftime("%Y-%m-%d").tolist()

    for i, t in enumerate(times):
        row = out.iloc[i]
        o = float(row["Open"])
        h = float(row["High"])
        l_ = float(row["Low"])
        c = float(row["Close"])
        v = int(row["Volume"]) if pd.notna(row["Volume"]) else 0
        result["candles"].append({"time": t, "open": o, "high": h, "low": l_, "close": c})
        result["volume"].append({"time": t, "value": v})

    if "RSI_14" in out.columns:
        for i, t in enumerate(times):
            val = out.iloc[i]["RSI_14"]
            if pd.notna(val):
                result["rsi"].append({"time": t, "value": round(float(val), 2)})
    if "SMA_50" in out.columns:
        for i, t in enumerate(times):
            val = out.iloc[i]["SMA_50"]
            if pd.notna(val):
                result["sma_50"].append({"time": t, "value": float(val)})
    if "SMA_200" in out.columns:
        for i, t in enumerate(times):
            val = out.iloc[i]["SMA_200"]
            if pd.notna(val):
                result["sma_200"].append({"time": t, "value": float(val)})
    if "BB_Upper" in out.columns:
        for i, t in enumerate(times):
            val = out.iloc[i]["BB_Upper"]
            if pd.notna(val):
                result["bb_upper"].append({"time": t, "value": float(val)})
    if "BB_Lower" in out.columns:
        for i, t in enumerate(times):
            val = out.iloc[i]["BB_Lower"]
            if pd.notna(val):
                result["bb_lower"].append({"time": t, "value": float(val)})

    if "Signal" in out.columns:
        gain_color = "#00ff41"
        loss_color = "#ff073a"
        for i, t in enumerate(times):
            sig = out.iloc[i].get("Signal")
            if not isinstance(sig, str) or not sig.strip():
                continue
            sig = sig.strip().upper()
            if sig == "BUY":
                result["markers"].append({
                    "time": t,
                    "position": "belowBar",
                    "shape": "arrowUp",
                    "color": gain_color,
                    "text": "",
                    "size": 2,
                })
            elif sig == "SELL":
                result["markers"].append({
                    "time": t,
                    "position": "aboveBar",
                    "shape": "arrowDown",
                    "color": loss_color,
                    "text": "",
                    "size": 2,
                })
            elif sig == "GOLDEN CROSS":
                result["markers"].append({
                    "time": t,
                    "position": "belowBar",
                    "shape": "circle",
                    "color": "#ffd700",
                    "text": "",
                    "size": 1,
                })
            elif sig == "DEATH CROSS":
                result["markers"].append({
                    "time": t,
                    "position": "aboveBar",
                    "shape": "circle",
                    "color": "#8b0000",
                    "text": "",
                    "size": 1,
                })

    return result


def build_technical_chart_config(
    ticker: str,
    candles: List[dict],
    volume: List[dict],
    dark_theme: bool = True,
    rsi: Optional[List[dict]] = None,
    sma_50: Optional[List[dict]] = None,
    sma_200: Optional[List[dict]] = None,
    bb_upper: Optional[List[dict]] = None,
    bb_lower: Optional[List[dict]] = None,
    markers: Optional[List[dict]] = None,
) -> List[dict]:
    """
    Build chart config for renderLightweightCharts: price chart (candlestick + volume + support lines + markers)
    and optionally a second chart for RSI below. Zoom and double-click reset are enabled.

    Returns a list of one or two chart dicts.
    """
    if dark_theme:
        bg_color = "#0d1117"
        text_color = "#c9d1d9"
        grid_color = "rgba(42, 46, 57, 0.6)"
        gain_color = "#00ff41"
        loss_color = "#ff073a"
        sma_50_color = "#58a6ff"
        sma_200_color = "#f78166"
        bb_color = "#8b949e"
        rsi_color = "#d2a8ff"
    else:
        bg_color = "white"
        text_color = "black"
        grid_color = "rgba(197, 203, 206, 0.5)"
        gain_color = "#26a69a"
        loss_color = "#ef5350"
        sma_50_color = "#2196F3"
        sma_200_color = "#ff9800"
        bb_color = "#9e9e9e"
        rsi_color = "#9c27b0"

    # Zoom: scroll to zoom; double-click to reset (time and price)
    handle_scale = {
        "axisDoubleClickReset": {"time": True, "price": True},
        "pinch": True,
        "mouseWheel": True,
        "axisPressedMouseMove": True,
    }

    chart_options: dict[str, Any] = {
        "height": 420,
        "rightPriceScale": {
            "scaleMargins": {"top": 0.05, "bottom": 0.05},
            "borderVisible": False,
        },
        "overlayPriceScales": {
            "scaleMargins": {"top": 0.7, "bottom": 0},
        },
        "layout": {
            "background": {"type": "solid", "color": bg_color},
            "textColor": text_color,
        },
        "grid": {
            "vertLines": {"color": grid_color},
            "horzLines": {"color": grid_color},
        },
        "timeScale": {
            "borderColor": grid_color,
            "timeVisible": True,
            "secondsVisible": False,
        },
        "handleScale": handle_scale,
        "watermark": {
            "visible": True,
            "fontSize": 22,
            "horzAlign": "center",
            "vertAlign": "center",
            "color": "rgba(201, 209, 217, 0.3)",
            "text": ticker,
        },
    }

    candlestick_opts: dict[str, Any] = {
        "upColor": gain_color,
        "downColor": loss_color,
        "borderVisible": False,
        "wickUpColor": gain_color,
        "wickDownColor": loss_color,
    }
    candlestick_series = {
        "type": "Candlestick",
        "data": candles,
        "options": candlestick_opts,
    }
    if markers:
        candlestick_series["markers"] = markers

    series: List[dict] = [candlestick_series]

    if sma_50:
        series.append({
            "type": "Line",
            "data": sma_50,
            "options": {"color": sma_50_color, "lineWidth": 2},
        })
    if sma_200:
        series.append({
            "type": "Line",
            "data": sma_200,
            "options": {"color": sma_200_color, "lineWidth": 2},
        })
    if bb_upper:
        series.append({
            "type": "Line",
            "data": bb_upper,
            "options": {"color": bb_color, "lineWidth": 1, "lineStyle": 1},
        })
    if bb_lower:
        series.append({
            "type": "Line",
            "data": bb_lower,
            "options": {"color": bb_color, "lineWidth": 1, "lineStyle": 1},
        })

    series.append({
        "type": "Histogram",
        "data": volume,
        "options": {
            "color": "#26a69a",
            "priceFormat": {"type": "volume"},
            "priceScaleId": "",
        },
        "priceScale": {
            "scaleMargins": {"top": 0.7, "bottom": 0},
        },
    })

    charts_out: List[dict] = [
        {"chart": chart_options, "series": series},
    ]

    if rsi:
        # Fix RSI y-axis at 0–100 (RSI is bounded). Use autoScale: false and an invisible
        # anchor series so the scale always shows 0 to 100.
        rsi_times = [x["time"] for x in rsi]
        rsi_anchor_data = []
        if len(rsi_times) >= 2:
            rsi_anchor_data = [
                {"time": rsi_times[0], "value": 0},
                {"time": rsi_times[-1], "value": 100},
            ]
        rsi_chart_options: dict[str, Any] = {
            "height": 200,
            "layout": {
                "background": {"type": "solid", "color": bg_color},
                "textColor": text_color,
            },
            "grid": {
                "vertLines": {"color": grid_color},
                "horzLines": {"color": grid_color},
            },
            "timeScale": {
                "borderColor": grid_color,
                "visible": True,
                "timeVisible": True,
                "secondsVisible": False,
            },
            "handleScale": handle_scale,
            "rightPriceScale": {
                "scaleMargins": {"top": 0.05, "bottom": 0.05},
                "borderVisible": True,
                "borderColor": grid_color,
            },
            "watermark": {
                "visible": True,
                "fontSize": 14,
                "horzAlign": "left",
                "vertAlign": "top",
                "color": "rgba(210, 168, 255, 0.5)",
                "text": "RSI (14)",
            },
        }
        rsi_series = {
            "type": "Line",
            "data": rsi,
            "options": {"color": rsi_color, "lineWidth": 2},
        }
        rsi_series_list: List[dict] = [rsi_series]
        # Dotted reference lines: overbought (70) and oversold (30)
        rsi_overbought = [{"time": t, "value": 70} for t in rsi_times]
        rsi_oversold = [{"time": t, "value": 30} for t in rsi_times]
        rsi_series_list.append({
            "type": "Line",
            "data": rsi_overbought,
            "options": {
                "color": "#ff6b6b",
                "lineWidth": 1,
                "lineStyle": 1,
                "lastValueVisible": False,
                "priceLineVisible": False,
            },
        })
        rsi_series_list.append({
            "type": "Line",
            "data": rsi_oversold,
            "options": {
                "color": "#69db7c",
                "lineWidth": 1,
                "lineStyle": 1,
                "lastValueVisible": False,
                "priceLineVisible": False,
            },
        })
        # Invisible anchor series so the price scale range is forced to 0–100
        if rsi_anchor_data:
            rsi_series_list.append({
                "type": "Line",
                "data": rsi_anchor_data,
                "options": {
                    "color": "rgba(0,0,0,0)",
                    "lineWidth": 0,
                    "lastValueVisible": False,
                    "priceLineVisible": False,
                },
            })
        charts_out.append({
            "chart": rsi_chart_options,
            "series": rsi_series_list,
        })

    return charts_out
