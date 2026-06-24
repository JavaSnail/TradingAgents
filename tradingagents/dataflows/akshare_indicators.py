from datetime import datetime
from typing import Annotated

import akshare as ak
import pandas as pd
from dateutil.relativedelta import relativedelta
from stockstats import wrap

from .akshare_common import bypass_proxy, is_a_share, normalize_symbol, to_akshare_date
from .symbol_utils import NoMarketDataError

# 与 y_finance.py 中保持一致的指标描述
_INDICATOR_DESC = {
    "close_50_sma": (
        "50 SMA: A medium-term trend indicator. "
        "Usage: Identify trend direction and serve as dynamic support/resistance. "
        "Tips: It lags price; combine with faster indicators for timely signals."
    ),
    "close_200_sma": (
        "200 SMA: A long-term trend benchmark. "
        "Usage: Confirm overall market trend and identify golden/death cross setups. "
        "Tips: It reacts slowly; best for strategic trend confirmation."
    ),
    "close_10_ema": (
        "10 EMA: A responsive short-term average. "
        "Usage: Capture quick shifts in momentum and potential entry points. "
        "Tips: Prone to noise in choppy markets."
    ),
    "macd": (
        "MACD: Computes momentum via differences of EMAs. "
        "Usage: Look for crossovers and divergence as signals of trend changes."
    ),
    "macds": "MACD Signal: An EMA smoothing of the MACD line.",
    "macdh": "MACD Histogram: Shows the gap between MACD and its signal.",
    "rsi": (
        "RSI: Measures momentum to flag overbought/oversold conditions. "
        "Usage: Apply 70/30 thresholds and watch for divergence."
    ),
    "boll": "Bollinger Middle: A 20 SMA serving as the basis for Bollinger Bands.",
    "boll_ub": "Bollinger Upper Band: Typically 2 standard deviations above middle.",
    "boll_lb": "Bollinger Lower Band: Typically 2 standard deviations below middle.",
    "atr": "ATR: Averages true range to measure volatility.",
    "vwma": "VWMA: A moving average weighted by volume.",
    "mfi": "MFI: The Money Flow Index uses price and volume to measure buying/selling pressure.",
}

# AkShare 列名 → stockstats 需要的标准列名
_COL_MAP = {
    "日期": "date",
    "开盘": "open",
    "最高": "high",
    "最低": "low",
    "收盘": "close",
    "成交量": "volume",
}


def _fetch_ohlcv_for_indicators(code: str, curr_date: str) -> pd.DataFrame:
    """获取足够长度的 OHLCV 数据用于指标计算（回溯 300 天）。

    注意：此函数不调用 bypass_proxy()——调用方 get_indicator() 已包裹整个上下文。
    """
    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = curr_dt - relativedelta(days=300)
    ak_start = start_dt.strftime("%Y%m%d")
    ak_end = curr_dt.strftime("%Y%m%d")

    df = ak.stock_zh_a_hist(
        symbol=code,
        period="daily",
        start_date=ak_start,
        end_date=ak_end,
        adjust="qfq",
    )

    if df is None or df.empty:
        raise NoMarketDataError(code, code, f"no OHLCV data up to {curr_date}")

    keep = [c for c in ["日期", "开盘", "最高", "最低", "收盘", "成交量"] if c in df.columns]
    df = df[keep].copy()
    df.rename(columns=_COL_MAP, inplace=True)
    df["date"] = pd.to_datetime(df["date"])
    df = df.sort_values("date").reset_index(drop=True)
    return df


def get_indicator(
    symbol: Annotated[str, "股票代码，如 002027.SZ"],
    indicator: Annotated[str, "技术指标名称，如 close_50_sma"],
    curr_date: Annotated[str, "当前交易日期 YYYY-MM-DD"],
    look_back_days: Annotated[int, "回溯天数"],
    interval: str = "daily",
    time_period: int = 14,
    series_type: str = "close",
) -> str:
    """用 AkShare 获取 A 股 OHLCV，再用 stockstats 计算技术指标。

    仅支持 A 股，非 A 股触发 NoMarketDataError 使路由回退 yfinance。
    """
    if not is_a_share(symbol):
        raise NoMarketDataError(
            symbol, symbol, "AkShare only supports A-share (6-digit) symbols"
        )

    if indicator not in _INDICATOR_DESC:
        raise ValueError(
            f"Indicator {indicator} is not supported. "
            f"Choose from: {list(_INDICATOR_DESC.keys())}"
        )

    code = normalize_symbol(symbol)

    try:
        with bypass_proxy():
            df = _fetch_ohlcv_for_indicators(code, curr_date)
    except Exception as e:
        raise NoMarketDataError(code, code, f"AkShare request failed: {e}") from e

    stock = wrap(df.copy())
    stock[indicator]  # 触发 stockstats 计算

    curr_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    before_dt = curr_dt - relativedelta(days=look_back_days)

    lines = []
    for _, row in stock.iterrows():
        row_date = row["date"] if hasattr(row["date"], "strftime") else pd.Timestamp(row["date"])
        if before_dt <= row_date <= curr_dt:
            val = row.get(indicator)
            date_str = row_date.strftime("%Y-%m-%d")
            if pd.isna(val):
                lines.append(f"{date_str}: N/A")
            else:
                lines.append(f"{date_str}: {round(float(val), 4)}")

    ind_str = "\n".join(reversed(lines)) if lines else "No data available."
    desc = _INDICATOR_DESC.get(indicator, "")
    return (
        f"## {indicator.upper()} values from "
        f"{before_dt.strftime('%Y-%m-%d')} to {curr_date}:\n\n"
        + ind_str
        + f"\n\n{desc}"
    )
