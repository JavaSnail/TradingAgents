from datetime import datetime
from typing import Annotated

import akshare as ak

from .akshare_common import (
    bypass_proxy,
    format_header,
    is_a_share,
    is_index,
    normalize_symbol,
    to_akshare_date,
)
from .symbol_utils import NoMarketDataError

# AkShare 返回的中文列名 → 标准英文列名
_COL_MAP = {
    "日期": "Date",
    "开盘": "Open",
    "最高": "High",
    "最低": "Low",
    "收盘": "Close",
    "成交量": "Volume",
    "成交额": "Amount",
    "涨跌幅": "Change%",
    "换手率": "Turnover%",
}


def get_stock(
    symbol: Annotated[str, "股票代码，如 002027.SZ 或 002027"],
    start_date: Annotated[str, "开始日期 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期 YYYY-MM-DD"],
) -> str:
    """从 AkShare 获取 A 股日线 OHLCV 数据，返回 CSV 字符串。

    个股（6 位代码）走 ``stock_zh_a_hist``；板块/市场指数（is_index 判定）
    走 ``index_zh_a_hist``。非 A 股触发 NoMarketDataError 使路由回退到 yfinance。

    指数分支失败时不抛 NoMarketDataError 而是返回 NO_DATA 哨兵字符串——
    指数代码在 yfinance 无对应标的，回退只会触发限流退避（约 8 分钟空转），
    故在此直接收口，避免指数输入触达 yfinance。
    """
    if not is_a_share(symbol):
        raise NoMarketDataError(
            symbol, symbol, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(symbol)
    ak_start = to_akshare_date(start_date)
    ak_end = to_akshare_date(end_date)

    index_mode = is_index(symbol)

    try:
        with bypass_proxy():
            if index_mode:
                # 指数无复权概念，不传 adjust；列名与个股接口一致（无成交额/换手率/涨跌幅）
                df = ak.index_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=ak_start,
                    end_date=ak_end,
                )
            else:
                df = ak.stock_zh_a_hist(
                    symbol=code,
                    period="daily",
                    start_date=ak_start,
                    end_date=ak_end,
                    adjust="qfq",
                )
    except TypeError as e:
        # akshare 内部 bug（部分非主流指数如 930908/932051 会抛 TypeError）
        if index_mode:
            return (
                f"NO_DATA_AVAILABLE: index {code} hit AkShare internal error "
                f"({e}). Try a different index code. Do not fabricate values."
            )
        raise NoMarketDataError(symbol, code, f"AkShare request failed: {e}") from e
    except Exception as e:
        if index_mode:
            return (
                f"NO_DATA_AVAILABLE: index {code} request failed ({e}). "
                f"Do not fabricate values."
            )
        raise NoMarketDataError(symbol, code, f"AkShare request failed: {e}") from e

    if df is None or df.empty:
        if index_mode:
            return (
                f"NO_DATA_AVAILABLE: no index data for {code} between "
                f"{start_date} and {end_date}. Do not fabricate values."
            )
        raise NoMarketDataError(
            symbol, code, f"no rows between {start_date} and {end_date}"
        )

    # 只保留需要的列，避免多余中文列干扰 LLM
    keep_cols = [c for c in ["日期", "开盘", "最高", "最低", "收盘", "成交量"] if c in df.columns]
    df = df[keep_cols].copy()
    df.rename(columns=_COL_MAP, inplace=True)

    # 数值列保留 2 位小数
    for col in ["Open", "High", "Low", "Close"]:
        if col in df.columns:
            df[col] = df[col].round(2)

    csv_str = df.to_csv(index=False)
    label = f"{code} (AkShare Index)" if index_mode else f"{code} (AkShare)"
    header = format_header(label, start_date, end_date, len(df))
    return header + csv_str
