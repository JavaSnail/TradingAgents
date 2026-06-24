from datetime import datetime
from typing import Annotated

import akshare as ak
import pandas as pd

from .akshare_common import bypass_proxy, is_a_share, normalize_symbol, to_sina_symbol
from .symbol_utils import NoMarketDataError


def _filter_by_date(df: pd.DataFrame, curr_date: str, date_col: str) -> pd.DataFrame:
    """过滤掉报告日期晚于 curr_date 的行，防止 look-ahead 偏差。"""
    if curr_date is None or date_col not in df.columns:
        return df
    cutoff = pd.Timestamp(curr_date)
    df = df.copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    return df[df[date_col] <= cutoff].reset_index(drop=True)


def get_fundamentals(
    ticker: Annotated[str, "股票代码，如 002027.SZ"],
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"] = None,
) -> str:
    """获取 A 股基本面概览：市值、估值、核心财务指标。"""
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(ticker)
    lines = [f"# Company Fundamentals for {code} (AkShare)"]
    if curr_date:
        lines.append(f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    lines.append("")

    # 个股基本信息
    try:
        with bypass_proxy():
            info_df = ak.stock_individual_info_em(symbol=code)
        if info_df is not None and not info_df.empty:
            for _, row in info_df.iterrows():
                item = str(row.iloc[0])
                val = str(row.iloc[1])
                if val and val != "nan":
                    lines.append(f"{item}: {val}")
    except Exception:
        pass

    # 财务摘要 —— 新浪来源（API 参数名为 symbol）
    try:
        with bypass_proxy():
            abs_df = ak.stock_financial_abstract(symbol=code)
        if abs_df is not None and not abs_df.empty:
            for col in abs_df.columns:
                val = abs_df[col].iloc[0]
                if pd.notna(val) and str(val) != "nan":
                    lines.append(f"{col}: {val}")
    except Exception:
        pass

    # 财务分析指标（ROE/ROA/毛利率等）—— 东方财富来源
    try:
        with bypass_proxy():
            fin_df = ak.stock_financial_analysis_indicator(symbol=code)
        if fin_df is not None and not fin_df.empty:
            latest = fin_df.iloc[-1]
            for col in fin_df.columns:
                val = latest[col]
                if pd.notna(val) and str(val) != "nan":
                    lines.append(f"{col}: {val}")
    except Exception:
        pass

    # 同花顺财务摘要（更全面的指标）
    try:
        with bypass_proxy():
            ths_df = ak.stock_financial_abstract_ths(symbol=code, indicator="按报告期")
        if ths_df is not None and not ths_df.empty:
            latest = ths_df.iloc[-1]
            for col in ths_df.columns[:20]:  # 限制字段数
                val = latest[col]
                if pd.notna(val) and str(val) != "nan":
                    lines.append(f"{col}: {val}")
    except Exception:
        pass

    if len(lines) <= 2:
        raise NoMarketDataError(ticker, code, "no fundamental fields returned")

    return "\n".join(lines)


def get_balance_sheet(
    ticker: Annotated[str, "股票代码"],
    freq: Annotated[str, "annual 或 quarterly"] = "quarterly",
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"] = None,
) -> str:
    """从新浪财经获取资产负债表。"""
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    sina_code = to_sina_symbol(ticker)
    try:
        with bypass_proxy():
            df = ak.stock_financial_report_sina(stock=sina_code, symbol="资产负债表")
    except Exception as e:
        raise NoMarketDataError(ticker, sina_code, f"balance sheet request failed: {e}") from e

    if df is None or df.empty:
        raise NoMarketDataError(ticker, sina_code, "no balance sheet data")

    # 新浪资产负债表第一列通常是报告期
    date_col = df.columns[0]
    df = _filter_by_date(df, curr_date, date_col)

    if df.empty:
        raise NoMarketDataError(ticker, sina_code, "no balance sheet data before curr_date")

    header = (
        f"# Balance Sheet for {normalize_symbol(ticker)} (AkShare/Sina, {freq})\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + df.to_csv(index=False)


def get_income_statement(
    ticker: Annotated[str, "股票代码"],
    freq: Annotated[str, "annual 或 quarterly"] = "quarterly",
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"] = None,
) -> str:
    """从东方财富获取利润表。"""
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(ticker)
    try:
        with bypass_proxy():
            df = ak.stock_profit_sheet_by_report_em(stock=code)
    except Exception as e:
        raise NoMarketDataError(ticker, code, f"income statement request failed: {e}") from e

    if df is None or df.empty:
        raise NoMarketDataError(ticker, code, "no income statement data")

    date_col = df.columns[0]
    df = _filter_by_date(df, curr_date, date_col)

    if df.empty:
        raise NoMarketDataError(ticker, code, "no income statement data before curr_date")

    header = (
        f"# Income Statement for {code} (AkShare/EM, {freq})\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + df.to_csv(index=False)


def get_cashflow(
    ticker: Annotated[str, "股票代码"],
    freq: Annotated[str, "annual 或 quarterly"] = "quarterly",
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"] = None,
) -> str:
    """从东方财富获取现金流量表。"""
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(ticker)
    try:
        with bypass_proxy():
            df = ak.stock_cash_flow_sheet_by_report_em(stock=code)
    except Exception as e:
        raise NoMarketDataError(ticker, code, f"cash flow request failed: {e}") from e

    if df is None or df.empty:
        raise NoMarketDataError(ticker, code, "no cash flow data")

    date_col = df.columns[0]
    df = _filter_by_date(df, curr_date, date_col)

    if df.empty:
        raise NoMarketDataError(ticker, code, "no cash flow data before curr_date")

    header = (
        f"# Cash Flow for {code} (AkShare/EM, {freq})\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + df.to_csv(index=False)


def get_insider_transactions(
    ticker: Annotated[str, "股票代码"],
) -> str:
    """AkShare 暂无标准内部人交易接口，返回说明文本。"""
    return (
        f"AkShare does not provide insider transaction data for A-shares ({ticker}). "
        "Please refer to the CNINFO or SZSE/SSE disclosure platforms for insider filings."
    )
