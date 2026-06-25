"""A 股上市公司公告数据（巨潮资讯 cninfo）。

为基本面分析师提供个股公告/披露数据源。公告（业绩预告/快报、重大事项、分红、
增减持、监管问询等）是 A 股基本面分析的核心事件源，原 ``akshare_fundamentals``
仅有财务报表，本模块补齐公告面。

数据源：``stock_zh_a_disclosure_report_cninfo``（巨潮资讯，官方披露平台，权威）。
返回 CSV 字符串（与 ``akshare_fundamentals.get_balance_sheet`` 等一致），含表头
注释。空结果或请求异常抛 ``NoMarketDataError``（核心类别不静默降级，与其它
fundamentals 实现一致）。
"""

from datetime import datetime, timedelta
from typing import Annotated

import akshare as ak

from .akshare_common import bypass_proxy, is_a_share, normalize_symbol, to_akshare_date
from .symbol_utils import NoMarketDataError


def get_disclosures(
    ticker: Annotated[str, "股票代码，如 002027.SZ"],
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"],
    look_back_days: int = 30,
) -> str:
    """获取 A 股个股近期公告（巨潮资讯），返回 CSV 字符串。

    Args:
        ticker: A 股代码，如 ``600519.SS`` / ``002027.SZ``。
        curr_date: 当前交易日，``YYYY-MM-DD``。
        look_back_days: 向前回看的公告天数，默认 30。

    Returns:
        带表头注释的 CSV 字符串，列为
        ``代码 / 简称 / 公告标题 / 公告时间 / 公告链接``。

    Raises:
        NoMarketDataError: 非 A 股标的、窗口内无公告、或请求失败时抛出。
    """
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(ticker)

    # 计算日期窗口并转为巨潮接口要求的 YYYYMMDD 格式
    end_dt = datetime.strptime(curr_date, "%Y-%m-%d")
    start_dt = end_dt - timedelta(days=int(look_back_days))
    start_date = to_akshare_date(start_dt.strftime("%Y-%m-%d"))
    end_date = to_akshare_date(curr_date)

    try:
        with bypass_proxy():
            df = ak.stock_zh_a_disclosure_report_cninfo(
                symbol=code,
                market="沪深京",
                start_date=start_date,
                end_date=end_date,
            )
    except Exception as e:
        raise NoMarketDataError(ticker, code, f"disclosure request failed: {e}") from e

    if df is None or df.empty:
        raise NoMarketDataError(
            ticker, code, f"no disclosures in window [{start_date}, {end_date}]"
        )

    header = (
        f"# Disclosures for {code} (AkShare/CNINFO, look_back={look_back_days}d)\n"
        f"# Window: {start_date} to {end_date}\n"
        f"# Total records: {len(df)}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
    return header + df.to_csv(index=False)
