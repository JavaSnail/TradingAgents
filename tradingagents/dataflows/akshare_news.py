import json
from datetime import datetime
from typing import Annotated

import akshare as ak
import pandas as pd
from dateutil.relativedelta import relativedelta

from .akshare_common import bypass_proxy, is_a_share, normalize_symbol
from .symbol_utils import NoMarketDataError


def get_news(
    ticker: Annotated[str, "股票代码，如 002027.SZ"],
    start_date: Annotated[str, "开始日期 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期 YYYY-MM-DD"],
) -> str:
    """从东方财富获取个股新闻，返回 JSON 字符串。

    AkShare 仅返回最近约 20 条新闻，无法精确按日期过滤，会尽量筛选日期范围内的数据。
    """
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(ticker)
    try:
        with bypass_proxy():
            df = ak.stock_news_em(stock=code)
    except Exception as e:
        raise NoMarketDataError(ticker, code, f"news request failed: {e}") from e

    if df is None or df.empty:
        return json.dumps({"feed": [], "items": 0})

    feed = []
    start_dt = pd.Timestamp(start_date)
    end_dt = pd.Timestamp(end_date)

    for _, row in df.iterrows():
        # 东方财富新闻列名可能为：关键词、新闻标题、新闻内容、发布时间、文章来源、新闻链接
        title = str(row.get("新闻标题", row.get("title", "")))
        summary = str(row.get("新闻内容", row.get("content", "")))[:500]
        pub_time_raw = row.get("发布时间", row.get("time", ""))
        source = str(row.get("文章来源", row.get("source", "")))
        url = str(row.get("新闻链接", row.get("url", "")))

        try:
            pub_dt = pd.Timestamp(str(pub_time_raw))
            # 过滤日期范围外的新闻
            if not (start_dt <= pub_dt <= end_dt):
                continue
            time_published = pub_dt.strftime("%Y%m%dT%H%M%S")
        except Exception:
            time_published = str(pub_time_raw)

        feed.append({
            "title": title,
            "summary": summary,
            "time_published": time_published,
            "source": source,
            "url": url,
        })

    return json.dumps({"feed": feed, "items": len(feed)}, ensure_ascii=False)


def get_global_news(
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"],
    look_back_days: int = 7,
    limit: int = 50,
) -> str:
    """获取 A 股市场整体行情概况作为宏观新闻背景。

    AkShare 没有全局财经新闻接口，改为返回当日沪深市场整体数据摘要。
    """
    try:
        with bypass_proxy():
            df = ak.stock_zh_a_spot_em()
        if df is None or df.empty:
            return json.dumps({"feed": [], "note": "no market data available"})

        # 计算市场整体统计
        up = (df["涨跌幅"] > 0).sum() if "涨跌幅" in df.columns else 0
        down = (df["涨跌幅"] < 0).sum() if "涨跌幅" in df.columns else 0
        flat = len(df) - up - down
        avg_change = df["涨跌幅"].mean() if "涨跌幅" in df.columns else 0

        summary = (
            f"A-share market overview on {curr_date}: "
            f"Total stocks: {len(df)}, "
            f"Rising: {up}, Falling: {down}, Flat: {flat}, "
            f"Average change: {avg_change:.2f}%"
        )

        feed = [{
            "title": f"A-Share Market Daily Summary ({curr_date})",
            "summary": summary,
            "time_published": curr_date,
            "source": "AkShare/EastMoney",
        }]

        return json.dumps({"feed": feed, "items": 1}, ensure_ascii=False)
    except Exception:
        return json.dumps({"feed": [], "note": "global news unavailable via AkShare"})
