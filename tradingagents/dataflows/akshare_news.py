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

    AkShare ``stock_news_em`` 仅返回最近的个股新闻（实测约 10 条，无法精确
    按日期回溯），这里尽量筛选日期范围内的数据。注意：参数名为 ``symbol``
    （旧版 AkShare 用 ``stock``，1.18.x 已改名，传 ``stock=`` 会抛 TypeError）。
    """
    if not is_a_share(ticker):
        raise NoMarketDataError(
            ticker, ticker, "AkShare only supports A-share (6-digit) symbols"
        )

    code = normalize_symbol(ticker)
    try:
        with bypass_proxy():
            df = ak.stock_news_em(symbol=code)
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


def _market_overview_summary(curr_date: str) -> str | None:
    """获取当日沪深市场整体涨跌统计，作为宏观新闻的量化背景补充。

    返回人类可读摘要字符串；数据不可用时返回 None（不阻断主新闻流）。
    """
    try:
        with bypass_proxy():
            df = ak.stock_zh_a_spot_em()
        if df is None or df.empty or "涨跌幅" not in df.columns:
            return None
        up = (df["涨跌幅"] > 0).sum()
        down = (df["涨跌幅"] < 0).sum()
        flat = len(df) - up - down
        avg_change = df["涨跌幅"].mean()
        return (
            f"A-share market overview on {curr_date}: "
            f"Total stocks: {len(df)}, "
            f"Rising: {up}, Falling: {down}, Flat: {flat}, "
            f"Average change: {avg_change:.2f}%"
        )
    except Exception:
        return None


def get_global_news(
    curr_date: Annotated[str, "当前日期 YYYY-MM-DD"],
    look_back_days: int = 7,
    limit: int = 50,
) -> str:
    """获取 A 股宏观财经新闻，返回 JSON 字符串。

    主体为财联社电报（``stock_info_global_cls``，7×24 快讯，事件驱动性强），
    按日期窗口 [curr_date - look_back_days, curr_date] 过滤并截断到 ``limit`` 条。
    末尾追加一条沪深市场整体涨跌统计作为量化背景补充（数据不可用时静默跳过）。

    AkShare 字段可能随版本变动，这里用 ``.get`` 做列名容错；任一环节失败都
    降级为空 feed，不抛异常（宏观新闻属于 ``news_data`` 类别，但 AkShare 实现
    内部本就 try/except 兜底，保持该约定）。
    """
    feed: list[dict] = []

    # 1) 财联社电报 —— 主体新闻
    try:
        with bypass_proxy():
            df = ak.stock_info_global_cls()
        if df is not None and not df.empty:
            start_dt = pd.Timestamp(curr_date) - pd.Timedelta(days=int(look_back_days))
            end_dt = pd.Timestamp(curr_date) + pd.Timedelta(days=1)  # 含当日全天
            for _, row in df.iterrows():
                # 财联社电报列名通常为：发布时间 / 标题 / 内容 / 分类（容错）
                title = str(row.get("标题", row.get("title", "")))
                summary = str(row.get("内容", row.get("content", "")))[:500]
                pub_time_raw = row.get("发布时间", row.get("time", ""))
                category = str(row.get("分类", row.get("category", "")))

                try:
                    pub_dt = pd.Timestamp(str(pub_time_raw))
                    # 过滤日期范围外的快讯
                    if not (start_dt <= pub_dt < end_dt):
                        continue
                    time_published = pub_dt.strftime("%Y%m%dT%H%M%S")
                except Exception:
                    time_published = str(pub_time_raw)

                feed.append({
                    "title": title,
                    "summary": summary,
                    "time_published": time_published,
                    "source": "财联社" + (f"/{category}" if category else ""),
                })
                if len(feed) >= int(limit):
                    break
    except Exception:
        # 财联社拉取失败不阻断 —— 仍尝试给出市场涨跌统计
        pass

    # 2) 沪深市场整体涨跌统计 —— 量化背景补充
    overview = _market_overview_summary(curr_date)
    if overview:
        feed.append({
            "title": f"A-Share Market Daily Summary ({curr_date})",
            "summary": overview,
            "time_published": curr_date,
            "source": "AkShare/EastMoney",
        })

    if not feed:
        return json.dumps({"feed": [], "note": "global news unavailable via AkShare"}, ensure_ascii=False)

    return json.dumps({"feed": feed, "items": len(feed)}, ensure_ascii=False)
