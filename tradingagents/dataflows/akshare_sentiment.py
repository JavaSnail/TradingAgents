"""A 股情绪与个股快讯数据封装。

为 :mod:`tradingagents.agents.analysts.sentiment_analyst` 提供 A 股专用的
情绪/事件数据块，替代对 A 股无效的 StockTwits / Reddit。返回字符串块，与
:func:`tradingagents.dataflows.stocktwits.fetch_stocktwits_messages` /
:func:`tradingagents.dataflows.reddit.fetch_reddit_posts` 同构 —— 带清晰头部、
占位降级、绝不抛异常（任何失败都返回明确的 ``<... unavailable>`` 占位，保证
情绪分析师总能拿到非空内容）。

数据源：
  - 千股千评 ``stock_comment_em``：综合得分 / 机构参与度 / 关注指数 / 主力成本
    等现成情绪与资金面标签，省去自建情感模型。
  - 个股快讯 ``stock_news_em``：复用 :mod:`akshare_news` 的日期过滤逻辑。
"""

from typing import Annotated

import akshare as ak

from .akshare_common import bypass_proxy, is_a_share, is_index, normalize_symbol
from .akshare_news import get_news as _get_ashare_news


def fetch_ashare_sentiment(ticker: str) -> str:
    """获取 A 股个股千股千评情绪数据块。

    Args:
        ticker: A 股代码，如 ``600519.SS`` / ``002027.SZ``。

    Returns:
        可读字符串块；非 A 股标的、指数（千股千评仅覆盖个股）、未命中或拉取
        失败时返回占位串。
    """
    if not is_a_share(ticker):
        return f"<ashare sentiment unavailable for {ticker}: not an A-share symbol>"
    if is_index(ticker):
        return (
            f"<ashare sentiment unavailable for {ticker}: market/sector index; "
            f"千股千评仅覆盖个股，指数情绪请参考 get_global_news 的财联社电报>"
        )

    code = normalize_symbol(ticker)
    try:
        with bypass_proxy():
            df = ak.stock_comment_em()
    except Exception as exc:
        return f"<ashare sentiment unavailable for {ticker}: request failed ({type(exc).__name__})>"

    if df is None or df.empty:
        return f"<ashare sentiment unavailable for {ticker}: stock_comment_em empty>"

    # 千股千评返回全市场表，按 6 位代码过滤到目标个股
    if "代码" not in df.columns:
        return f"<ashare sentiment unavailable for {ticker}: 代码 column missing>"
    row = df[df["代码"] == code]
    if row.empty:
        return f"<ashare sentiment unavailable for {ticker}: not in stock_comment_em>"

    rec = row.iloc[0]

    def _get(key: str, default: str = "") -> str:
        val = rec.get(key, default)
        return "" if val is None else str(val)

    name = _get("名称")
    price = _get("最新价")
    change_pct = _get("涨跌幅")
    turnover = _get("换手率")
    pe = _get("市盈率")
    main_cost = _get("主力成本")
    inst_part = _get("机构参与度")
    score = _get("综合得分")
    rank_rise = _get("上升")
    rank = _get("目前排名")
    focus = _get("关注指数")

    lines = [
        f"# 千股千评情绪数据 — {name}({code})",
        f"- 最新价: {price}  涨跌幅: {change_pct}%  换手率: {turnover}%  市盈率: {pe}",
        f"- 综合得分: {score}  (0-100，越高越偏多；排名变化: {rank_rise})",
        f"- 目前排名: {rank}  关注指数: {focus}",
        f"- 机构参与度: {inst_part}  主力成本: {main_cost}",
        "",
        "解读提示：综合得分反映东方财富聚合的个股综合评价；机构参与度越高代表机构资金"
        "活跃度越强；主力成本高于现价意味着多数筹码处于浮亏（潜在支撑），反之则浮盈"
        "（潜在抛压）。关注指数衡量散户关注度，过高需警惕情绪过热。",
    ]
    return "\n".join(lines)


def fetch_ashare_quick_news(
    ticker: Annotated[str, "A 股代码，如 002027.SZ"],
    start_date: Annotated[str, "开始日期 YYYY-MM-DD"],
    end_date: Annotated[str, "结束日期 YYYY-MM-DD"],
) -> str:
    """获取 A 股个股快讯块（东方财富个股新闻）。

    复用 :func:`akshare_news.get_news` 的日期过滤与字段适配逻辑，作为情绪
    分析师的"事件/快讯"输入。失败时降级为占位串，不抛异常。
    """
    try:
        return _get_ashare_news(ticker, start_date, end_date)
    except Exception as exc:
        return f"<ashare quick news unavailable for {ticker}: {type(exc).__name__}>"
