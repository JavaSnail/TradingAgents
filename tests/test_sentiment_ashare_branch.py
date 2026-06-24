"""情绪分析师 A 股分支路由单测。

验证 A 股标的走 ``fetch_ashare_*``、非 A 股仍走 StockTwits/Reddit。
不调用真实 LLM —— 只验证数据预取分支与 system_message 组装。
"""

from __future__ import annotations

from unittest.mock import patch

import pytest

from tradingagents.agents.analysts import sentiment_analyst as sa


def _state(ticker: str) -> dict:
    return {
        "company_of_interest": ticker,
        "trade_date": "2026-06-24",
        "messages": [],
    }


@pytest.mark.unit
def test_ashare_ticker_uses_ashare_sources():
    """A 股标的应调用 fetch_ashare_quick_news / fetch_ashare_sentiment，不调 stocktwits/reddit。"""
    with patch.object(sa, "fetch_ashare_quick_news", return_value="<quick>") as m_quick, \
         patch.object(sa, "fetch_ashare_sentiment", return_value="<sentiment>") as m_sent, \
         patch.object(sa, "fetch_stocktwits_messages", return_value="<st>") as m_st, \
         patch.object(sa, "fetch_reddit_posts", return_value="<reddit>") as m_reddit, \
         patch.object(sa, "invoke_structured_or_freetext", return_value="report") as m_invoke:
        node = sa.create_sentiment_analyst(llm=None)
        node(_state("600519.SS"))
    m_quick.assert_called_once()
    m_sent.assert_called_once()
    m_st.assert_not_called()
    m_reddit.assert_not_called()
    # 组装的 system_message 应含 A 股分析指南与千股千评块
    sys_msg = m_invoke.call_args.args[2]  # formatted_messages 是第 3 个位置参数
    sys_text = "\n".join(m.content for m in sys_msg)
    assert "千股千评情绪数据" in sys_text
    assert "东方财富个股新闻" in sys_text


@pytest.mark.unit
def test_non_ashare_ticker_uses_global_sources():
    """美股标的应调用 get_news / stocktwits / reddit，不调 ashare 源。"""
    fake_news_tool = type("T", (), {"func": staticmethod(lambda *a: "<news>")})()
    with patch.object(sa, "get_news", fake_news_tool), \
         patch.object(sa, "fetch_ashare_quick_news", return_value="<quick>") as m_quick, \
         patch.object(sa, "fetch_ashare_sentiment", return_value="<sentiment>") as m_sent, \
         patch.object(sa, "fetch_stocktwits_messages", return_value="<st>") as m_st, \
         patch.object(sa, "fetch_reddit_posts", return_value="<reddit>") as m_reddit, \
         patch.object(sa, "invoke_structured_or_freetext", return_value="report") as m_invoke:
        node = sa.create_sentiment_analyst(llm=None)
        node(_state("AAPL"))
    m_quick.assert_not_called()
    m_sent.assert_not_called()
    m_st.assert_called_once()
    m_reddit.assert_called_once()
    sys_msg = m_invoke.call_args.args[2]
    sys_text = "\n".join(m.content for m in sys_msg)
    assert "StockTwits messages" in sys_text
    assert "Reddit posts" in sys_text
