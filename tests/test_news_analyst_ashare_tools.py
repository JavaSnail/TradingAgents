"""新闻分析师 A 股工具分支单测。

验证 A 股标的不加载 get_macro_indicators / get_prediction_markets（避免 FRED/
Polymarket 在境内网络反复重试失败刷屏），非 A 股仍加载全部四工具。
不调用真实 LLM —— 用 RunnableLambda 包装记录 bind_tools 收到的工具集。
"""

from __future__ import annotations

from unittest.mock import MagicMock

import pytest
from langchain_core.runnables import RunnableLambda

from tradingagents.agents.analysts import news_analyst as na


def _state(ticker: str) -> dict:
    return {
        "company_of_interest": ticker,
        "trade_date": "2026-06-25",
        "asset_type": "stock",
        "messages": [],
    }


class _ToolCapturingLLM:
    """记录 bind_tools 传入的工具；invoke 返回无 tool_calls 的结果。

    ``bind_tools`` 返回一个 RunnableLambda，使 ``prompt | bound`` 管道成立。
    """

    def __init__(self):
        self.bound_tools: list = []

    def bind_tools(self, tools):
        self.bound_tools = list(tools)

        def _invoke(_messages):
            result = MagicMock()
            result.tool_calls = []
            result.content = "report"
            return result

        return RunnableLambda(_invoke)


@pytest.mark.unit
def test_ashare_news_analyst_omits_macro_and_prediction_tools():
    llm = _ToolCapturingLLM()
    node = na.create_news_analyst(llm)
    node(_state("600519.SS"))
    tool_names = [t.name for t in llm.bound_tools]
    assert "get_news" in tool_names
    assert "get_global_news" in tool_names
    # A 股不应加载这两个境外/美国宏观源
    assert "get_macro_indicators" not in tool_names
    assert "get_prediction_markets" not in tool_names


@pytest.mark.unit
def test_non_ashare_news_analyst_keeps_all_four_tools():
    llm = _ToolCapturingLLM()
    node = na.create_news_analyst(llm)
    node(_state("AAPL"))
    tool_names = [t.name for t in llm.bound_tools]
    assert "get_news" in tool_names
    assert "get_global_news" in tool_names
    assert "get_macro_indicators" in tool_names
    assert "get_prediction_markets" in tool_names
