"""A 股情绪数据封装 (akshare_sentiment) 单测。

不依赖网络：monkeypatch ``akshare`` 接口返回固定 DataFrame，验证命中/未命中/
异常三种路径下的输出契约。参照 ``test_stocktwits_resilience.py`` 的降级断言风格。
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import akshare_sentiment


def _comment_df(code: str = "600519") -> pd.DataFrame:
    return pd.DataFrame([{
        "序号": 1,
        "代码": code,
        "名称": "贵州茅台",
        "最新价": 1207.68,
        "涨跌幅": -1.21,
        "换手率": 0.36,
        "市盈率": 13.85,
        "主力成本": 1216.84,
        "机构参与度": 0.5383,
        "综合得分": 75.27,
        "上升": -22,
        "目前排名": 264,
        "关注指数": 93.6,
        "交易日": "2026-06-24",
    }])


@pytest.mark.unit
def test_ashare_sentiment_hit_returns_score_block():
    with patch.object(akshare_sentiment.ak, "stock_comment_em", return_value=_comment_df()):
        out = akshare_sentiment.fetch_ashare_sentiment("600519.SS")
    assert "千股千评情绪数据" in out
    assert "贵州茅台" in out
    assert "综合得分" in out
    assert "75.27" in out
    assert "机构参与度" in out
    assert "主力成本" in out


@pytest.mark.unit
def test_ashare_sentiment_not_in_comment_returns_unavailable():
    with patch.object(akshare_sentiment.ak, "stock_comment_em", return_value=_comment_df("000001")):
        out = akshare_sentiment.fetch_ashare_sentiment("600519.SS")
    assert "unavailable" in out.lower()
    assert "not in stock_comment_em" in out


@pytest.mark.unit
def test_ashare_sentiment_request_failure_returns_placeholder():
    with patch.object(akshare_sentiment.ak, "stock_comment_em", side_effect=RuntimeError("boom")):
        out = akshare_sentiment.fetch_ashare_sentiment("600519.SS")
    assert "unavailable" in out.lower()
    assert "request failed" in out


@pytest.mark.unit
def test_ashare_sentiment_empty_df_returns_unavailable():
    with patch.object(akshare_sentiment.ak, "stock_comment_em", return_value=pd.DataFrame()):
        out = akshare_sentiment.fetch_ashare_sentiment("600519.SS")
    assert "unavailable" in out.lower()


@pytest.mark.unit
def test_ashare_sentiment_non_ashare_symbol_returns_unavailable():
    out = akshare_sentiment.fetch_ashare_sentiment("AAPL")
    assert "unavailable" in out.lower()
    assert "not an A-share" in out


@pytest.mark.unit
def test_ashare_quick_news_delegates_to_get_news():
    with patch.object(akshare_sentiment, "_get_ashare_news", return_value='{"feed": [], "items": 0}') as m:
        out = akshare_sentiment.fetch_ashare_quick_news("600519.SS", "2026-06-17", "2026-06-24")
    assert "feed" in out
    m.assert_called_once_with("600519.SS", "2026-06-17", "2026-06-24")


@pytest.mark.unit
def test_ashare_quick_news_failure_returns_placeholder():
    with patch.object(akshare_sentiment, "_get_ashare_news", side_effect=RuntimeError("boom")):
        out = akshare_sentiment.fetch_ashare_quick_news("600519.SS", "2026-06-17", "2026-06-24")
    assert "unavailable" in out.lower()
