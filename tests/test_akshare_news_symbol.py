"""A 股个股新闻 (akshare_news.get_news) 单测。

锁住 ``stock_news_em`` 的 ``symbol=`` 参数名（AkShare 1.18.x 从 ``stock=`` 改名，
传旧名会抛 TypeError 导致情绪/新闻分析师拿到占位符）与列名适配、日期过滤。
不依赖网络：monkeypatch ``akshare`` 接口。
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import akshare_news


def _news_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"关键词": "600519", "新闻标题": "贵州茅台权益分派", "新闻内容": "每股派现28元",
         "发布时间": "2026-06-21 18:05:00", "文章来源": "证券时报网",
         "新闻链接": "http://finance.eastmoney.com/a/1.html"},
        # 窗口外：应被过滤
        {"关键词": "600519", "新闻标题": "旧闻", "新闻内容": "不应出现",
         "发布时间": "2026-05-01 09:00:00", "文章来源": "旧源", "新闻链接": "http://x"},
    ])


@pytest.mark.unit
def test_get_news_uses_symbol_keyword():
    """必须用 symbol=（不是 stock=），否则 akshare 1.18.x 抛 TypeError。"""
    with patch.object(akshare_news.ak, "stock_news_em", return_value=_news_df()) as m:
        out = akshare_news.get_news("600519.SS", "2026-06-01", "2026-06-25")
    _, kwargs = m.call_args
    assert "symbol" in kwargs
    assert kwargs["symbol"] == "600519"
    assert "stock" not in kwargs  # 旧参数名绝不能再用


@pytest.mark.unit
def test_get_news_filters_by_date_and_maps_columns():
    with patch.object(akshare_news.ak, "stock_news_em", return_value=_news_df()):
        out = akshare_news.get_news("600519.SS", "2026-06-01", "2026-06-25")
    data = json.loads(out)
    titles = [f["title"] for f in data["feed"]]
    assert "贵州茅台权益分派" in titles
    assert "旧闻" not in titles  # 窗口外被过滤
    item = data["feed"][0]
    assert item["source"] == "证券时报网"
    assert item["url"].startswith("http")
    assert item["time_published"].startswith("20260621")


@pytest.mark.unit
def test_get_news_non_ashare_raises_no_data():
    from tradingagents.dataflows.errors import NoMarketDataError
    with pytest.raises(NoMarketDataError):
        akshare_news.get_news("AAPL", "2026-06-01", "2026-06-25")


@pytest.mark.unit
def test_get_news_request_failure_raises_no_data():
    from tradingagents.dataflows.errors import NoMarketDataError
    with patch.object(akshare_news.ak, "stock_news_em", side_effect=RuntimeError("boom")):
        with pytest.raises(NoMarketDataError):
            akshare_news.get_news("600519.SS", "2026-06-01", "2026-06-25")


@pytest.mark.unit
def test_get_news_empty_returns_empty_feed():
    with patch.object(akshare_news.ak, "stock_news_em", return_value=pd.DataFrame()):
        out = akshare_news.get_news("600519.SS", "2026-06-01", "2026-06-25")
    data = json.loads(out)
    assert data["feed"] == []
    assert data["items"] == 0
