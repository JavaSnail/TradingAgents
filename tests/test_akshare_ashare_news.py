"""A 股宏观新闻 (akshare_news.get_global_news) 改造单测。

验证财联社电报为主体、日期窗口过滤生效、涨跌统计作为补充段、异常降级。
不依赖网络：monkeypatch ``akshare`` 接口。
"""

from __future__ import annotations

import json
from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import akshare_news


def _cls_df() -> pd.DataFrame:
    """模拟财联社电报返回。"""
    return pd.DataFrame([
        {"发布时间": "2026-06-24 10:00:00", "标题": "央行降准", "内容": "央行决定降准0.5个百分点", "分类": "宏观"},
        {"发布时间": "2026-06-23 14:00:00", "标题": "科创板利好", "内容": "政策支持硬科技", "分类": "政策"},
        # 窗口外：应被过滤
        {"发布时间": "2026-06-10 09:00:00", "标题": "旧闻", "内容": "不应出现", "分类": "其他"},
    ])


def _spot_df() -> pd.DataFrame:
    return pd.DataFrame({"涨跌幅": [1.2, -0.8, 0.0, 2.1]})


@pytest.mark.unit
def test_global_news_cls_headlines_with_overview():
    with patch.object(akshare_news.ak, "stock_info_global_cls", return_value=_cls_df()), \
         patch.object(akshare_news.ak, "stock_zh_a_spot_em", return_value=_spot_df()):
        out = akshare_news.get_global_news("2026-06-24", look_back_days=7, limit=50)
    data = json.loads(out)
    titles = [f["title"] for f in data["feed"]]
    # 财联社两条在窗口内
    assert "央行降准" in titles
    assert "科创板利好" in titles
    # 窗口外旧闻被过滤
    assert "旧闻" not in titles
    # 末尾追加市场涨跌统计
    assert any("A-Share Market Daily Summary" in t for t in titles)
    # 财联社条目来源标记
    cls_items = [f for f in data["feed"] if f["source"].startswith("财联社")]
    assert len(cls_items) == 2


@pytest.mark.unit
def test_global_news_limit_truncates_cls():
    with patch.object(akshare_news.ak, "stock_info_global_cls", return_value=_cls_df()), \
         patch.object(akshare_news.ak, "stock_zh_a_spot_em", return_value=_spot_df()):
        out = akshare_news.get_global_news("2026-06-24", look_back_days=7, limit=1)
    data = json.loads(out)
    # limit=1 截断财联社到 1 条；overview 仍会 append，故 items 为 1 或 2
    assert 1 <= data["items"] <= 2


@pytest.mark.unit
def test_global_news_cls_failure_falls_back_to_overview():
    with patch.object(akshare_news.ak, "stock_info_global_cls", side_effect=RuntimeError("net")), \
         patch.object(akshare_news.ak, "stock_zh_a_spot_em", return_value=_spot_df()):
        out = akshare_news.get_global_news("2026-06-24", look_back_days=7, limit=50)
    data = json.loads(out)
    # 财联社失败，但涨跌统计仍可用
    assert any("A-Share Market Daily Summary" in f["title"] for f in data["feed"])


@pytest.mark.unit
def test_global_news_all_failure_returns_empty_feed():
    with patch.object(akshare_news.ak, "stock_info_global_cls", side_effect=RuntimeError("net")), \
         patch.object(akshare_news.ak, "stock_zh_a_spot_em", side_effect=RuntimeError("net")):
        out = akshare_news.get_global_news("2026-06-24", look_back_days=7, limit=50)
    data = json.loads(out)
    assert data["feed"] == []
    assert "unavailable" in data["note"]


@pytest.mark.unit
def test_global_news_overview_missing_column_skipped():
    # 涨跌幅列缺失时 overview 返回 None，不追加、不崩溃
    with patch.object(akshare_news.ak, "stock_info_global_cls", return_value=_cls_df()), \
         patch.object(akshare_news.ak, "stock_zh_a_spot_em", return_value=pd.DataFrame({"代码": ["000001"]})):
        out = akshare_news.get_global_news("2026-06-24", look_back_days=7, limit=50)
    data = json.loads(out)
    assert not any("A-Share Market Daily Summary" in f["title"] for f in data["feed"])
    assert data["items"] == 2  # 仅财联社两条
