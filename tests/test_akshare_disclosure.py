"""A 股公告数据 (akshare_disclosure.get_disclosures) 单测。

不依赖网络：monkeypatch ``akshare`` 接口，验证命中/日期窗口/非A股/空/异常/路由
六种路径。参照 ``test_akshare_ashare_news.py`` 与 ``test_no_data_handling.py``。
"""

from __future__ import annotations

from unittest.mock import patch

import pandas as pd
import pytest

from tradingagents.dataflows import akshare_disclosure
from tradingagents.dataflows.akshare import get_disclosures
from tradingagents.dataflows import interface
from tradingagents.dataflows.config import set_config
from tradingagents.dataflows.errors import NoMarketDataError


def _disclosure_df() -> pd.DataFrame:
    return pd.DataFrame([
        {"代码": "600519", "简称": "贵州茅台", "公告标题": "2025年度权益分派实施公告", "公告时间": "2026-06-22", "公告链接": "http://www.cninfo.com.cn/1"},
        {"代码": "600519", "简称": "贵州茅台", "公告标题": "关于召开2025年年度股东大会的通知", "公告时间": "2026-06-12", "公告链接": "http://www.cninfo.com.cn/2"},
    ])


@pytest.mark.unit
def test_disclosures_hit_returns_csv():
    with patch.object(akshare_disclosure.ak, "stock_zh_a_disclosure_report_cninfo", return_value=_disclosure_df()):
        out = get_disclosures("600519.SS", "2026-06-25", 30)
    assert "# Disclosures for 600519" in out
    assert "公告标题" in out  # CSV 表头
    assert "权益分派实施公告" in out
    assert "Total records: 2" in out


@pytest.mark.unit
def test_disclosures_date_window_passed():
    with patch.object(
        akshare_disclosure.ak, "stock_zh_a_disclosure_report_cninfo", return_value=_disclosure_df()
    ) as m:
        get_disclosures("600519.SS", "2026-06-25", 30)
    _, kwargs = m.call_args
    assert kwargs["symbol"] == "600519"
    assert kwargs["market"] == "沪深京"
    # curr_date=2026-06-25, look_back=30 -> start=2026-05-26
    assert kwargs["start_date"] == "20260526"
    assert kwargs["end_date"] == "20260625"


@pytest.mark.unit
def test_disclosures_non_ashare_raises_no_data():
    with pytest.raises(NoMarketDataError):
        get_disclosures("AAPL", "2026-06-25", 30)


@pytest.mark.unit
def test_disclosures_empty_raises_no_data():
    with patch.object(akshare_disclosure.ak, "stock_zh_a_disclosure_report_cninfo", return_value=pd.DataFrame()):
        with pytest.raises(NoMarketDataError):
            get_disclosures("600519.SS", "2026-06-25", 30)


@pytest.mark.unit
def test_disclosures_request_failure_raises_no_data():
    with patch.object(akshare_disclosure.ak, "stock_zh_a_disclosure_report_cninfo", side_effect=RuntimeError("boom")):
        with pytest.raises(NoMarketDataError):
            get_disclosures("600519.SS", "2026-06-25", 30)


@pytest.mark.unit
def test_disclosures_route_via_vendor():
    """通过 route_to_vendor 调用，确认 fundamental_data=akshare 时路由到实现。"""
    set_config({"data_vendors": {"fundamental_data": "akshare"}})
    with patch.object(akshare_disclosure.ak, "stock_zh_a_disclosure_report_cninfo", return_value=_disclosure_df()):
        out = interface.route_to_vendor("get_disclosures", "600519.SS", "2026-06-25", 30)
    assert "公告标题" in out
    # 路由表注册确认
    assert "get_disclosures" in interface.VENDOR_METHODS
    assert "akshare" in interface.VENDOR_METHODS["get_disclosures"]
    # 归入 fundamental_data 类别
    assert "get_disclosures" in interface.TOOLS_CATEGORIES["fundamental_data"]["tools"]
