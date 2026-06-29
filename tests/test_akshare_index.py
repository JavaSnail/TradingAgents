"""板块/市场指数分析分支的回归测试。

背景：用户输入 A 股指数代码（如 931743 中证半导体设备指数）曾被当作个股，
akshare 个股接口 ``stock_zh_a_hist`` 返回空 → route_to_vendor 回退到 yfinance
→ yfinance 查不到 → yf_retry 限流退避约 8 分钟空转（"无限尝试雅虎"）。

本模块验证：
  1. ``is_index`` 白名单 + 后缀消歧判定（绝不靠接口探测——实测指数接口对个股
     代码也返回错误数据，会误判）。
  2. ``get_stock`` 指数分支走 ``index_zh_a_hist``，失败返回 NO_DATA 哨兵而非
     抛 NoMarketDataError（从而 route_to_vendor 永不回退 yfinance）。
  3. 个股（002027）仍走 ``stock_zh_a_hist``，不被误判为指数——这是致命陷阱
     回归点：指数接口对 002027 也返回"收盘=4.82"的错误数据。
  4. 财务/披露/新闻工具对指数返回哨兵，不触达 yfinance。
"""
import unittest
from unittest import mock

import pandas as pd
import pytest

from tradingagents.dataflows import akshare_stock, akshare_indicators
from tradingagents.dataflows import akshare_fundamentals, akshare_disclosure, akshare_news
from tradingagents.dataflows import interface
from tradingagents.dataflows.akshare_common import is_index, is_a_share
from tradingagents.dataflows.symbol_utils import NoMarketDataError


def _idx_ohlcv_df():
    """构造 index_zh_a_hist 风格的返回 DataFrame（列名与个股接口一致）。"""
    return pd.DataFrame({
        "日期": ["2026-06-26", "2026-06-27", "2026-06-28"],
        "开盘": [100.0, 101.0, 102.0],
        "收盘": [100.5, 101.5, 102.5],
        "最高": [101.0, 102.0, 103.0],
        "最低": [99.5, 100.5, 101.5],
        "成交量": [10000, 11000, 12000],
    })


@pytest.mark.unit
class IsIndexTests(unittest.TestCase):
    """白名单 + 后缀消歧判定。"""

    def test_unambiguous_index_prefixes(self):
        # 930/931/932/880 段无视后缀即为指数
        for code in ("931743", "930908", "932051", "880001", "931743.SZ"):
            self.assertTrue(is_index(code), f"{code} 应判为指数")

    def test_known_index_whitelist(self):
        for code in ("000300", "000016", "000852", "399001", "399006"):
            self.assertTrue(is_index(code), f"{code} 应判为指数")

    def test_ambiguous_code_requires_sh_ss_suffix(self):
        # 000001 歧义：带 .SH/.SS 判指数，否则按个股
        self.assertTrue(is_index("000001.SH"))
        self.assertTrue(is_index("000001.SS"))
        self.assertFalse(is_index("000001.SZ"))
        self.assertFalse(is_index("000001"))  # 无后缀按个股（平安银行）

    def test_individual_stocks_not_index(self):
        for code in ("002027", "600519", "688981", "300750", "000333"):
            self.assertFalse(is_index(code), f"{code} 不应判为指数")

    def test_individual_stock_with_ss_sh_suffix_not_index(self):
        # 后缀消歧只作用于歧义段(000/399)。个股段(600/688/002)带 .SS/.SH
        # 仍是个股——否则 600519.SS(茅台)会被误判为指数，触发财务哨兵误伤。
        for code in ("600519.SS", "600519.SH", "688981.SS", "002027.SZ"):
            self.assertFalse(is_index(code), f"{code} 不应判为指数")

    def test_non_a_share_and_invalid(self):
        for code in ("AAPL", "TSM", "", "12345", "abc123"):
            self.assertFalse(is_index(code))

    def test_index_is_still_a_share(self):
        # 指数也是 A 股市场标的，is_a_share 对其返回 True（语义保持，
        # news/sentiment 据此走中文数据源、跳过 FRED/Reddit）
        self.assertTrue(is_a_share("931743"))
        self.assertTrue(is_a_share("000001.SH"))


@pytest.mark.unit
class GetStockIndexTests(unittest.TestCase):
    """get_stock 指数分支：走 index_zh_a_hist，失败返回哨兵不抛异常。"""

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_index_uses_index_api_not_stock_api(self, mock_ak):
        mock_ak.index_zh_a_hist.return_value = _idx_ohlcv_df()

        result = akshare_stock.get_stock("931743", "2026-06-01", "2026-06-29")

        self.assertIn("AkShare Index", result)
        self.assertIn("Close", result)
        mock_ak.index_zh_a_hist.assert_called_once()
        mock_ak.stock_zh_a_hist.assert_not_called()  # 关键：未走个股接口

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_index_empty_returns_sentinel_not_exception(self, mock_ak):
        # 指数接口返回空 → 返回哨兵字符串（不抛 NoMarketDataError）
        mock_ak.index_zh_a_hist.return_value = pd.DataFrame()

        result = akshare_stock.get_stock("931743", "2026-06-01", "2026-06-29")

        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("index", result.lower())

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_index_internal_error_returns_sentinel(self, mock_ak):
        # akshare 内部 bug（930908/932051 抛 TypeError）→ 返回哨兵不抛
        mock_ak.index_zh_a_hist.side_effect = TypeError("NoneType not subscriptable")

        result = akshare_stock.get_stock("931743", "2026-06-01", "2026-06-29")

        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("internal error", result)

    @mock.patch("tradingagents.dataflows.akshare_stock.ak")
    def test_individual_stock_uses_stock_api(self, mock_ak):
        # 个股回归：002027 走 stock_zh_a_hist，不被误判为指数
        mock_ak.stock_zh_a_hist.return_value = _idx_ohlcv_df()

        result = akshare_stock.get_stock("002027", "2026-06-01", "2026-06-29")

        self.assertNotIn("AkShare Index", result)
        mock_ak.stock_zh_a_hist.assert_called_once()
        mock_ak.index_zh_a_hist.assert_not_called()


@pytest.mark.unit
class RoutingNeverYfinanceTests(unittest.TestCase):
    """指数输入经 route_to_vendor 永不触达 yfinance（消除 8 分钟空转）。"""

    def test_index_real_path_never_yfinance(self):
        # 真实 get_akshare_stock 在指数输入下返回数据（成功路径）或哨兵
        # （失败路径），绝不抛 NoMarketDataError，故即便 chain 含 yfinance
        # 也不会触达它。yfinance 实现设为 spy，被调用即记录。
        yfinance_called = []

        def yfinance_spy(*a, **k):
            yfinance_called.append(True)
            return "YFIN_DATA"

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"akshare": akshare_stock.get_stock, "yfinance": yfinance_spy}},
            clear=False,
        ), mock.patch("tradingagents.dataflows.akshare_stock.ak") as mock_ak:
            # 成功路径：指数接口返回数据，akshare 直接 return，不触达 yfinance
            mock_ak.index_zh_a_hist.return_value = _idx_ohlcv_df()
            result = interface.route_to_vendor(
                "get_stock_data", "931743", "2026-06-01", "2026-06-29"
            )
        self.assertIn("AkShare Index", result)
        self.assertEqual(yfinance_called, [])

    def test_index_failure_returns_sentinel_never_yfinance(self):
        # 失败路径：指数接口返回空 → get_stock 返回哨兵字符串（不抛
        # NoMarketDataError）→ route_to_vendor 直接 return，不触达 yfinance。
        yfinance_called = []

        def yfinance_spy(*a, **k):
            yfinance_called.append(True)
            return "YFIN_DATA"

        with mock.patch.dict(
            interface.VENDOR_METHODS,
            {"get_stock_data": {"akshare": akshare_stock.get_stock, "yfinance": yfinance_spy}},
            clear=False,
        ), mock.patch("tradingagents.dataflows.akshare_stock.ak") as mock_ak:
            mock_ak.index_zh_a_hist.return_value = pd.DataFrame()
            result = interface.route_to_vendor(
                "get_stock_data", "931743", "2026-06-01", "2026-06-29"
            )
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertEqual(yfinance_called, [])


@pytest.mark.unit
class FundamentalsIndexSentinelTests(unittest.TestCase):
    """财务/披露/新闻工具对指数统一返回哨兵，不触达个股接口。"""

    @mock.patch("tradingagents.dataflows.akshare_fundamentals.ak")
    def test_fundamentals_index_sentinel(self, mock_ak):
        result = akshare_fundamentals.get_fundamentals("931743", "2026-06-29")
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("index", result.lower())
        mock_ak.stock_individual_info_em.assert_not_called()

    @mock.patch("tradingagents.dataflows.akshare_fundamentals.ak")
    def test_balance_sheet_index_sentinel(self, mock_ak):
        result = akshare_fundamentals.get_balance_sheet("931743", "quarterly", "2026-06-29")
        self.assertIn("NO_DATA_AVAILABLE", result)
        mock_ak.stock_financial_report_sina.assert_not_called()

    @mock.patch("tradingagents.dataflows.akshare_fundamentals.ak")
    def test_income_statement_index_sentinel(self, mock_ak):
        result = akshare_fundamentals.get_income_statement("931743", "quarterly", "2026-06-29")
        self.assertIn("NO_DATA_AVAILABLE", result)
        mock_ak.stock_profit_sheet_by_report_em.assert_not_called()

    @mock.patch("tradingagents.dataflows.akshare_fundamentals.ak")
    def test_cashflow_index_sentinel(self, mock_ak):
        result = akshare_fundamentals.get_cashflow("931743", "quarterly", "2026-06-29")
        self.assertIn("NO_DATA_AVAILABLE", result)
        mock_ak.stock_cash_flow_sheet_by_report_em.assert_not_called()

    def test_insider_transactions_index_sentinel(self):
        result = akshare_fundamentals.get_insider_transactions("931743")
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("index", result.lower())

    @mock.patch("tradingagents.dataflows.akshare_disclosure.ak")
    def test_disclosures_index_sentinel(self, mock_ak):
        result = akshare_disclosure.get_disclosures("931743", "2026-06-29", 30)
        self.assertIn("NO_DATA_AVAILABLE", result)
        self.assertIn("index", result.lower())
        mock_ak.stock_zh_a_disclosure_report_cninfo.assert_not_called()

    @mock.patch("tradingagents.dataflows.akshare_news.ak")
    def test_news_index_returns_empty_feed_hint(self, mock_ak):
        import json
        result = akshare_news.get_news("931743", "2026-06-01", "2026-06-29")
        payload = json.loads(result)
        self.assertEqual(payload["items"], 0)
        self.assertIn("get_global_news", payload["note"])
        mock_ak.stock_news_em.assert_not_called()


@pytest.mark.unit
class IndicatorsIndexTests(unittest.TestCase):
    """get_indicator 指数分支：走 index_zh_a_hist 算技术指标。"""

    @mock.patch("tradingagents.dataflows.akshare_indicators.ak")
    def test_index_indicator_uses_index_api(self, mock_ak):
        # 构造足够长的 OHLCV 供 stockstats 计算 RSI
        n = 60
        df = pd.DataFrame({
            "日期": pd.date_range(end="2026-06-29", periods=n).strftime("%Y-%m-%d"),
            "开盘": [100.0 + i for i in range(n)],
            "收盘": [100.5 + i for i in range(n)],
            "最高": [101.0 + i for i in range(n)],
            "最低": [99.5 + i for i in range(n)],
            "成交量": [10000 + i for i in range(n)],
        })
        mock_ak.index_zh_a_hist.return_value = df

        result = akshare_indicators.get_indicator("931743", "rsi", "2026-06-29", 5)

        mock_ak.index_zh_a_hist.assert_called_once()
        mock_ak.stock_zh_a_hist.assert_not_called()
        self.assertIn("RSI", result)

    @mock.patch("tradingagents.dataflows.akshare_indicators.ak")
    def test_index_indicator_empty_returns_sentinel(self, mock_ak):
        mock_ak.index_zh_a_hist.return_value = pd.DataFrame()

        result = akshare_indicators.get_indicator("931743", "rsi", "2026-06-29", 5)

        self.assertIn("NO_DATA_AVAILABLE", result)


if __name__ == "__main__":
    unittest.main()
