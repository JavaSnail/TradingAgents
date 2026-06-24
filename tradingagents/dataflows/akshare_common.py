import logging
import time
from contextlib import contextmanager
from datetime import datetime

import requests as _requests

logger = logging.getLogger(__name__)

# 持久化 Session，trust_env=False 绕过 Windows 系统代理。
# 复用同一 Session 避免频繁 TLS 握手被东方财富 CDN 断开。
_proxy_free_session = _requests.Session()
_proxy_free_session.trust_env = False
_proxy_free_session.proxies = {}
_proxy_free_session.headers.update({
    "User-Agent": (
        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
    ),
    "Referer": "https://data.eastmoney.com/",
})
logger.debug("AkShare proxy-free session initialized")


def _do_request(method: str, url: str, max_retries: int = 3, base_delay: float = 5.0, **kwargs):
    """使用绕过代理的 Session 发送请求，带指数退避重试。

    东方财富 CDN 偶尔会断开连接（限流或网络波动），指数退避可平滑恢复。
    """
    global _proxy_free_session

    for attempt in range(max_retries + 1):
        try:
            if method == "GET":
                return _proxy_free_session.get(url, **kwargs)
            else:
                return _proxy_free_session.post(url, **kwargs)
        except _requests.exceptions.ConnectionError as e:
            if attempt < max_retries:
                delay = base_delay * (2 ** attempt)
                logger.warning(
                    "AkShare ConnectionError (attempt %d/%d), retrying in %.0fs: %s",
                    attempt + 1, max_retries, delay, e,
                )
                time.sleep(delay)
                # 重建 Session（旧连接池可能已被远程关闭）
                _proxy_free_session.close()
                _proxy_free_session = _requests.Session()
                _proxy_free_session.trust_env = False
                _proxy_free_session.proxies = {}
                _proxy_free_session.headers.update({
                    "User-Agent": (
                        "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 "
                        "(KHTML, like Gecko) Chrome/120.0.0.0 Safari/537.36"
                    ),
                    "Referer": "https://data.eastmoney.com/",
                })
            else:
                raise
        except Exception:
            if attempt < max_retries:
                time.sleep(base_delay * (2 ** attempt))
                continue
            raise


def _patch_get(url, **kwargs):
    """替代 requests.get，使用绕过代理的持久 Session。"""
    return _do_request("GET", url, **kwargs)


def _patch_post(url, **kwargs):
    """替代 requests.post，使用绕过代理的持久 Session。"""
    return _do_request("POST", url, **kwargs)


# ---- 全局 patch ---- #
# AkShare 只访问国内数据源（东方财富、新浪），全局绕过代理不会影响
# yfinance/Fred 等其他 vendor（它们通过自己的 Session 发请求）。
_orig_get = _requests.get
_orig_post = _requests.post
_requests.get = _patch_get
_requests.post = _patch_post
_patch_active = True


@contextmanager
def bypass_proxy():
    """上下文管理器，保持绕过代理的 patch 生效（兼容旧代码调用）。

    由于模块加载时已全局 patch，此函数仅做占位兼容。
    退出上下文时不会恢复原始 requests 方法，确保后续调用不受影响。
    """
    yield


def restore_requests():
    """恢复原始 requests.get/post（仅测试用）。"""
    global _patch_active
    if _patch_active:
        _requests.get = _orig_get
        _requests.post = _orig_post
        _patch_active = False


def normalize_symbol(symbol: str) -> str:
    """将各种格式的 A 股代码统一为 6 位纯数字。

    支持：002027.SZ / 002027.SS / 002027 → 002027
    """
    return symbol.split(".")[0].strip()


def to_sina_symbol(symbol: str) -> str:
    """转换为新浪财报 API 需要的格式（sh/sz 前缀）。

    规则：6 开头 → 沪市 sh，其他 → 深市 sz
    示例：600519 → sh600519，002027 → sz002027
    """
    code = normalize_symbol(symbol)
    prefix = "sh" if code.startswith("6") else "sz"
    return prefix + code


def is_a_share(symbol: str) -> bool:
    """判断是否为 A 股代码（6 位纯数字，或带 .SZ/.SS 后缀的 6 位数字）。"""
    code = normalize_symbol(symbol)
    return code.isdigit() and len(code) == 6


def to_akshare_date(date_str: str) -> str:
    """将 YYYY-MM-DD 转为 AkShare 要求的 YYYYMMDD 格式。"""
    return date_str.replace("-", "")


def format_header(label: str, start_date: str, end_date: str, record_count: int) -> str:
    """生成与 yfinance 适配器格式一致的 CSV 头注释。"""
    return (
        f"# Stock data for {label} from {start_date} to {end_date}\n"
        f"# Total records: {record_count}\n"
        f"# Data retrieved on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}\n\n"
    )
