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
    """判断是否为 A 股代码（6 位纯数字，或带 .SZ/.SS 后缀的 6 位数字）。

    注意：本函数语义是"A 股市场标识"，对板块/市场指数（如 931743）也返回
    True——指数同样是 A 股市场标的，需要走中文数据源、跳过 FRED/Reddit 等
    境外数据。要区分"个股 vs 指数"，请用 :func:`is_index` 做正交判断。
    """
    code = normalize_symbol(symbol)
    return code.isdigit() and len(code) == 6


# 无歧义指数代码段（前缀）：纯数字即可判定为指数，无需交易所后缀消歧。
# 930/931/932 为中证规模/行业/主题指数，880 为申万行业指数。
_INDEX_PREFIXES = ("930", "931", "932", "880")

# 歧义段（000xxx/399xxx）中无歧义的知名指数码白名单。
# 这些代码在个股接口 stock_zh_a_hist 返回空，且与个股无代码冲突，
# 可直接判为指数。冷门指数若不在白名单，需用户带 .SZ/.SS 后缀消歧。
_INDEX_KNOWN_CODES = frozenset({
    "000300",  # 沪深300
    "000016",  # 上证50
    "000852",  # 中证1000
    "000905",  # 中证500
    "000010",  # 上证180
    "000906",  # 中证800
    "000907",  # 中证700
    "000908",  # 中证全指
    "399001",  # 深证成指
    "399006",  # 创业板指
    "399005",  # 中小板指
    "399300",  # 沪深300（深所）
    "399010",  # 深证200
    "399015",  # 深证100
    "399106",  # 深证综指
    "399107",  # 深证A指
    "399108",  # 深证B指
    "399311",  # 国证1000
    "399365",  # 国证行业
})


def is_index(symbol: str) -> bool:
    """判断是否为 A 股板块/市场指数代码（白名单 + 后缀消歧）。

    识别策略（绝不依赖接口探测——``index_zh_a_hist`` 对个股代码也会返回
    错误数据，探测会误判，见实测事实）：

      1. 取交易所后缀（``.SH/.SS/.SZ/.BJ`` 等，大小写不敏感）与 6 位代码。
         注意：``normalize_symbol`` 取 ``.`` 前会丢后缀，而后缀是歧义码消歧
         的唯一依据，故本函数先保留原始 symbol 再拆分。
      2. 无歧义指数段（930/931/932/880）→ True（无视后缀）。
      3. 知名指数码白名单（000300/399001 等）→ True。
      4. 歧义码（如 000001）带 ``.SH/.SS`` 后缀 → True（上证指数）；
         带 ``.SZ`` 或无后缀 → False（按个股，如平安银行）。
      5. 其余 6 位数字 → False。

    典型判别：
      ``931743``→True、``000300``→True、``399001``→True、
      ``000001.SH``→True、``000001.SZ``→False、``000001``→False、
      ``002027``→False。
    """
    if not isinstance(symbol, str) or not symbol.strip():
        return False
    s = symbol.strip().upper()
    code = s.split(".", 1)[0]
    suffix = s[len(code):] if "." in s else ""
    if not (code.isdigit() and len(code) == 6):
        return False
    if code.startswith(_INDEX_PREFIXES):
        return True
    if code in _INDEX_KNOWN_CODES:
        return True
    # 歧义段（000xxx/399xxx）：仅当带沪市后缀才判为指数
    # （000001.SH = 上证指数；000001.SZ / 无后缀 = 平安银行个股）。
    # 后缀消歧只作用于歧义段——600519.SS（茅台个股）等个股段带 .SS 仍是个股。
    if code.startswith(("000", "399")) and suffix in (".SH", ".SS"):
        return True
    return False


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
