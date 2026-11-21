"""
market_data.py - Asynchronous TSETMC data layer (httpx) with automatic proxy & TLS fallback.
"""
from __future__ import annotations

import asyncio
import io
import ssl
import threading
import time
from dataclasses import dataclass, field
from typing import Any, Coroutine, Dict, List, Optional, Tuple, TypeVar
from urllib.parse import quote

import httpx
import numpy as np
import pandas as pd

CDN_BASE = "https://cdn.tsetmc.com"
OLD_BASE = "http://old.tsetmc.com"
YTM_URL = "https://ifb.ir/ytm.aspx"
USER_AGENT = (
    "Mozilla/5.0 (Windows NT 10.0; Win64; x64) AppleWebKit/537.36 (KHTML, like Gecko) "
    "Chrome/124.0.0.0 Safari/537.36"
)
TLS_WARNING = "گواهی TLS سرور تأیید نشد؛ اتصال بدون تأیید گواهی برقرار شد."
PROXY_WARNING = "اتصال از طریق پراکسی سیستم ناموفق بود؛ اتصال مستقیم (بدون پراکسی) برقرار شد."

CHAIN_COLUMNS = [
    "Symbol", "UA_Symbol", "Title", "Type", "Strike_Price", "Stock_Price", "Last_Price", "Close_Price",
    "Best_Ask", "Best_Bid", "Ask_Vol", "Bid_Vol", "Open_Interest", "Volume", "Value", "Trades",
    "Days_to_Maturity", "Expiration_Date", "Contract_Size", "Market",
]

T = TypeVar("T")


class MarketDataError(RuntimeError):
    """Raised when market data cannot be fetched or parsed."""


_TO_ARABIC = str.maketrans({"ی": "ي", "ک": "ك"})
_TO_CANONICAL = str.maketrans(
    {"ي": "ی", "ك": "ک", "‌": None, " ": None, "\u00a0": None,
     **{chr(0x06F0 + i): str(i) for i in range(10)}, **{chr(0x0660 + i): str(i) for i in range(10)}}
)


def to_tsetmc_form(symbol: str) -> str:
    return str(symbol).strip().translate(_TO_ARABIC)


def canonical_symbol(symbol: str) -> str:
    return str(symbol).translate(_TO_CANONICAL)


def underlying_from_title(title: str) -> str:
    head = str(title).split("-")[0].strip()
    parts = head.split(None, 1)
    return parts[1].strip() if len(parts) == 2 else head


def _num(series: pd.Series) -> pd.Series:
    return pd.to_numeric(series, errors="coerce")


def _pick(raw: pd.DataFrame, *names: str) -> pd.Series:
    for name in names:
        if name in raw.columns:
            return raw[name]
    return pd.Series(np.nan, index=raw.index, dtype=object)


def parse_option_watch(payload: Dict[str, Any], market: str = "") -> pd.DataFrame:
    rows = payload.get("instrumentOptMarketWatch") if isinstance(payload, dict) else None
    if not isinstance(rows, list):
        raise MarketDataError("ساختار پاسخ بازار اختیار معامله ناشناخته است.")
    if not rows:
        return pd.DataFrame(columns=CHAIN_COLUMNS)

    raw = pd.DataFrame(rows)
    frames = []
    for side, kind in (("C", "Call"), ("P", "Put")):
        title = _pick(raw, f"lVal30_{side}", f"lval30_{side}").astype(str).str.strip()
        frame = pd.DataFrame({
            "Symbol": _pick(raw, f"lVal18AFC_{side}", f"lval18AFC_{side}").astype(str).str.strip(),
            "Title": title,
            "Type": kind,
            "Strike_Price": _num(_pick(raw, "strikePrice")),
            "Stock_Price": _num(_pick(raw, "pClosing_UA")),
            "Last_Price": _num(_pick(raw, f"pDrCotVal_{side}")),
            "Close_Price": _num(_pick(raw, f"pClosing_{side}")),
            "Best_Ask": _num(_pick(raw, f"pMeOf_{side}")),
            "Best_Bid": _num(_pick(raw, f"pMeDem_{side}")),
            "Ask_Vol": _num(_pick(raw, f"qTitMeOf_{side}")),
            "Bid_Vol": _num(_pick(raw, f"qTitMeDem_{side}")),
            "Open_Interest": _num(_pick(raw, f"oP_{side}")),
            "Volume": _num(_pick(raw, f"qTotTran5J_{side}")),
            "Value": _num(_pick(raw, f"qTotCap_{side}")),
            "Trades": _num(_pick(raw, f"zTotTran_{side}")),
            "Days_to_Maturity": _num(_pick(raw, "remainedDay")),
            "Expiration_Date": pd.to_datetime(
                _pick(raw, "endDate").astype(str).str[:8], format="%Y%m%d", errors="coerce"
            ),
            "Contract_Size": _num(_pick(raw, "contractSize")),
            "Market": market,
        })
        frame["UA_Symbol"] = title.map(underlying_from_title)
        frames.append(frame)

    out = pd.concat(frames, ignore_index=True)
    out = out[(out["Symbol"].str.len() > 0) & (out["Symbol"] != "nan") & out["Strike_Price"].notna()]
    numeric = ["Last_Price", "Close_Price", "Best_Ask", "Best_Bid", "Ask_Vol", "Bid_Vol",
               "Open_Interest", "Volume", "Value", "Trades"]
    out[numeric] = out[numeric].fillna(0.0)
    out["Days_to_Maturity"] = out["Days_to_Maturity"].fillna(0).astype(int)
    out["Contract_Size"] = out["Contract_Size"].fillna(1000).astype(float)
    return out[CHAIN_COLUMNS].reset_index(drop=True)


def parse_history_cdn(payload: Dict[str, Any]) -> pd.DataFrame:
    rows = payload.get("closingPriceDaily") if isinstance(payload, dict) else None
    if not isinstance(rows, list) or not rows:
        raise MarketDataError("تاریخچه قیمت دریافت نشد.")
    raw = pd.DataFrame(rows)
    if "dEven" not in raw.columns or "pClosing" not in raw.columns:
        raise MarketDataError("ساختار تاریخچه قیمت ناشناخته است.")
    out = pd.DataFrame({
        "Date": pd.to_datetime(raw["dEven"].astype(str), format="%Y%m%d", errors="coerce"),
        "Open": _num(_pick(raw, "priceFirst")),
        "High": _num(_pick(raw, "priceMax")),
        "Low": _num(_pick(raw, "priceMin")),
        "Close": _num(raw["pClosing"]),
        "Last": _num(_pick(raw, "pDrCotVal")),
        "Volume": _num(_pick(raw, "qTotTran5J")),
        "Value": _num(_pick(raw, "qTotCap")),
        "Trades": _num(_pick(raw, "zTotTran")),
        "Yesterday": _num(_pick(raw, "priceYesterday")),
    })
    out = out.dropna(subset=["Date"]).drop_duplicates("Date").sort_values("Date").set_index("Date")
    out = out[out["Close"] > 0]
    out["Volume"] = out["Volume"].fillna(0.0)
    if out.empty:
        raise MarketDataError("تاریخچه قیمت معتبری پیدا نشد.")
    return out


def parse_history_csv(text: str) -> pd.DataFrame:
    try:
        raw = pd.read_csv(io.StringIO(text), index_col="<DTYYYYMMDD>", parse_dates=True)
    except Exception as exc:
        raise MarketDataError(f"فایل تاریخچه قیمت قابل خواندن نیست: {exc}") from exc
    raw = raw.rename(columns={
        "<OPEN>": "Yesterday", "<CLOSE>": "Close", "<FIRST>": "Open", "<HIGH>": "High",
        "<LOW>": "Low", "<VOL>": "Volume", "<LAST>": "Last", "<VALUE>": "Value",
    })
    raw.index.name = "Date"
    keep = [c for c in ["Open", "High", "Low", "Close", "Last", "Volume", "Value", "Yesterday"] if c in raw.columns]
    out = raw[keep].sort_index()
    out = out[~out.index.duplicated(keep="last")]
    if "Close" not in out.columns or out.empty:
        raise MarketDataError("تاریخچه قیمت معتبری پیدا نشد.")
    return out[out["Close"] > 0]


def adjust_prices(history: pd.DataFrame) -> pd.DataFrame:
    out = history.copy()
    close = out["Close"].astype(float)
    if "Yesterday" in out.columns:
        yesterday = out["Yesterday"].astype(float)
        with np.errstate(all="ignore"):
            ratio = (close.shift(1) / yesterday).where(yesterday > 0)
        ratio = ratio.replace([np.inf, -np.inf], np.nan).fillna(1.0)
    else:
        ratio = pd.Series(1.0, index=out.index)
    factor = ratio.iloc[::-1].cumprod().iloc[::-1].shift(-1).fillna(1.0)
    out["Adj Factor"] = factor
    out["Adj Close"] = close / factor
    return out


def parse_ytm_html(html: str) -> float:
    try:
        tables = pd.read_html(io.StringIO(html))
    except Exception as exc:
        raise MarketDataError(f"جدول نرخ بهره قابل خواندن نیست: {exc}") from exc
    for table in tables:
        if "YTM" not in table.columns:
            continue
        ytm = table["YTM"].astype(str)
        ytm = ytm[~ytm.str.contains("سررسید", na=False)]
        values = pd.to_numeric(ytm.str.replace("/", ".", regex=False).str.rstrip("%"), errors="coerce").dropna()
        if len(values):
            return float(values.mean()) / 100.0
    raise MarketDataError("ستون YTM در صفحه نرخ بهره پیدا نشد.")


@dataclass
class MarketSnapshot:
    symbol: str
    ins_code: str
    chain: pd.DataFrame
    history: pd.DataFrame
    risk_free_rate: Optional[float]
    elapsed_seconds: float
    warnings: List[str] = field(default_factory=list)


def _is_cert_error(exc: BaseException) -> bool:
    seen = set()
    while exc is not None and id(exc) not in seen:
        seen.add(id(exc))
        if isinstance(exc, ssl.SSLCertVerificationError) or "CERTIFICATE_VERIFY_FAILED" in str(exc):
            return True
        exc = exc.__cause__ or exc.__context__
    return False


class TsetmcClient:
    def __init__(
        self,
        *,
        cdn_base: str = CDN_BASE,
        old_base: str = OLD_BASE,
        ytm_url: str = YTM_URL,
        timeout: float = 20.0,
        connect_timeout: float = 10.0,
        retries: int = 2,
        verify: Optional[bool] = None,
        transport: Optional[httpx.AsyncBaseTransport] = None,
        trust_env: bool = True,
        ytm_timeout: float = 6.0,
    ) -> None:
        self.cdn_base = cdn_base.rstrip("/")
        self.old_base = old_base.rstrip("/")
        self.ytm_url = ytm_url
        self._timeout = httpx.Timeout(timeout, connect=connect_timeout)
        self._ytm_timeout = httpx.Timeout(ytm_timeout, connect=min(connect_timeout, ytm_timeout))
        self._retries = max(0, int(retries))
        self._verify = verify
        self._transport = transport
        self._trust_env = trust_env
        self._clients: Dict[Tuple[bool, bool], httpx.AsyncClient] = {}
        self._tls_fallback = False
        self._proxy_fallback = False
        self._id_cache: Dict[str, str] = {}
        self.warnings: List[str] = []

    async def __aenter__(self) -> "TsetmcClient":
        return self

    async def __aexit__(self, *exc_info: Any) -> None:
        await self.aclose()

    async def aclose(self) -> None:
        for client in self._clients.values():
            await client.aclose()
        self._clients.clear()

    def _client(self, verify: bool, trust_env: bool) -> httpx.AsyncClient:
        key = (verify, trust_env)
        if key not in self._clients:
            kwargs: Dict[str, Any] = dict(
                headers={
                    "User-Agent": USER_AGENT,
                    "Accept": "application/json, text/plain, */*",
                    "Accept-Language": "fa-IR,fa;q=0.9,en-US;q=0.8,en;q=0.7",
                    "Referer": "https://tsetmc.com/",
                    "Origin": "https://tsetmc.com",
                },
                timeout=self._timeout,
                follow_redirects=True,
                trust_env=trust_env,
                limits=httpx.Limits(max_connections=20, max_keepalive_connections=10),
            )
            if self._transport is not None:
                kwargs["transport"] = self._transport
            else:
                kwargs["verify"] = verify
            self._clients[key] = httpx.AsyncClient(**kwargs)
        return self._clients[key]

    async def _get(self, url: str, *, timeout: Optional[httpx.Timeout] = None) -> httpx.Response:
        verify = False if self._tls_fallback else (True if self._verify is None else self._verify)
        trust_env = False if self._proxy_fallback else self._trust_env
        last_error: Optional[BaseException] = None
        attempt = 0
        while True:
            try:
                kwargs = {"timeout": timeout} if timeout is not None else {}
                response = await self._client(verify, trust_env).get(url, **kwargs)
                if response.status_code < 400:
                    return response
                if response.status_code < 500 and response.status_code != 429:
                    raise MarketDataError(f"پاسخ نامعتبر از سرور ({response.status_code}): {url}")
                last_error = httpx.HTTPStatusError("server error", request=response.request, response=response)
            except MarketDataError:
                raise
            except httpx.ConnectError as exc:
                last_error = exc
                if verify and self._verify is None and _is_cert_error(exc):
                    self._tls_fallback = True
                    verify = False
                    if TLS_WARNING not in self.warnings:
                        self.warnings.append(TLS_WARNING)
                    continue
                if trust_env and self._transport is None:
                    self._proxy_fallback = True
                    trust_env = False
                    if PROXY_WARNING not in self.warnings:
                        self.warnings.append(PROXY_WARNING)
                    continue
            except httpx.TransportError as exc:
                last_error = exc
            if attempt >= self._retries:
                break
            await asyncio.sleep(0.25 * (2 ** attempt))
            attempt += 1
        raise MarketDataError(
            f"اتصال به سرور ممکن نشد: {url} ({type(last_error).__name__}). "
            "اتصال اینترنت را بررسی کنید؛ اگر VPN یا پراکسی روشن است، خاموش/روشن کردن آن را امتحان کنید."
        )

    async def _get_json(self, url: str) -> Any:
        response = await self._get(url)
        try:
            return response.json()
        except ValueError as exc:
            raise MarketDataError(f"پاسخ JSON معتبر نیست: {url}") from exc

    async def option_watch(self, market: int) -> pd.DataFrame:
        payload = await self._get_json(f"{self.cdn_base}/api/Instrument/GetInstrumentOptionMarketWatch/{market}")
        return parse_option_watch(payload, "bourse" if market == 1 else "farabourse")

    async def option_chain(self, symbol: str) -> pd.DataFrame:
        results = await asyncio.gather(self.option_watch(1), self.option_watch(2), return_exceptions=True)
        frames = [r for r in results if isinstance(r, pd.DataFrame)]
        if not frames:
            raise MarketDataError(f"دریافت بازار اختیار معامله ناموفق بود: {results[0]}")
        for market, result in zip(("بورس", "فرابورس"), results):
            if isinstance(result, Exception):
                self.warnings.append(f"دریافت اختیارهای {market} ناموفق بود؛ نتیجه ممکن است ناقص باشد.")
        data = pd.concat([f for f in frames if not f.empty], ignore_index=True) if any(
            not f.empty for f in frames) else pd.DataFrame(columns=CHAIN_COLUMNS)
        target = canonical_symbol(symbol)
        chain = data[data["UA_Symbol"].map(canonical_symbol) == target].reset_index(drop=True)
        if chain.empty:
            raise MarketDataError(
                f"برای نماد «{symbol}» اختیار معامله فعالی پیدا نشد؛ نماد دارایی پایه (مثل اهرم) را وارد کنید."
            )
        return chain

    async def find_instrument_id(self, symbol: str) -> str:
        key = canonical_symbol(symbol)
        if key in self._id_cache:
            return self._id_cache[key]
        query = quote(to_tsetmc_form(symbol))
        candidates: List[str] = []
        found: Optional[str] = None
        try:
            payload = await self._get_json(f"{self.cdn_base}/api/Instrument/GetInstrumentSearch/{query}")
            for item in (payload or {}).get("instrumentSearch", []) or []:
                code, ticker = str(item.get("insCode", "")), str(item.get("lVal18AFC", ""))
                candidates.append(ticker)
                if code and canonical_symbol(ticker) == key:
                    found = code
                    break
        except MarketDataError:
            found = None
        if found is None:
            try:
                text = (await self._get(f"{self.old_base}/tsev2/data/search.aspx?skey={query}")).text
                for row in text.split(";"):
                    cells = row.split(",")
                    if len(cells) >= 3:
                        candidates.append(cells[0])
                        if canonical_symbol(cells[0]) == key and cells[2].strip():
                            found = cells[2].strip()
                            break
            except MarketDataError:
                found = None
        if found is None:
            similar = list(dict.fromkeys(c for c in candidates if c))[:5]
            hint = f" (نمادهای مشابه: {'، '.join(similar)})" if similar else ""
            raise MarketDataError(f"نماد «{symbol}» در بورس پیدا نشد{hint}.")
        self._id_cache[key] = found
        return found

    async def price_history(self, ins_code: str, days: int = 120) -> pd.DataFrame:
        days = max(int(days), 30)
        errors: List[str] = []
        try:
            payload = await self._get_json(f"{self.cdn_base}/api/ClosingPrice/GetClosingPriceDailyList/{ins_code}/{days}")
            return adjust_prices(parse_history_cdn(payload).tail(days))
        except MarketDataError as exc:
            errors.append(str(exc))
        try:
            text = (await self._get(f"{self.old_base}/tsev2/data/Export-txt.aspx?t=i&a=1&b=0&i={ins_code}")).text
            return adjust_prices(parse_history_csv(text).tail(days))
        except MarketDataError as exc:
            errors.append(str(exc))
        raise MarketDataError("دریافت تاریخچه قیمت ناموفق بود: " + " | ".join(errors))

    async def risk_free_rate(self) -> Optional[float]:
        try:
            response = await self._get(self.ytm_url, timeout=self._ytm_timeout)
            return parse_ytm_html(response.text)
        except MarketDataError:
            return None

    async def snapshot(self, symbol: str, history_days: int = 120) -> MarketSnapshot:
        started = time.perf_counter()
        chain, ins_code, rate = await asyncio.gather(
            self.option_chain(symbol), self.find_instrument_id(symbol), self.risk_free_rate(),
            return_exceptions=True,
        )
        for result in (chain, ins_code):
            if isinstance(result, BaseException):
                raise result
        if isinstance(rate, BaseException):
            rate = None
        history = await self.price_history(ins_code, history_days)
        return MarketSnapshot(
            symbol=symbol,
            ins_code=ins_code,
            chain=chain,
            history=history,
            risk_free_rate=rate,
            elapsed_seconds=time.perf_counter() - started,
            warnings=list(self.warnings),
        )


def run_sync(coro: Coroutine[Any, Any, T]) -> T:
    try:
        asyncio.get_running_loop()
    except RuntimeError:
        return asyncio.run(coro)
    box: Dict[str, Any] = {}

    def runner() -> None:
        try:
            box["value"] = asyncio.run(coro)
        except BaseException as exc:
            box["error"] = exc

    worker = threading.Thread(target=runner, daemon=True)
    worker.start()
    worker.join()
    if "error" in box:
        raise box["error"]
    return box["value"]


def fetch_snapshot(symbol: str, history_days: int = 120, **client_kwargs: Any) -> MarketSnapshot:
    async def _run() -> MarketSnapshot:
        async with TsetmcClient(**client_kwargs) as client:
            return await client.snapshot(symbol, history_days)

    return run_sync(_run())
