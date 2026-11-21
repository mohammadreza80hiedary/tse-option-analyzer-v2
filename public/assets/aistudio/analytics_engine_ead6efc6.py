from __future__ import annotations

import json
import math
from pathlib import Path
from typing import Any, Dict, Optional, Tuple

import jdatetime
import numpy as np
import pandas as pd


DEFAULT_CONFIG: Dict[str, Any] = {
    "default_symbol": "اهرم",
    "default_risk_free_rate": 0.34,
    "default_capital_rial": 500000000,
    "default_history_days": 100,
    "option_buy_commission": 0.00103,
    "option_sell_commission": 0.00103,
    "trading_days_per_year": 240,
}


def load_config(path: str = "config.json") -> Dict[str, Any]:
    cfg = dict(DEFAULT_CONFIG)
    p = Path(path)
    if p.exists():
        try:
            user_cfg = json.loads(p.read_text(encoding="utf-8"))
            if isinstance(user_cfg, dict):
                cfg.update(user_cfg)
        except Exception:
            pass
    else:
        try:
            p.write_text(json.dumps(cfg, indent=2, ensure_ascii=False), encoding="utf-8")
        except Exception:
            pass
    return cfg


def _norm_cdf(x: float) -> float:
    return 0.5 * (1.0 + math.erf(x / math.sqrt(2.0)))


def _norm_pdf(x: float) -> float:
    return math.exp(-0.5 * x * x) / math.sqrt(2.0 * math.pi)


def compute_historical_volatility(history: pd.DataFrame, trading_days: int = 240) -> float:
    col = "Adj Close" if "Adj Close" in history.columns else "Close"
    prices = pd.to_numeric(history[col], errors="coerce").dropna()
    prices = prices[prices > 0]
    if len(prices) < 5:
        return 0.45
    log_ret = np.log(prices / prices.shift(1)).dropna()
    hv = float(log_ret.std(ddof=1) * math.sqrt(trading_days))
    return max(0.05, min(hv, 3.0))


def bs_price_and_greeks(
    S: float,
    K: float,
    T: float,
    r: float,
    sigma: float,
    option_type: str,
    q: float = 0.0,
) -> Dict[str, float]:
    is_call = option_type.lower() == "call"
    if S <= 0 or K <= 0 or T <= 0 or sigma <= 0:
        intrinsic = max(0.0, S - K) if is_call else max(0.0, K - S)
        return {
            "bs_price": intrinsic,
            "delta": 1.0 if (is_call and S > K) else (-1.0 if (not is_call and K > S) else 0.0),
            "gamma": 0.0,
            "theta_daily": 0.0,
            "vega": 0.0,
            "rho": 0.0,
        }

    sqrt_T = math.sqrt(T)
    d1 = (math.log(S / K) + (r - q + 0.5 * sigma * sigma) * T) / (sigma * sqrt_T)
    d2 = d1 - sigma * sqrt_T

    disc_r = math.exp(-r * T)
    disc_q = math.exp(-q * T)
    pdf_d1 = _norm_pdf(d1)

    if is_call:
        price = S * disc_q * _norm_cdf(d1) - K * disc_r * _norm_cdf(d2)
        delta = disc_q * _norm_cdf(d1)
        theta_annual = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            - r * K * disc_r * _norm_cdf(d2)
            + q * S * disc_q * _norm_cdf(d1)
        )
        rho = K * T * disc_r * _norm_cdf(d2) / 100.0
    else:
        price = K * disc_r * _norm_cdf(-d2) - S * disc_q * _norm_cdf(-d1)
        delta = disc_q * (_norm_cdf(d1) - 1.0)
        theta_annual = (
            -(S * disc_q * pdf_d1 * sigma) / (2.0 * sqrt_T)
            + r * K * disc_r * _norm_cdf(-d2)
            - q * S * disc_q * _norm_cdf(-d1)
        )
        rho = -K * T * disc_r * _norm_cdf(-d2) / 100.0

    gamma = (disc_q * pdf_d1) / (S * sigma * sqrt_T)
    vega = (S * disc_q * pdf_d1 * sqrt_T) / 100.0

    return {
        "bs_price": max(0.0, price),
        "delta": delta,
        "gamma": gamma,
        "theta_daily": theta_annual / 365.0,
        "vega": vega,
        "rho": rho,
    }


def implied_volatility(
    market_price: float,
    S: float,
    K: float,
    T: float,
    r: float,
    option_type: str,
    q: float = 0.0,
) -> float:
    if market_price <= 0 or S <= 0 or K <= 0 or T <= 0:
        return float("nan")
    is_call = option_type.lower() == "call"
    intrinsic = max(0.0, S * math.exp(-q * T) - K * math.exp(-r * T)) if is_call else max(
        0.0, K * math.exp(-r * T) - S * math.exp(-q * T)
    )
    if market_price <= intrinsic:
        return float("nan")

    low, high = 0.01, 5.0
    for _ in range(50):
        mid = 0.5 * (low + high)
        val = bs_price_and_greeks(S, K, T, r, mid, option_type, q)["bs_price"]
        if abs(val - market_price) < 1e-3:
            return mid
        if val > market_price:
            high = mid
        else:
            low = mid
    return 0.5 * (low + high)


def estimate_margin(S: float, K: float, option_type: str, contract_size: float = 1000.0) -> float:
    is_call = option_type.lower() == "call"
    otm = max(0.0, K - S) if is_call else max(0.0, S - K)
    per_share = max(0.20 * S - otm, 0.10 * S)
    return round(per_share * contract_size, -5)


def _to_jalali(dt: Any) -> str:
    if pd.isna(dt):
        return "-"
    try:
        j = jdatetime.date.fromgregorian(date=pd.Timestamp(dt).date())
        return j.strftime("%Y/%m/%d")
    except Exception:
        return "-"


def _scenario_roi(
    S0: float, K: float, entry_price: float, pct_move: float, is_call: bool, buy_comm: float
) -> float:
    if entry_price <= 0:
        return float("nan")
    cost = entry_price * (1.0 + buy_comm)
    s_end = S0 * (1.0 + pct_move)
    intrinsic = max(0.0, s_end - K) if is_call else max(0.0, K - s_end)
    return ((intrinsic - cost) / cost) * 100.0


def enrich_chain(
    chain: pd.DataFrame,
    history: pd.DataFrame,
    risk_free_rate: float,
    capital_rial: float,
    config: Dict[str, Any],
    ref_price_override: Optional[float] = None,
    dividend_yield: float = 0.0,
) -> Tuple[pd.DataFrame, float, float]:
    hv = compute_historical_volatility(history, int(config.get("trading_days_per_year", 240)))
    buy_comm = float(config.get("option_buy_commission", 0.00103))
    sell_comm = float(config.get("option_sell_commission", 0.00103))

    if ref_price_override and ref_price_override > 0:
        S0 = float(ref_price_override)
    else:
        valid_s = pd.to_numeric(chain["Stock_Price"], errors="coerce").dropna()
        valid_s = valid_s[valid_s > 0]
        S0 = float(valid_s.iloc[0]) if not valid_s.empty else float(history["Close"].iloc[-1])

    rows = []
    for _, row in chain.iterrows():
        sym = str(row["Symbol"])
        opt_type = str(row["Type"])
        is_call = opt_type.lower() == "call"
        K = float(row["Strike_Price"])
        days = max(int(row["Days_to_Maturity"]), 0)
        T = max(days, 1) / 365.0
        csize = float(row["Contract_Size"]) if row["Contract_Size"] > 0 else 1000.0

        bid = float(row["Best_Bid"])
        ask = float(row["Best_Ask"])
        last = float(row["Last_Price"])
        oi = float(row["Open_Interest"])
        vol = float(row["Volume"])

        has_quote = (ask > 0) or (bid > 0)
        entry_buy = ask if ask > 0 else last
        entry_sell = bid if bid > 0 else last

        if ask > 0 and bid > 0:
            mid = 0.5 * (ask + bid)
            spread_pct = ((ask - bid) / mid) * 100.0 if mid > 0 else float("nan")
        else:
            spread_pct = float("nan")

        bs_hv = bs_price_and_greeks(S0, K, T, risk_free_rate, hv, opt_type, dividend_yield)
        bs_price = bs_hv["bs_price"]

        iv_ask = implied_volatility(ask, S0, K, T, risk_free_rate, opt_type, dividend_yield) if ask > 0 else float("nan")
        iv_bid = implied_volatility(bid, S0, K, T, risk_free_rate, opt_type, dividend_yield) if bid > 0 else float("nan")

        sigma_for_greeks = iv_ask if (not math.isnan(iv_ask) and iv_ask > 0.02) else hv
        greeks = bs_price_and_greeks(S0, K, T, risk_free_rate, sigma_for_greeks, opt_type, dividend_yield)

        intrinsic = max(0.0, S0 - K) if is_call else max(0.0, K - S0)
        time_value = max(0.0, entry_buy - intrinsic) if entry_buy > 0 else float("nan")
        time_val_pct = (time_value / entry_buy * 100.0) if (entry_buy > 0 and not math.isnan(time_value)) else float("nan")

        moneyness = "در سود (ITM)" if intrinsic > 0 else ("سر به سر (ATM)" if abs(S0 - K) / S0 < 0.015 else "در زیان (OTM)")
        nominal_lev = (S0 / entry_buy) if entry_buy > 0 else float("nan")
        omega = (abs(greeks["delta"]) * nominal_lev) if not math.isnan(nominal_lev) else float("nan")
        theta_pct = (greeks["theta_daily"] / entry_buy * 100.0) if entry_buy > 0 else float("nan")

        breakeven = (K + entry_buy * (1.0 + buy_comm)) if (is_call and entry_buy > 0) else (
            (K - entry_buy * (1.0 + buy_comm)) if (not is_call and entry_buy > 0) else float("nan")
        )
        be_dist_pct = ((breakeven - S0) / S0 * 100.0) if (not math.isnan(breakeven) and S0 > 0) else float("nan")

        if not math.isnan(breakeven) and breakeven > 0 and T > 0:
            d2_be = (math.log(S0 / breakeven) + (risk_free_rate - dividend_yield - 0.5 * sigma_for_greeks ** 2) * T) / (
                sigma_for_greeks * math.sqrt(T)
            )
            prob_profit = (_norm_cdf(d2_be) if is_call else _norm_cdf(-d2_be)) * 100.0
        else:
            prob_profit = float("nan")

        margin = estimate_margin(S0, K, opt_type, csize)
        if entry_sell > 0 and margin > 0:
            net_premium = entry_sell * (1.0 - sell_comm) * csize
            rom_pct = (net_premium / margin) * 100.0
            ann_rom_pct = rom_pct * (365.0 / max(days, 1))
        else:
            rom_pct = float("nan")
            ann_rom_pct = float("nan")

        max_buy_contracts = int(capital_rial // (entry_buy * (1.0 + buy_comm) * csize)) if entry_buy > 0 else 0
        max_write_contracts = int(capital_rial // margin) if margin > 0 else 0

        iv_hv_ratio = (iv_ask / hv) if (not math.isnan(iv_ask) and hv > 0) else float("nan")
        p_to_bs = (entry_buy / bs_price) if (entry_buy > 0 and bs_price > 0.5) else float("nan")

        rows.append({
            "symbol": sym,
            "type": opt_type,
            "type_fa": "اختیار خرید (Call)" if is_call else "اختیار فروش (Put)",
            "strike": K,
            "exp_jalali": _to_jalali(row["Expiration_Date"]),
            "days": days,
            "horizon_tag": "کوتاه‌مدت (<۱۴ روز)" if days <= 14 else ("میان‌مدت (۱۵-۶۵ روز)" if days <= 65 else "بلندمدت (>۶۵ روز)"),
            "moneyness": moneyness,
            "bid": bid,
            "ask": ask,
            "last": last,
            "has_quote": has_quote,
            "spread_pct": round(spread_pct, 2) if not math.isnan(spread_pct) else np.nan,
            "oi": int(oi),
            "volume": int(vol),
            "intrinsic": round(intrinsic, 1),
            "time_value": round(time_value, 1) if not math.isnan(time_value) else np.nan,
            "time_val_pct": round(time_val_pct, 1) if not math.isnan(time_val_pct) else np.nan,
            "bs_price": round(bs_price, 1),
            "p_to_bs": round(p_to_bs, 2) if not math.isnan(p_to_bs) else np.nan,
            "iv_ask_pct": round(iv_ask * 100.0, 2) if not math.isnan(iv_ask) else np.nan,
            "iv_bid_pct": round(iv_bid * 100.0, 2) if not math.isnan(iv_bid) else np.nan,
            "hv_pct": round(hv * 100.0, 2),
            "iv_hv_ratio": round(iv_hv_ratio, 2) if not math.isnan(iv_hv_ratio) else np.nan,
            "delta": round(greeks["delta"], 3),
            "gamma": round(greeks["gamma"], 6),
            "theta_daily": round(greeks["theta_daily"], 2),
            "theta_pct": round(theta_pct, 2) if not math.isnan(theta_pct) else np.nan,
            "vega": round(greeks["vega"], 2),
            "rho": round(greeks["rho"], 2),
            "nominal_lev": round(nominal_lev, 2) if not math.isnan(nominal_lev) else np.nan,
            "omega": round(omega, 2) if not math.isnan(omega) else np.nan,
            "breakeven": round(breakeven, 1) if not math.isnan(breakeven) else np.nan,
            "be_dist_pct": round(be_dist_pct, 2) if not math.isnan(be_dist_pct) else np.nan,
            "prob_profit_pct": round(prob_profit, 2) if not math.isnan(prob_profit) else np.nan,
            "margin": margin,
            "rom_pct": round(rom_pct, 2) if not math.isnan(rom_pct) else np.nan,
            "ann_rom_pct": round(ann_rom_pct, 1) if not math.isnan(ann_rom_pct) else np.nan,
            "max_buy_contracts": max_buy_contracts,
            "max_write_contracts": max_write_contracts,
            "roi_m20": round(_scenario_roi(S0, K, entry_buy, -0.20, is_call, buy_comm), 1),
            "roi_m10": round(_scenario_roi(S0, K, entry_buy, -0.10, is_call, buy_comm), 1),
            "roi_m5": round(_scenario_roi(S0, K, entry_buy, -0.05, is_call, buy_comm), 1),
            "roi_p5": round(_scenario_roi(S0, K, entry_buy, 0.05, is_call, buy_comm), 1),
            "roi_p10": round(_scenario_roi(S0, K, entry_buy, 0.10, is_call, buy_comm), 1),
            "roi_p20": round(_scenario_roi(S0, K, entry_buy, 0.20, is_call, buy_comm), 1),
        })

    return pd.DataFrame(rows), S0, hv
