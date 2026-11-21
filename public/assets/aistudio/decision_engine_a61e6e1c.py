from __future__ import annotations

from typing import Dict, Tuple

import pandas as pd


PERSIAN_COLUMNS: Dict[str, str] = {
    "symbol": "نماد آپشن",
    "type_fa": "نوع قرارداد",
    "strategy_tag": "پیشنهاد و استراتژی",
    "buy_score": "امتیاز خرید (۱۰۰)",
    "write_score": "امتیاز فروش (۱۰۰)",
    "liq_grade": "درجه نقدشوندگی",
    "strike": "قیمت اعمال",
    "exp_jalali": "تاریخ سررسید (شمسی)",
    "days": "روز تا سررسید",
    "horizon_tag": "افق زمانی",
    "moneyness": "وضعیت سوددهی",
    "bid": "بهترین خریدار (مظنه فروش)",
    "ask": "بهترین فروشنده (مظنه خرید)",
    "spread_pct": "شکاف قیمتی (٪)",
    "oi": "موقعیت‌های باز",
    "volume": "حجم معاملات",
    "omega": "اهرم موثر (امگا)",
    "nominal_lev": "اهرم اسمی",
    "delta": "دلتا",
    "gamma": "گاما",
    "theta_daily": "تتا (ریال/روز)",
    "theta_pct": "افت زمانی (٪/روز)",
    "vega": "وگا",
    "rho": "رو",
    "intrinsic": "ارزش ذاتی",
    "time_value": "ارزش زمانی",
    "time_val_pct": "سهم ارزش زمانی (٪)",
    "bs_price": "ارزش بلک-شولز",
    "p_to_bs": "نسبت قیمت به BS",
    "iv_ask_pct": "نوسان ضمنی خرید (٪)",
    "iv_bid_pct": "نوسان ضمنی فروش (٪)",
    "hv_pct": "نوسان تاریخی (٪)",
    "iv_hv_ratio": "نسبت حباب (IV/HV)",
    "breakeven": "نقطه سر به سر",
    "be_dist_pct": "فاصله تا سر به سر (٪)",
    "prob_profit_pct": "احتمال سود (٪)",
    "margin": "وجه تضمین (ریال)",
    "rom_pct": "بازده وجه تضمین ROM (٪)",
    "ann_rom_pct": "ROM سالانه (٪)",
    "max_buy_contracts": "تعداد مجاز خرید",
    "max_write_contracts": "تعداد مجاز فروش",
    "roi_p5": "سود/زیان رشد ۵٪",
    "roi_p10": "سود/زیان رشد ۱۰٪",
    "roi_p20": "سود/زیان رشد ۲۰٪",
    "roi_m5": "سود/زیان افت ۵٪",
    "roi_m10": "سود/زیان افت ۱۰٪",
    "roi_m20": "سود/زیان افت ۲۰٪",
}


def _clip(val: float, lo: float = 0.0, hi: float = 100.0) -> float:
    return max(lo, min(hi, val))


def _liquidity_grade(row: pd.Series) -> Tuple[str, float]:
    if not row["has_quote"] or (row["ask"] <= 0 and row["bid"] <= 0):
        return "F (بدون مظنه)", 0.0
    spread = row["spread_pct"]
    oi = row["oi"]

    if pd.isna(spread):
        return "D (یک‌طرفه)", 20.0
    if spread <= 1.5 and oi >= 10000:
        return "A (عالی)", 100.0
    if spread <= 3.5 and oi >= 2000:
        return "B (خوب)", 80.0
    if spread <= 8.0 and oi >= 300:
        return "C (متوسط)", 55.0
    return "D (ضعیف)", 30.0


def score_and_rank(
    df: pd.DataFrame,
    view: str = "صعودی",
    risk: str = "بالا",
    side_focus: str = "هر دو",
) -> pd.DataFrame:
    if df.empty:
        return df

    out = df.copy()
    liq_grades = []
    buy_scores = []
    write_scores = []
    tags = []

    for _, r in out.iterrows():
        grade, liq_pts = _liquidity_grade(r)
        liq_grades.append(grade)

        if liq_pts == 0.0:
            buy_scores.append(0.0)
            write_scores.append(0.0)
            tags.append("بدون مظنه (غیرفعال)")
            continue

        is_call = r["type"].lower() == "call"
        omega = 0.0 if pd.isna(r["omega"]) else float(r["omega"])
        theta_pct = abs(float(r["theta_pct"])) if not pd.isna(r["theta_pct"]) else 10.0
        be_dist = abs(float(r["be_dist_pct"])) if not pd.isna(r["be_dist_pct"]) else 50.0
        iv_hv = float(r["iv_hv_ratio"]) if not pd.isna(r["iv_hv_ratio"]) else 1.0
        ann_rom = float(r["ann_rom_pct"]) if not pd.isna(r["ann_rom_pct"]) else 0.0
        prob = float(r["prob_profit_pct"]) if not pd.isna(r["prob_profit_pct"]) else 20.0
        days = int(r["days"])

        if risk == "بالا":
            omega_pts = _clip((omega / 6.5) * 100.0) if omega <= 10 else _clip(100.0 - (omega - 10) * 5)
            theta_penalty = _clip(theta_pct * 8.0, 0, 45)
        elif risk == "متوسط":
            omega_pts = _clip((omega / 4.5) * 100.0) if omega <= 6 else _clip(100.0 - (omega - 6) * 10)
            theta_penalty = _clip(theta_pct * 12.0, 0, 60)
        else:
            omega_pts = _clip((omega / 3.0) * 100.0) if omega <= 4 else _clip(100.0 - (omega - 4) * 15)
            theta_penalty = _clip(theta_pct * 18.0, 0, 75)

        be_pts = _clip(100.0 - be_dist * 3.0)
        val_pts = _clip(100.0 - max(0.0, iv_hv - 1.0) * 35.0)

        b_score = (
            0.35 * liq_pts
            + 0.30 * omega_pts
            + 0.20 * be_pts
            + 0.15 * val_pts
            - 0.20 * theta_penalty
        )

        if view == "صعودی" and not is_call:
            b_score *= 0.55
        elif view == "نزولی" and is_call:
            b_score *= 0.55

        if r["ask"] <= 0:
            b_score = 0.0
        b_score = round(_clip(b_score), 1)
        buy_scores.append(b_score)

        bubble_pts = _clip((iv_hv - 0.8) * 70.0)
        rom_pts = _clip((ann_rom / 80.0) * 100.0)
        otm_prob_pts = _clip(100.0 - prob)
        theta_gain_pts = _clip(theta_pct * 20.0)

        w_score = (
            0.30 * liq_pts
            + 0.30 * bubble_pts
            + 0.25 * rom_pts
            + 0.15 * otm_prob_pts
            + 0.10 * theta_gain_pts
        )

        if view == "صعودی" and is_call:
            w_score *= 0.70
        elif view == "نزولی" and not is_call:
            w_score *= 0.70

        if r["bid"] <= 0:
            w_score = 0.0
        w_score = round(_clip(w_score), 1)
        write_scores.append(w_score)

        if b_score >= 65 and b_score >= w_score:
            if days <= 14:
                tag = "نوسان‌گیری سریع (خرید)"
            elif omega >= 4.5:
                tag = "خرید پرشتاب اهرمی"
            else:
                tag = "خرید متعادل"
        elif w_score >= 70 and w_score > b_score:
            tag = "شکار حباب / اخذ تعهد (فروش)"
        elif b_score >= 50:
            tag = "مناسب خرید (با مدیریت ریسک)"
        elif w_score >= 55:
            tag = "مناسب فروش (اخذ تعهد)"
        else:
            tag = "زیر نظر / کم‌جذابیت"
        tags.append(tag)

    out["liq_grade"] = liq_grades
    out["buy_score"] = buy_scores
    out["write_score"] = write_scores
    out["strategy_tag"] = tags

    if side_focus == "خرید":
        sort_col = "buy_score"
    elif side_focus == "فروش (اخذ تعهد)":
        sort_col = "write_score"
    else:
        out["_best"] = out[["buy_score", "write_score"]].max(axis=1)
        sort_col = "_best"

    out = out.sort_values(by=[sort_col, "oi"], ascending=[False, False]).reset_index(drop=True)
    if "_best" in out.columns:
        out = out.drop(columns=["_best"])
    return out


def to_persian_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    cols = [c for c in PERSIAN_COLUMNS.keys() if c in df.columns]
    return df[cols].rename(columns=PERSIAN_COLUMNS)
