from __future__ import annotations

import threading
import tkinter as tk
from tkinter import filedialog, messagebox, ttk
from typing import Optional

import pandas as pd

from analytics_engine import enrich_chain, load_config
from decision_engine import score_and_rank, to_persian_dataframe
from market_data import fetch_snapshot


class OptionAnalyzerApp(tk.Tk):
    def __init__(self) -> None:
        super().__init__()
        self.config_data = load_config("config.json")
        self.title("تحلیلگر هوشمند اختیار معامله بورس تهران - نسخه ۲ (V2)")
        self.geometry("1320x780")
        self.minsize(1100, 650)

        self.result_df: Optional[pd.DataFrame] = None
        self.persian_df: Optional[pd.DataFrame] = None

        self._build_ui()

    def _build_ui(self) -> None:
        top = ttk.LabelFrame(self, text="تنظیمات تحلیل (بدون حذف قراردادها - رتبه‌بندی هوشمند دوطرفه)", padding=10)
        top.pack(fill="x", padx=10, pady=8)

        # Row 0
        ttk.Label(top, text="نماد دارایی پایه:").grid(row=0, column=0, sticky="e", padx=4, pady=4)
        self.symbol_var = tk.StringVar(value=str(self.config_data.get("default_symbol", "اهرم")))
        ttk.Entry(top, textvariable=self.symbol_var, width=14).grid(row=0, column=1, padx=4, pady=4)

        ttk.Label(top, text="دیدگاه بازار:").grid(row=0, column=2, sticky="e", padx=4, pady=4)
        self.view_var = tk.StringVar(value="صعودی")
        ttk.Combobox(top, textvariable=self.view_var, values=["صعودی", "نزولی", "خنثی"], state="readonly", width=11).grid(row=0, column=3, padx=4, pady=4)

        ttk.Label(top, text="سطح ریسک:").grid(row=0, column=4, sticky="e", padx=4, pady=4)
        self.risk_var = tk.StringVar(value="بالا")
        ttk.Combobox(top, textvariable=self.risk_var, values=["بالا", "متوسط", "کم"], state="readonly", width=10).grid(row=0, column=5, padx=4, pady=4)

        ttk.Label(top, text="مرتب‌سازی بر اساس:").grid(row=0, column=6, sticky="e", padx=4, pady=4)
        self.side_var = tk.StringVar(value="هر دو")
        ttk.Combobox(top, textvariable=self.side_var, values=["هر دو", "خرید", "فروش (اخذ تعهد)"], state="readonly", width=15).grid(row=0, column=7, padx=4, pady=4)

        # Row 1
        ttk.Label(top, text="سرمایه (ریال):").grid(row=1, column=0, sticky="e", padx=4, pady=4)
        self.capital_var = tk.StringVar(value=str(self.config_data.get("default_capital_rial", 500000000)))
        ttk.Entry(top, textvariable=self.capital_var, width=14).grid(row=1, column=1, padx=4, pady=4)

        ttk.Label(top, text="نرخ بهره (خالی=خودکار):").grid(row=1, column=2, sticky="e", padx=4, pady=4)
        self.rf_var = tk.StringVar(value=str(self.config_data.get("default_risk_free_rate", 0.34)))
        ttk.Entry(top, textvariable=self.rf_var, width=11).grid(row=1, column=3, padx=4, pady=4)

        ttk.Label(top, text="قیمت مرجع / NAV (اختیاری):").grid(row=1, column=4, sticky="e", padx=4, pady=4)
        self.nav_var = tk.StringVar(value="")
        ttk.Entry(top, textvariable=self.nav_var, width=12).grid(row=1, column=5, padx=4, pady=4)

        ttk.Label(top, text="روز تاریخچه نوسان:").grid(row=1, column=6, sticky="e", padx=4, pady=4)
        self.days_var = tk.StringVar(value=str(self.config_data.get("default_history_days", 100)))
        ttk.Entry(top, textvariable=self.days_var, width=10).grid(row=1, column=7, padx=4, pady=4)

        # Action buttons
        btn_frame = ttk.Frame(top)
        btn_frame.grid(row=0, column=8, rowspan=2, padx=12)
        self.run_btn = ttk.Button(btn_frame, text="اجرای تحلیل", command=self.start_analysis, width=16)
        self.run_btn.pack(pady=3)
        self.export_btn = ttk.Button(btn_frame, text="ذخیره کامل CSV", command=self.export_csv, width=16, state="disabled")
        self.export_btn.pack(pady=3)

        # Status bar
        self.status_var = tk.StringVar(value="آماده اجرا. نماد دارایی پایه (مثل اهرم) را وارد کرده و دکمه «اجرای تحلیل» را بزنید.")
        ttk.Label(self, textvariable=self.status_var, foreground="#1a4e8a", padding=(12, 2)).pack(fill="x")

        # Notebook tabs
        self.notebook = ttk.Notebook(self)
        self.notebook.pack(fill="both", expand=True, padx=10, pady=6)

        # Tab 1: Full Ranked Table
        tab_table = ttk.Frame(self.notebook)
        self.notebook.add(tab_table, text="جدول کامل و رتبه‌بندی (همه ستون‌ها)")

        self.tree = ttk.Treeview(tab_table, show="headings")
        vsb = ttk.Scrollbar(tab_table, orient="vertical", command=self.tree.yview)
        hsb = ttk.Scrollbar(tab_table, orient="horizontal", command=self.tree.xview)
        self.tree.configure(yscrollcommand=vsb.set, xscrollcommand=hsb.set)

        self.tree.grid(row=0, column=0, sticky="nsew")
        vsb.grid(row=0, column=1, sticky="ns")
        hsb.grid(row=1, column=0, sticky="ew")
        tab_table.grid_rowconfigure(0, weight=1)
        tab_table.grid_columnconfigure(0, weight=1)

        # Tab 2: Top Picks Summary
        tab_summary = ttk.Frame(self.notebook)
        self.notebook.add(tab_summary, text="خلاصه برترین‌ها (خرید و فروش)")
        self.summary_text = tk.Text(tab_summary, wrap="word", font=("Tahoma", 10), padx=12, pady=12)
        self.summary_text.pack(fill="both", expand=True)

    def start_analysis(self) -> None:
        sym = self.symbol_var.get().strip()
        if not sym:
            messagebox.showwarning("خطا", "لطفاً نام نماد دارایی پایه (مثلاً اهرم) را وارد کنید.")
            return

        self.run_btn.config(state="disabled")
        self.export_btn.config(state="disabled")
        self.status_var.set(f"در حال دریافت اطلاعات لحظه‌ای بازار برای «{sym}» ...")

        thread = threading.Thread(target=self._worker, args=(sym,), daemon=True)
        thread.start()

    def _worker(self, sym: str) -> None:
        try:
            history_days = int(self.days_var.get().strip() or 100)
            capital = float(self.capital_var.get().strip() or 500000000)
            nav_str = self.nav_var.get().strip()
            nav_override = float(nav_str) if nav_str else None
            rf_str = self.rf_var.get().strip()

            snap = fetch_snapshot(sym, history_days=history_days)

            if rf_str:
                rf = float(rf_str)
            elif snap.risk_free_rate is not None:
                rf = float(snap.risk_free_rate)
            else:
                rf = float(self.config_data.get("default_risk_free_rate", 0.34))

            enriched, s0, hv = enrich_chain(
                chain=snap.chain,
                history=snap.history,
                risk_free_rate=rf,
                capital_rial=capital,
                config=self.config_data,
                ref_price_override=nav_override,
            )
            ranked = score_and_rank(
                enriched,
                view=self.view_var.get(),
                risk=self.risk_var.get(),
                side_focus=self.side_var.get(),
            )
            persian_df = to_persian_dataframe(ranked)
            self.after(0, lambda: self._on_success(ranked, persian_df, s0, hv, rf, snap.elapsed_seconds, snap.warnings))
        except Exception as exc:
            self.after(0, lambda: self._on_error(str(exc)))

    def _on_success(self, ranked, persian_df, s0, hv, rf, elapsed, warnings) -> None:
        self.result_df = ranked
        self.persian_df = persian_df

        self.tree.delete(*self.tree.get_children())
        cols = list(persian_df.columns)
        self.tree["columns"] = cols
        for col in cols:
            self.tree.heading(col, text=col)
            self.tree.column(col, width=125, anchor="center")

        for _, row in persian_df.iterrows():
            vals = ["-" if pd.isna(v) else v for v in row.tolist()]
            self.tree.insert("", "end", values=vals)

        # Build summary tab
        self.summary_text.delete("1.0", "end")
        header = (
            f"نماد: {self.symbol_var.get()} | قیمت مرجع: {s0:,.0f} ریال | "
            f"نوسان تاریخی (HV): {hv*100:.2f}% | نرخ بهره: {rf*100:.1f}% | "
            f"تعداد کل قراردادها: {len(ranked)} | زمان دریافت: {elapsed:.1f} ثانیه\n"
        )
        self.summary_text.insert("end", header + "=" * 95 + "\n\n")

        top_buy = ranked[ranked["buy_score"] > 0].sort_values("buy_score", ascending=False).head(5)
        self.summary_text.insert("end", ">>> ۵ گزینه برتر برای خرید (Long):\n")
        for _, r in top_buy.iterrows():
            self.summary_text.insert(
                "end",
                f"  • {r['symbol']} ({r['type_fa']}) | اعمال: {r['strike']:,.0f} | روز: {r['days']} | "
                f"مظنه خرید: {r['ask']:,.0f} | امگا: {r['omega']} | نقدشوندگی: {r['liq_grade']} | "
                f"امتیاز خرید: {r['buy_score']}/100 | رشد+۲۰٪: {r['roi_p20']}% | افت-۲۰٪: {r['roi_m20']}%\n"
            )

        top_write = ranked[ranked["write_score"] > 0].sort_values("write_score", ascending=False).head(5)
        self.summary_text.insert("end", "\n>>> ۵ گزینه برتر برای فروش / اخذ تعهد (Write):\n")
        for _, r in top_write.iterrows():
            self.summary_text.insert(
                "end",
                f"  • {r['symbol']} ({r['type_fa']}) | اعمال: {r['strike']:,.0f} | روز: {r['days']} | "
                f"مظنه فروش: {r['bid']:,.0f} | ROM سالانه: {r['ann_rom_pct']}% | نسبت حباب IV/HV: {r['iv_hv_ratio']} | "
                f"نقدشوندگی: {r['liq_grade']} | امتیاز فروش: {r['write_score']}/100\n"
            )

        if warnings:
            self.summary_text.insert("end", "\nهشدارهای سیستم:\n" + "\n".join(f"  - {w}" for w in warnings))

        self.status_var.set(f"تحلیل کامل شد: {len(ranked)} قرارداد رتبه‌بندی شدند (قیمت مرجع: {s0:,.0f} ریال).")
        self.run_btn.config(state="normal")
        self.export_btn.config(state="normal")

    def _on_error(self, msg: str) -> None:
        self.status_var.set("خطا در اجرای تحلیل.")
        self.run_btn.config(state="normal")
        messagebox.showerror("خطا", msg)

    def export_csv(self) -> None:
        if self.persian_df is None or self.persian_df.empty:
            return
        path = filedialog.asksaveasfilename(
            defaultextension=".csv",
            filetypes=[("CSV Files", "*.csv")],
            initialfile=f"{self.symbol_var.get().strip()}_options_v2.csv",
        )
        if not path:
            return
        self.persian_df.to_csv(path, index=False, encoding="utf-8-sig")
        messagebox.showinfo("ذخیره شد", f"فایل خروجی کامل با {len(self.persian_df.columns)} ستون ذخیره شد:\n{path}")


if __name__ == "__main__":
    app = OptionAnalyzerApp()
    app.mainloop()
