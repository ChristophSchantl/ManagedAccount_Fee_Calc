# Managed Account Fee Calculator (Investor-ready, Streamlit Cloud safe)
# Dependencies: streamlit, pandas, numpy, plotly, openpyxl

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# App Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(page_title="FeeEngine – Managed Account Fees", page_icon="📈", layout="wide")

st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
      div[data-testid="stMetricValue"] { font-size: 1.45rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.9rem; opacity: 0.85; }
      .smallnote { font-size: 0.85rem; opacity: 0.80; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes (NO Streamlit UI code here!)
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeeParams:
    mgmt_fee_pa: float              # e.g. 0.02
    perf_fee: float                 # e.g. 0.20
    min_mgmt_fee_monthly: float     # monthly floor, NAV-relevant (month-end true-up)
    start_nav: float                # e.g. 1_000_000
    daycount: int                   # 360/365
    perf_crystallization: str       # "daily" or "monthly"
    date_col: str
    price_col: str
    resample_bdays: bool            # optional hardening


# ──────────────────────────────────────────────────────────────────────────────
# Helpers (robust parsing + Arrow-safe display)
# ──────────────────────────────────────────────────────────────────────────────
def _to_float_pct(x: float) -> float:
    return x / 100.0 if x > 1.0 else x


def to_number_series(s: pd.Series) -> pd.Series:
    """Robust numeric parser for EU/US formats."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace("\u00A0", "", regex=False)  # non-breaking space
    s2 = s2.str.replace(" ", "", regex=False)

    last_comma = s2.str.rfind(",")
    last_dot = s2.str.rfind(".")
    eu_mask = (last_comma > last_dot) & s2.str.contains(",")

    s2_eu = s2.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s2_us = s2.str.replace(",", "", regex=False)

    s3 = pd.Series(np.where(eu_mask, s2_eu, s2_us), index=s.index)
    return pd.to_numeric(s3, errors="coerce")


def parse_prices(upload: Optional[io.BytesIO], date_col: str, price_col: str) -> pd.DataFrame:
    if upload is None:
        idx = pd.bdate_range("2023-01-02", periods=520)
        rng = np.random.default_rng(7)
        rets = rng.normal(loc=0.00025, scale=0.012, size=len(idx))
        prices = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame({date_col: idx, price_col: prices})

    name = getattr(upload, "name", "").lower()
    raw = upload.read()

    try:
        if name.endswith(".xlsx") or name.endswith(".xls"):
            df = pd.read_excel(io.BytesIO(raw))
        else:
            df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python", on_bad_lines="warn")
    except Exception as e:
        raise ValueError(f"Datei-Parsing-Fehler: {e}")

    if df is None or df.empty:
        raise ValueError("Die Datei ist leer oder konnte nicht geparst werden.")

    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Spalten nicht gefunden. Erwartet: '{date_col}' und '{price_col}'.")

    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = to_number_series(df[price_col])
    df = df.dropna().sort_values(date_col).drop_duplicates(subset=[date_col])
    return df


def cagr(start: float, end: float, days: int) -> float:
    if days <= 0 or start <= 0:
        return np.nan
    years = days / 365.0
    return (end / start) ** (1.0 / years) - 1.0


def ann_vol_log(nav: pd.Series) -> float:
    nav = nav.replace([0, np.inf, -np.inf], np.nan).dropna()
    if len(nav) < 3:
        return np.nan
    r = np.log(nav).diff().dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(252))


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x*100:,.2f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Fee Engine (inkl. Min. Mgmt Fee/Monat als Month-End True-Up) — NAV-relevant
# ──────────────────────────────────────────────────────────────────────────────
def compute_fee_engine(df_prices: pd.DataFrame, p: FeeParams) -> pd.DataFrame:
    df = df_prices.copy().sort_values(p.date_col).reset_index(drop=True)
    df = df.rename(columns={p.date_col: "Date", p.price_col: "Close"})

    if p.resample_bdays:
        df = df.set_index("Date").asfreq("B").ffill().reset_index()

    df["Tage"] = df["Date"].diff().dt.days.fillna(0).astype(int)
    df.loc[df["Tage"] < 0, "Tage"] = 0

    base_price = float(df.loc[0, "Close"])
    df["Brutto_Rendite"] = df["Close"] / base_price
    df["NAV_gross"] = p.start_nav * df["Brutto_Rendite"]

    # Arrow-safe month key (string)
    df["Month"] = df["Date"].dt.strftime("%Y-%m")

    # Mgmt Fee: daily accrual (MF_Base) + month-end True-Up to reach floor (NAV-relevant)
    df["MF_Base"] = df["NAV_gross"] * (p.mgmt_fee_pa * df["Tage"] / p.daycount)
    df.loc[df["Tage"] == 0, "MF_Base"] = 0.0

    # detect month-end
    months = df["Month"].values
    is_month_end = np.zeros(len(df), dtype=bool)
    if len(df) > 0:
        is_month_end[-1] = True
    if len(df) > 1:
        is_month_end[:-1] = months[:-1] != months[1:]

    df["MF_MinAdj"] = 0.0
    floor = float(p.min_mgmt_fee_monthly or 0.0)

    if floor > 0:
        mf_month_sum = df.groupby("Month")["MF_Base"].sum()
        shortfall = (floor - mf_month_sum).clip(lower=0.0)

        # allocate shortfall to last row of each month (true-up)
        for m_key, adj in shortfall.items():
            if adj > 0:
                idx_last = df.index[(df["Month"] == m_key) & (is_month_end)].tolist()
                if idx_last:
                    df.at[idx_last[0], "MF_MinAdj"] = float(adj)

    df["MF_Amount"] = df["MF_Base"] + df["MF_MinAdj"]
    df["NAV_nach_MF"] = df["NAV_gross"] - df["MF_Amount"]

    # Performance fee + HWM (basis: NAV_nach_MF)
    df["HWM_alt"] = np.nan
    df["PF_Basis"] = 0.0
    df["PF_Amount"] = 0.0
    df["NAV_net"] = np.nan
    df["HWM_neu"] = np.nan

    hwm = float(p.start_nav)

    if p.perf_crystallization == "daily":
        for i in range(len(df)):
            df.at[i, "HWM_alt"] = hwm
            nav_after_mf = float(df.at[i, "NAV_nach_MF"])
            pf_basis = max(0.0, nav_after_mf - hwm)
            pf_amt = pf_basis * p.perf_fee
            nav_net = nav_after_mf - pf_amt
            hwm_new = nav_net if pf_basis > 0 else hwm

            df.at[i, "PF_Basis"] = pf_basis
            df.at[i, "PF_Amount"] = pf_amt
            df.at[i, "NAV_net"] = nav_net
            df.at[i, "HWM_neu"] = hwm_new
            hwm = hwm_new
    else:
        for i in range(len(df)):
            df.at[i, "HWM_alt"] = hwm
            nav_after_mf = float(df.at[i, "NAV_nach_MF"])
            month_end = is_month_end[i]

            pf_basis = 0.0
            pf_amt = 0.0
            nav_net = nav_after_mf
            hwm_new = hwm

            if month_end:
                pf_basis = max(0.0, nav_after_mf - hwm)
                pf_amt = pf_basis * p.perf_fee
                nav_net = nav_after_mf - pf_amt
                hwm_new = nav_net if pf_basis > 0 else hwm
                hwm = hwm_new

            df.at[i, "PF_Basis"] = pf_basis
            df.at[i, "PF_Amount"] = pf_amt
            df.at[i, "NAV_net"] = nav_net
            df.at[i, "HWM_neu"] = hwm_new

    # Cum sums + indices + drawdowns
    df["MF_kum"] = df["MF_Amount"].cumsum()
    df["PF_kum"] = df["PF_Amount"].cumsum()
    df["Fees_kum_total"] = df["MF_kum"] + df["PF_kum"]

    df["NAV_gross_idx"] = df["NAV_gross"] / p.start_nav
    df["NAV_net_idx"] = df["NAV_net"] / p.start_nav
    df["DD_gross"] = df["NAV_gross"] / df["NAV_gross"].cummax() - 1.0
    df["DD_net"] = df["NAV_net"] / df["NAV_net"].cummax() - 1.0

    # Monthly summary
    m = df.groupby("Month", as_index=False).agg(
        NAV_gross=("NAV_gross", "last"),
        NAV_net=("NAV_net", "last"),
        MF=("MF_Amount", "sum"),
        MF_Base=("MF_Base", "sum"),
        MF_MinAdj=("MF_MinAdj", "sum"),
        PF=("PF_Amount", "sum"),
        Fees=("Fees_kum_total", "last"),
    )

    m["Month_date"] = pd.to_datetime(m["Month"] + "-01")
    m["FeeRate_MF_bps"] = np.where(m["NAV_gross"] > 0, (m["MF"] / m["NAV_gross"]) * 1e4, np.nan)
    m["FeeRate_total_bps"] = np.where(m["NAV_gross"] > 0, ((m["MF"] + m["PF"]) / m["NAV_gross"]) * 1e4, np.nan)

    df.attrs["monthly"] = m
    return df


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("Managed Account Fee Calculator")
st.caption("Asset Manager: Christoph Schantl")

with st.sidebar:
    st.header("Inputs")
    upload = st.file_uploader("Preisserie (CSV/XLSX) hochladen", type=["csv", "xlsx", "xls"])

    st.subheader("Spaltenmapping")
    date_col = st.text_input("Datum-Spalte", value="Date")
    price_col = st.text_input("Preis-Spalte", value="Close")

    st.subheader("Fee-Parameter (NAV-relevant)")
    mgmt_fee_ui = st.number_input("Mgmt_Fee_p_a (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25)
    perf_fee_ui = st.number_input("Perf_Fee (%)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)
    start_nav = st.number_input("Start_NAV", min_value=1_000.0, max_value=1_000_000_000.0, value=1_000_000.0, step=10_000.0)

    # Floor (Minimum) bleibt — entspricht deinem "Fixum" in der Diskussion
    min_mgmt_fee_monthly = st.number_input(
        "Management Fee Minimum / Floor (€/Monat, NAV-relevant)",
        min_value=0.0, max_value=1_000_000.0,
        value=3000.0, step=100.0,
        key="min_mgmt_fee_monthly"
    )

    daycount = st.selectbox("Daycount", options=[360, 365], index=1)

    st.subheader("Performance Fee Modus")
    perf_mode = st.radio(
        "Crystallization",
        options=["monthly", "daily"],
        index=0,
        help="monthly = Monatsultimo (typisch). daily = tägliche Crystallization (aggressiver Modell-Case).",
    )

    st.subheader("Daten-Härtung")
    resample_bdays = st.checkbox("Business-Day Resample + Forward Fill", value=False)

    st.divider()

    # INFO BLOCK: Spain payroll assumptions (only informational; NOT NAV-relevant)
    with st.expander("ℹ️ Spanien (Balearen) – Netto/Abgaben (nur Info)", expanded=True):
        st.caption("Diese Annahmen dienen nur zur Information (FO/Investor). Sie beeinflussen den NAV nicht.")

        with st.form("pm_costs_form"):
            employer_social_rate = st.number_input(
                "AG Sozialabgaben (%)",
                min_value=0.0, max_value=50.0,
                value=30.0, step=1.0,
                key="ag_social_rate_pct"
            ) / 100.0

            employee_social_rate = st.number_input(
                "AN Sozialabgaben (%)",
                min_value=0.0, max_value=20.0,
                value=6.35, step=0.25,
                key="an_social_rate_pct"
            ) / 100.0

            employee_irpf_rate = st.number_input(
                "IRPF effektiv (Balearen, %)",
                min_value=0.0, max_value=40.0,
                value=17.0, step=0.5,
                key="irpf_rate_pct"
            ) / 100.0

            st.form_submit_button("Apply / Recalculate")

        st.markdown(
            '<div class="smallnote">'
            'Default: Balearen ~6.35 % AN-Sozialabgaben, ~16–18 % IRPF effektiv '
            '(vereinfachte Näherung, keine Steuerberatung).'
            '</div>',
            unsafe_allow_html=True
        )

    st.divider()
    st.subheader("Charts")
    show_gross = st.checkbox("Gross NAV anzeigen", value=True)



# ──────────────────────────────────────────────────────────────────────────────
# Compute
# ──────────────────────────────────────────────────────────────────────────────
try:
    df_prices = parse_prices(upload, date_col, price_col)
    params = FeeParams(
        mgmt_fee_pa=_to_float_pct(float(mgmt_fee_ui)),
        perf_fee=_to_float_pct(float(perf_fee_ui)),
        min_mgmt_fee_monthly=float(min_mgmt_fee_monthly),
        start_nav=float(start_nav),
        daycount=int(daycount),
        perf_crystallization=str(perf_mode),
        date_col=str(date_col),
        price_col=str(price_col),
        resample_bdays=bool(resample_bdays),
    )

    df = compute_fee_engine(df_prices, params)
    m = df.attrs["monthly"]
except Exception as e:
    st.error(f"Input/Parsing/Compute Fehler: {e}")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# Spain payroll INFO (OPEX / compensation view; NOT NAV-relevant)
# Convention: "Fixum" = Floor (Minimum Mgmt Fee per month)
# ──────────────────────────────────────────────────────────────────────────────
pm_gross_month = float(params.min_mgmt_fee_monthly)  # Fixum-Interpretation
pm_gross_year = pm_gross_month * 12.0

employer_cost_month = pm_gross_month * (1.0 + float(employer_social_rate))
employer_cost_year = employer_cost_month * 12.0

employee_net_month = (
    pm_gross_month
    * (1.0 - float(employee_social_rate))
    * (1.0 - float(employee_irpf_rate))
)
employee_net_year = employee_net_month * 12.0

# extra: realized/avg mgmt fee (useful for FO economics)
avg_mf_month = float(m["MF"].mean()) if len(m) else np.nan
binds_floor_share = float((m["MF_MinAdj"] > 0).mean()) if len(m) else np.nan  # share of months where floor binds


# ──────────────────────────────────────────────────────────────────────────────
# KPIs
# ──────────────────────────────────────────────────────────────────────────────
start_date = df["Date"].iloc[0]
end_date = df["Date"].iloc[-1]
total_days = int((end_date - start_date).days)

nav_gross_end = float(df["NAV_gross"].iloc[-1])
nav_net_end = float(df["NAV_net"].iloc[-1])
fees_total = float(df["Fees_kum_total"].iloc[-1])
fees_mf = float(df["MF_kum"].iloc[-1])
fees_pf = float(df["PF_kum"].iloc[-1])

gross_tr = nav_gross_end / params.start_nav - 1.0
net_tr = nav_net_end / params.start_nav - 1.0
fee_drag_tr = gross_tr - net_tr

gross_cagr = cagr(params.start_nav, nav_gross_end, total_days)
net_cagr = cagr(params.start_nav, nav_net_end, total_days)
net_vol = ann_vol_log(df["NAV_net"])

avg_total_fee_month = float((m["MF"] + m["PF"]).mean()) if len(m) else np.nan

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Start NAV", fmt_money(params.start_nav))
c2.metric("End NAV (Net)", fmt_money(nav_net_end), fmt_pct(net_tr))
c3.metric("CAGR (Net)", fmt_pct(net_cagr) if np.isfinite(net_cagr) else "n/a")
c4.metric("Fees Total", fmt_money(fees_total), f"{(fees_total/params.start_nav)*100:,.2f}% of Start")
c5.metric("Mgmt / Perf Fees", f"{fmt_money(fees_mf)} / {fmt_money(fees_pf)}")
c6.metric("Fee Drag (TR)", fmt_pct(fee_drag_tr))


# ──────────────────────────────────────────────────────────────────────────────
# Economics / Info block (aligns with your narrative)
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
st.subheader("Fixum & Payroll-Info (Spanien/Balearen) – nur Information (nicht NAV-relevant)")

k1, k2, k3, k4, k5 = st.columns(5)
k1.metric("Fixum = Mgmt Fee Floor (€/Monat)", f"{fmt_money(pm_gross_month)} €")
k2.metric("Netto-Schätzung PM (€/Monat)", f"{fmt_money(employee_net_month)} €")
k3.metric("FO Gesamtkosten (€/Monat)", f"{fmt_money(employer_cost_month)} €")
k4.metric("Avg Mgmt Fee realisiert (€/Monat)", f"{fmt_money(avg_mf_month)} €" if np.isfinite(avg_mf_month) else "n/a")
k5.metric("Floor bindet (Monate)", f"{binds_floor_share*100:,.0f}%" if np.isfinite(binds_floor_share) else "n/a")

st.caption(
    "Interpretation: Im ersten Jahr wird das Fixum als Management-Fee-Minimum (Floor) modelliert. "
    "Die Payroll-Zahlen dienen nur als Info für FO/Investor und beeinflussen die NAV-Berechnung nicht."
)


# ──────────────────────────────────────────────────────────────────────────────
# Charts – NAV & HWM + Cum Fees
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
left, right = st.columns([1.35, 1.0])

with left:
    fig_nav = go.Figure()
    if show_gross:
        fig_nav.add_trace(go.Scatter(x=df["Date"], y=df["NAV_gross"], mode="lines", name="NAV Gross", line=dict(width=2)))
    fig_nav.add_trace(go.Scatter(x=df["Date"], y=df["NAV_net"], mode="lines", name="NAV Net", line=dict(width=3)))
    fig_nav.add_trace(go.Scatter(
        x=df["Date"], y=df["HWM_neu"], mode="lines", name="HWM (Net)",
        line=dict(width=3, dash="dot", color="red")
    ))
    fig_nav.update_layout(
        title="NAV Entwicklung & High Water Mark",
        xaxis_title="Datum",
        yaxis_title="NAV",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=460,
    )
    st.plotly_chart(fig_nav, use_container_width=True)

with right:
    fig_fees = go.Figure()
    fig_fees.add_trace(go.Scatter(x=df["Date"], y=df["MF_kum"], mode="lines", name="Mgmt Fees (cum)", line=dict(width=2)))
    fig_fees.add_trace(go.Scatter(x=df["Date"], y=df["PF_kum"], mode="lines", name="Perf Fees (cum)", line=dict(width=2)))
    fig_fees.add_trace(go.Scatter(x=df["Date"], y=df["Fees_kum_total"], mode="lines", name="Total Fees (cum)", line=dict(width=3)))
    fig_fees.update_layout(
        title="Kumulierte Fees",
        xaxis_title="Datum",
        yaxis_title="Fees",
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
        margin=dict(l=10, r=10, t=60, b=10),
        height=460,
    )
    st.plotly_chart(fig_fees, use_container_width=True)

# Monthly charts
row2a, row2b = st.columns([1.0, 1.0])

with row2a:
    m_plot = m.copy()
    m_plot["MonthStr"] = m_plot["Month"]

    fig_mfees = go.Figure()
    fig_mfees.add_trace(go.Bar(x=m_plot["MonthStr"], y=m_plot["MF_Base"], name="Mgmt Fee (Base)"))
    fig_mfees.add_trace(go.Bar(x=m_plot["MonthStr"], y=m_plot["MF_MinAdj"], name="Mgmt Fee (Min Adj / Floor True-Up)"))
    fig_mfees.add_trace(go.Bar(x=m_plot["MonthStr"], y=m_plot["PF"], name="Perf Fee"))
    fig_mfees.update_layout(
        barmode="stack",
        title="Fees pro Monat (Stacked) – inkl. Floor True-Up",
        xaxis_title="Monat",
        yaxis_title="Fee Amount",
        margin=dict(l=10, r=10, t=60, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_mfees, use_container_width=True)




    if show_fee_drag:
        with cfd:
            df_drag = df[["Date", "NAV_gross_idx", "NAV_net_idx"]].copy()
            df_drag["FeeDrag_idx"] = df_drag["NAV_gross_idx"] - df_drag["NAV_net_idx"]
            fig_drag = go.Figure()
            fig_drag.add_trace(go.Scatter(x=df_drag["Date"], y=df_drag["FeeDrag_idx"], mode="lines", name="Fee Drag (Index)", line=dict(width=3)))
            fig_drag.update_layout(
                title="Fee Drag über Zeit (Gross-Index minus Net-Index)",
                xaxis_title="Datum",
                yaxis_title="Index Punkte",
                margin=dict(l=10, r=10, t=60, b=10),
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_drag, use_container_width=True)

# Tables + Exports (Arrow-safe)
st.divider()
tab1, tab2 = st.tabs(["Detail-Tabelle", "Downloads"])

with tab1:
    view_cols = [
        "Date", "Close", "Tage", "Brutto_Rendite",
        "NAV_gross", "MF_Base", "MF_MinAdj", "MF_Amount", "NAV_nach_MF",
        "HWM_alt", "PF_Basis", "PF_Amount",
        "NAV_net", "HWM_neu",
        "MF_kum", "PF_kum", "Fees_kum_total",
    ]
    view_cols = [c for c in view_cols if c in df.columns]

    display_df = df[view_cols].copy()
    display_df = display_df.loc[:, ~display_df.columns.duplicated()].copy()

    round_map = {
        "Close": 2,
        "Brutto_Rendite": 6,
        "NAV_gross": 2,
        "MF_Base": 2,
        "MF_MinAdj": 2,
        "MF_Amount": 2,
        "NAV_nach_MF": 2,
        "HWM_alt": 2,
        "PF_Basis": 2,
        "PF_Amount": 2,
        "NAV_net": 2,
        "HWM_neu": 2,
        "MF_kum": 2,
        "PF_kum": 2,
        "Fees_kum_total": 2,
    }
    for col, decimals in round_map.items():
        if col in display_df.columns:
            display_df[col] = display_df[col].round(decimals)

    # Arrow-safe: render as HTML (no Arrow serialization)
    display_df["Date"] = display_df["Date"].dt.strftime("%Y-%m-%d")
    for col in display_df.columns:
        display_df[col] = display_df[col].astype(str)

    html = display_df.to_html(index=False)
    st.markdown(
        f"""
        <div style="max-height:520px; overflow:auto; border:1px solid rgba(0,0,0,0.1);
                    border-radius:8px; padding:6px;">
          {html}
        </div>
        """,
        unsafe_allow_html=True
    )

with tab2:
    out = df.copy()
    out.insert(0, "Mgmt_Fee_p_a", params.mgmt_fee_pa)
    out.insert(1, "Perf_Fee", params.perf_fee)
    out.insert(2, "Mgmt_Fee_Floor_Monthly", params.min_mgmt_fee_monthly)
    out.insert(3, "Start_NAV", params.start_nav)
    out.insert(4, "Daycount", params.daycount)
    out.insert(5, "Perf_Mode", params.perf_crystallization)
    out.insert(6, "Resample_BDays", params.resample_bdays)

    # Spain payroll assumptions (INFO only) — exported for transparency
    out.insert(7, "PM_Fixum_Floor_Month", float(pm_gross_month))
    out.insert(8, "AG_Social_Rate", float(employer_social_rate))
    out.insert(9, "AN_Social_Rate", float(employee_social_rate))
    out.insert(10, "IRPF_Rate", float(employee_irpf_rate))

    out["Date"] = out["Date"].dt.strftime("%Y-%m-%d")

    m_export = m.copy()
    if "Month_date" in m_export.columns:
        m_export["Month_date"] = m_export["Month_date"].dt.strftime("%Y-%m-%d")

    st.download_button(
        "⬇️ Download Detail (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="feeengine_detail.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download Monthly Summary (CSV)",
        data=m_export.to_csv(index=False).encode("utf-8"),
        file_name="feeengine_monthly.csv",
        mime="text/csv",
    )

# Footer
st.caption(
    f"Modus: {params.perf_crystallization.upper()} | "
    f"Zeitraum: {start_date.date()} – {end_date.date()} | "
    f"Gross CAGR: {fmt_pct(gross_cagr) if np.isfinite(gross_cagr) else 'n/a'} | "
    f"Net CAGR: {fmt_pct(net_cagr) if np.isfinite(net_cagr) else 'n/a'} | "
    f"Net Vol (ann., log): {fmt_pct(net_vol) if np.isfinite(net_vol) else 'n/a'}"
)

