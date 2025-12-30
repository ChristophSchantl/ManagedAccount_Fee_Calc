# MA_Fee_Calc.py
# FeeEngine – Managed Account Fee Calculator (Streamlit Cloud / Python 3.13 SAFE)
# IMPORTANT: NO st.dataframe / NO st.table  -> avoids pyarrow JSON metadata crashes
# Run: streamlit run MA_Fee_Calc.py

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Dict, List

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

      table { border-collapse: collapse; width: 100%; font-size: 12px; }
      th, td { border: 1px solid rgba(0,0,0,0.10); padding: 6px 8px; text-align: right; }
      th { position: sticky; top: 0; background: rgba(250,250,250,0.98); z-index: 2; text-align: right; }
      th:first-child, td:first-child { text-align: left; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Data classes
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeeParams:
    mgmt_fee_pa: float          # 0.02
    perf_fee: float             # 0.20
    start_nav: float            # 1_000_000
    daycount: int               # 360/365
    perf_crystallization: str   # "monthly" or "daily"
    date_col: str
    price_col: str
    resample_bdays: bool


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
def pct_to_float(x: float) -> float:
    return x / 100.0 if x > 1.0 else x


def to_number_series(s: pd.Series) -> pd.Series:
    """Robust numeric parser for EU/US formats."""
    if pd.api.types.is_numeric_dtype(s):
        return s.astype(float)

    s2 = s.astype(str).str.strip()
    s2 = s2.str.replace("\u00A0", "", regex=False).str.replace(" ", "", regex=False)

    last_comma = s2.str.rfind(",")
    last_dot = s2.str.rfind(".")
    eu_mask = (last_comma > last_dot) & s2.str.contains(",")

    s_eu = s2.str.replace(".", "", regex=False).str.replace(",", ".", regex=False)
    s_us = s2.str.replace(",", "", regex=False)

    s3 = pd.Series(np.where(eu_mask, s_eu, s_us), index=s.index)
    return pd.to_numeric(s3, errors="coerce")


def parse_prices(upload: Optional[io.BytesIO], date_col: str, price_col: str) -> pd.DataFrame:
    """Input: CSV/XLSX with date_col + price_col."""
    if upload is None:
        # demo series
        idx = pd.bdate_range("2023-01-02", periods=520)
        rng = np.random.default_rng(7)
        rets = rng.normal(loc=0.00025, scale=0.012, size=len(idx))
        prices = 100.0 * np.exp(np.cumsum(rets))
        return pd.DataFrame({date_col: idx, price_col: prices})

    name = getattr(upload, "name", "").lower()
    raw = upload.read()

    if name.endswith((".xlsx", ".xls")):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Spalten nicht gefunden. Erwartet: '{date_col}' und '{price_col}'.")

    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = to_number_series(df[price_col])

    df = df.dropna().sort_values(date_col).drop_duplicates(subset=[date_col])
    if df.empty:
        raise ValueError("Nach Parsing ist keine gültige Zeitreihe übrig geblieben.")
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


def fmt_money0(x: float) -> str:
    return f"{x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x * 100:,.2f}%"


def render_html_table(df: pd.DataFrame, height_px: int = 520, title: Optional[str] = None) -> None:
    """Arrow-free HTML table renderer (Streamlit Cloud safe)."""
    if title:
        st.subheader(title)

    # Make sure nothing exotic is inside: convert to strings for display only
    safe = df.copy()
    for c in safe.columns:
        if pd.api.types.is_datetime64_any_dtype(safe[c]):
            safe[c] = pd.to_datetime(safe[c], errors="coerce").dt.strftime("%Y-%m-%d")
    safe = safe.astype(str)

    st.markdown(
        f"""
        <div style="max-height:{height_px}px; overflow:auto; border:1px solid rgba(0,0,0,0.12);
                    border-radius:12px; padding:8px; background: white;">
          {safe.to_html(index=False, escape=False)}
        </div>
        """,
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Fee Engine
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

    # Mgmt fee accrual (calendar daycount)
    df["MF_Amount"] = df["NAV_gross"] * (p.mgmt_fee_pa * df["Tage"] / p.daycount)
    df.loc[df["Tage"] == 0, "MF_Amount"] = 0.0
    df["NAV_nach_MF"] = df["NAV_gross"] - df["MF_Amount"]

    # Month marker (keep as datetime, not Period, to avoid exotic dtypes)
    df["Month"] = pd.to_datetime(df["Date"].dt.to_period("M").dt.to_timestamp())

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
        months = df["Month"].values
        for i in range(len(df)):
            df.at[i, "HWM_alt"] = hwm
            nav_after_mf = float(df.at[i, "NAV_nach_MF"])
            is_month_end = (i == len(df) - 1) or (months[i] != months[i + 1])

            pf_basis = 0.0
            pf_amt = 0.0
            nav_net = nav_after_mf
            hwm_new = hwm

            if is_month_end:
                pf_basis = max(0.0, nav_after_mf - hwm)
                pf_amt = pf_basis * p.perf_fee
                nav_net = nav_after_mf - pf_amt
                hwm_new = nav_net if pf_basis > 0 else hwm
                hwm = hwm_new

            df.at[i, "PF_Basis"] = pf_basis
            df.at[i, "PF_Amount"] = pf_amt
            df.at[i, "NAV_net"] = nav_net
            df.at[i, "HWM_neu"] = hwm_new

    df["MF_kum"] = df["MF_Amount"].cumsum()
    df["PF_kum"] = df["PF_Amount"].cumsum()
    df["Fees_kum_total"] = df["MF_kum"] + df["PF_kum"]

    df["NAV_gross_idx"] = df["NAV_gross"] / p.start_nav
    df["NAV_net_idx"] = df["NAV_net"] / p.start_nav
    df["DD_gross"] = df["NAV_gross"] / df["NAV_gross"].cummax() - 1.0
    df["DD_net"] = df["NAV_net"] / df["NAV_net"].cummax() - 1.0

    monthly = df.groupby("Month", as_index=False).agg(
        NAV_gross=("NAV_gross", "last"),
        NAV_net=("NAV_net", "last"),
        MF=("MF_Amount", "sum"),
        PF=("PF_Amount", "sum"),
        Fees=("Fees_kum_total", "last"),
    )
    monthly["FeeRate_MF_bps"] = (monthly["MF"] / monthly["NAV_gross"]) * 1e4
    monthly["FeeRate_total_bps"] = ((monthly["MF"] + monthly["PF"]) / monthly["NAV_gross"]) * 1e4

    df.attrs["monthly"] = monthly
    return df


# ──────────────────────────────────────────────────────────────────────────────
# UI
# ──────────────────────────────────────────────────────────────────────────────
st.title("FeeEngine – Managed Account Fee Calculator")
st.caption("NAV, Fees, HWM & Fee Drag – CSV/XLSX Input, KPI-Tiles, Plotly-Charts, Exports. (Arrow/pyarrow-proof)")

with st.sidebar:
    st.header("Inputs")
    upload = st.file_uploader("Preisserie (CSV/XLSX) hochladen", type=["csv", "xlsx", "xls"])

    st.subheader("Spaltenmapping")
    date_col = st.text_input("Datum-Spalte", value="Date")
    price_col = st.text_input("Preis-Spalte", value="Close")

    st.subheader("Fee-Parameter")
    mgmt_fee_ui = st.number_input("Mgmt_Fee_p_a (%)", min_value=0.0, max_value=20.0, value=2.0, step=0.25)
    perf_fee_ui = st.number_input("Perf_Fee (%)", min_value=0.0, max_value=50.0, value=20.0, step=1.0)
    start_nav = st.number_input("Start_NAV", min_value=1_000.0, max_value=1_000_000_000.0, value=1_000_000.0, step=10_000.0)

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
    st.subheader("Charts")
    show_gross = st.checkbox("Gross NAV anzeigen", value=True)
    show_drawdown = st.checkbox("Drawdown Chart", value=True)
    show_fee_drag = st.checkbox("Fee Drag Chart", value=True)


# Compute
try:
    df_prices = parse_prices(upload, date_col, price_col)
    params = FeeParams(
        mgmt_fee_pa=pct_to_float(float(mgmt_fee_ui)),
        perf_fee=pct_to_float(float(perf_fee_ui)),
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


# KPIs
start_date = pd.to_datetime(df["Date"].iloc[0])
end_date = pd.to_datetime(df["Date"].iloc[-1])
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
gross_vol = ann_vol_log(df["NAV_gross"])

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Start NAV", fmt_money0(params.start_nav))
c2.metric("End NAV (Net)", fmt_money0(nav_net_end), fmt_pct(net_tr))
c3.metric("CAGR (Net)", fmt_pct(net_cagr) if np.isfinite(net_cagr) else "n/a")
c4.metric("Fees Total", fmt_money0(fees_total), f"{(fees_total/params.start_nav)*100:,.2f}% of Start")
c5.metric("Mgmt / Perf Fees", f"{fmt_money0(fees_mf)} / {fmt_money0(fees_pf)}")
c6.metric("Fee Drag (TR)", fmt_pct(fee_drag_tr))

st.divider()


# Charts – NAV & HWM + Cum Fees
left, right = st.columns([1.35, 1.0])

with left:
    fig_nav = go.Figure()
    if show_gross:
        fig_nav.add_trace(go.Scatter(x=df["Date"], y=df["NAV_gross"], mode="lines", name="NAV Gross", line=dict(width=2)))
    fig_nav.add_trace(go.Scatter(x=df["Date"], y=df["NAV_net"], mode="lines", name="NAV Net", line=dict(width=3)))
    fig_nav.add_trace(go.Scatter(x=df["Date"], y=df["HWM_neu"], mode="lines", name="HWM (Net)", line=dict(width=1, dash="dot")))
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
    m_plot["MonthStr"] = pd.to_datetime(m_plot["Month"]).dt.strftime("%Y-%m")
    fig_mfees = go.Figure()
    fig_mfees.add_trace(go.Bar(x=m_plot["MonthStr"], y=m_plot["MF"], name="Mgmt Fee"))
    fig_mfees.add_trace(go.Bar(x=m_plot["MonthStr"], y=m_plot["PF"], name="Perf Fee"))
    fig_mfees.update_layout(
        barmode="stack",
        title="Fees pro Monat (Stacked)",
        xaxis_title="Monat",
        yaxis_title="Fee Amount",
        margin=dict(l=10, r=10, t=60, b=40),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_mfees, use_container_width=True)

with row2b:
    fig_bps = go.Figure()
    fig_bps.add_trace(go.Scatter(x=m_plot["Month"], y=m_plot["FeeRate_MF_bps"], mode="lines+markers", name="Mgmt Fee (bps)"))
    fig_bps.add_trace(go.Scatter(x=m_plot["Month"], y=m_plot["FeeRate_total_bps"], mode="lines+markers", name="Total Fee (bps)"))
    fig_bps.update_layout(
        title="Fee Rate (bps) – Monatsbasis",
        xaxis_title="Monat",
        yaxis_title="bps",
        margin=dict(l=10, r=10, t=60, b=10),
        height=420,
        legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
    )
    st.plotly_chart(fig_bps, use_container_width=True)


if show_drawdown or show_fee_drag:
    st.divider()
    cdd, cfd = st.columns([1.0, 1.0])

    if show_drawdown:
        with cdd:
            fig_dd = go.Figure()
            if show_gross:
                fig_dd.add_trace(go.Scatter(x=df["Date"], y=df["DD_gross"], mode="lines", name="Drawdown Gross", line=dict(width=2)))
            fig_dd.add_trace(go.Scatter(x=df["Date"], y=df["DD_net"], mode="lines", name="Drawdown Net", line=dict(width=3)))
            fig_dd.update_layout(
                title="Drawdown (Gross vs Net)",
                xaxis_title="Datum",
                yaxis_title="Drawdown",
                yaxis_tickformat=".1%",
                margin=dict(l=10, r=10, t=60, b=10),
                height=420,
                legend=dict(orientation="h", yanchor="bottom", y=1.02, xanchor="left", x=0),
            )
            st.plotly_chart(fig_dd, use_container_width=True)

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


# Tables + Exports (NO Arrow)
st.divider()
tab1, tab2 = st.tabs(["Detail-Tabelle", "Downloads"])

with tab1:
    view_cols: List[str] = [
        "Date", "Close", "Tage", "Brutto_Rendite",
        "NAV_gross", "MF_Amount", "NAV_nach_MF",
        "HWM_alt", "PF_Basis", "PF_Amount",
        "NAV_net", "HWM_neu",
        "MF_kum", "PF_kum", "Fees_kum_total",
    ]
    view_cols = [c for c in view_cols if c in df.columns]
    display_df = df[view_cols].copy()
    display_df = display_df.loc[:, ~display_df.columns.duplicated()].copy()

    # Numeric rounding for display
    round_map: Dict[str, int] = {
        "Close": 2, "Brutto_Rendite": 6,
        "NAV_gross": 2, "MF_Amount": 2, "NAV_nach_MF": 2,
        "HWM_alt": 2, "PF_Basis": 2, "PF_Amount": 2,
        "NAV_net": 2, "HWM_neu": 2,
        "MF_kum": 2, "PF_kum": 2, "Fees_kum_total": 2,
    }
    for col, nd in round_map.items():
        if col in display_df.columns:
            display_df[col] = pd.to_numeric(display_df[col], errors="coerce").round(nd)

    render_html_table(display_df, height_px=520)

with tab2:
    out = df.copy()
    out.insert(0, "Mgmt_Fee_p_a", params.mgmt_fee_pa)
    out.insert(1, "Perf_Fee", params.perf_fee)
    out.insert(2, "Start_NAV", params.start_nav)
    out.insert(3, "Daycount", params.daycount)
    out.insert(4, "Perf_Mode", params.perf_crystallization)
    out.insert(5, "Resample_BDays", params.resample_bdays)

    st.download_button(
        "⬇️ Download Detail (CSV)",
        data=out.to_csv(index=False).encode("utf-8"),
        file_name="feeengine_detail.csv",
        mime="text/csv",
    )

    st.download_button(
        "⬇️ Download Monthly Summary (CSV)",
        data=m.to_csv(index=False).encode("utf-8"),
        file_name="feeengine_monthly.csv",
        mime="text/csv",
    )


st.caption(
    f"Modus: {params.perf_crystallization.upper()} | "
    f"Zeitraum: {start_date.date()} – {end_date.date()} | "
    f"Gross CAGR: {fmt_pct(gross_cagr) if np.isfinite(gross_cagr) else 'n/a'} | "
    f"Net CAGR: {fmt_pct(net_cagr) if np.isfinite(net_cagr) else 'n/a'} | "
    f"Net Vol (ann., log): {fmt_pct(net_vol) if np.isfinite(net_vol) else 'n/a'}"
)
