# streamlit_app.py
# FeeEngine – Managed Account Fee Calculator (Investor-ready)
# Dependencies: streamlit, pandas, numpy, plotly

from __future__ import annotations

import io
from dataclasses import dataclass
from typing import Optional, Tuple

import numpy as np
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import streamlit as st


# ──────────────────────────────────────────────────────────────────────────────
# Config
# ──────────────────────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="FeeEngine – Managed Account Fees",
    page_icon="📈",
    layout="wide",
)

# Small UI polish (clean, presentation-ready)
st.markdown(
    """
    <style>
      .block-container { padding-top: 1.0rem; padding-bottom: 2.0rem; }
      div[data-testid="stMetricValue"] { font-size: 1.45rem; }
      div[data-testid="stMetricLabel"] { font-size: 0.9rem; opacity: 0.85; }
      .stPlotlyChart { background: transparent; }
      .caption { opacity: 0.8; font-size: 0.9rem; }
    </style>
    """,
    unsafe_allow_html=True,
)


# ──────────────────────────────────────────────────────────────────────────────
# Helpers
# ──────────────────────────────────────────────────────────────────────────────
@dataclass(frozen=True)
class FeeParams:
    mgmt_fee_pa: float          # e.g. 0.02
    perf_fee: float             # e.g. 0.20
    start_nav: float            # e.g. 1_000_000
    daycount: int               # e.g. 365
    perf_crystallization: str   # "daily" or "monthly"
    date_col: str
    price_col: str


def _to_float_pct(x: float) -> float:
    # UI convenience: accept 2 => 0.02
    return x / 100.0 if x > 1.0 else x


def parse_prices(upload: Optional[io.BytesIO], date_col: str, price_col: str) -> pd.DataFrame:
    """
    Expects at least two columns:
      - date_col: parseable datetime
      - price_col: numeric
    """
    if upload is None:
        # Fallback: synthetic demo series (business days)
        idx = pd.bdate_range("2023-01-02", periods=520)
        rng = np.random.default_rng(7)
        rets = rng.normal(loc=0.00025, scale=0.012, size=len(idx))
        px0 = 100.0
        prices = px0 * np.exp(np.cumsum(rets))
        df = pd.DataFrame({date_col: idx, price_col: prices})
        return df

    name = getattr(upload, "name", "").lower()
    raw = upload.read()

    if name.endswith(".xlsx") or name.endswith(".xls"):
        df = pd.read_excel(io.BytesIO(raw))
    else:
        # CSV default, auto-sep
        df = pd.read_csv(io.BytesIO(raw), sep=None, engine="python")

    if date_col not in df.columns or price_col not in df.columns:
        raise ValueError(f"Spalten nicht gefunden. Erwartet: '{date_col}' und '{price_col}'.")

    df = df[[date_col, price_col]].copy()
    df[date_col] = pd.to_datetime(df[date_col], errors="coerce")
    df[price_col] = pd.to_numeric(df[price_col], errors="coerce")
    df = df.dropna().sort_values(date_col).drop_duplicates(subset=[date_col])
    return df


def compute_fee_engine(df_prices: pd.DataFrame, p: FeeParams) -> pd.DataFrame:
    df = df_prices.copy()
    df = df.sort_values(p.date_col).reset_index(drop=True)
    df.rename(columns={p.date_col: "Date", p.price_col: "Close"}, inplace=True)

    # Day gaps (calendar days) like the FeeEngine table "Tage"
    df["Tage"] = df["Date"].diff().dt.days.fillna(0).astype(int)

    # Gross NAV based on price index
    base_price = float(df.loc[0, "Close"])
    df["Brutto_Rendite"] = df["Close"] / base_price
    df["NAV_gross"] = p.start_nav * df["Brutto_Rendite"]

    # Mgmt fee: linear accrual per day on NAV_gross
    # MF_Amount = NAV_gross * mgmt_fee_pa * days / daycount
    df["MF_Faktor"] = 1.0 - (p.mgmt_fee_pa * df["Tage"] / p.daycount)
    df.loc[df["Tage"] == 0, "MF_Faktor"] = 1.0
    df["MF_Amount"] = df["NAV_gross"] * (p.mgmt_fee_pa * df["Tage"] / p.daycount)
    df["NAV_nach_MF"] = df["NAV_gross"] - df["MF_Amount"]

    # Performance fee on High Water Mark (after Mgmt fee)
    df["Month"] = df["Date"].dt.to_period("M").dt.to_timestamp()
    df["HWM_alt"] = np.nan
    df["PF_Basis"] = 0.0
    df["PF_Amount"] = 0.0
    df["NAV_net"] = np.nan
    df["HWM_neu"] = np.nan

    hwm = p.start_nav

    if p.perf_crystallization == "daily":
        # Daily crystallization (matches the sheet behavior in the excerpt)
        for i in range(len(df)):
            df.at[i, "HWM_alt"] = hwm
            nav_after_mf = float(df.at[i, "NAV_nach_MF"])
            pf_basis = max(0.0, nav_after_mf - hwm)
            pf_amt = pf_basis * p.perf_fee
            nav_net = nav_after_mf - pf_amt

            # Update HWM to NAV_net when perf fee paid; else keep
            hwm_new = nav_net if pf_basis > 0 else hwm

            df.at[i, "PF_Basis"] = pf_basis
            df.at[i, "PF_Amount"] = pf_amt
            df.at[i, "NAV_net"] = nav_net
            df.at[i, "HWM_neu"] = hwm_new
            hwm = hwm_new

    else:
        # Monthly crystallization (typical managed account / wikifolio-style month-end)
        # Accrue PF only at month-end using month-end NAV_nach_MF vs HWM.
        # HWM updates only at crystallization points.
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

    # Cum fees
    df["MF_kum"] = df["MF_Amount"].cumsum()
    df["PF_kum"] = df["PF_Amount"].cumsum()
    df["Fees_kum_total"] = df["MF_kum"] + df["PF_kum"]

    # Fee per day (for “fee drag” visuals)
    df["T_Fee_pDay"] = np.where(df["Tage"] > 0, (df["MF_Amount"] + df["PF_Amount"]) / df["Tage"], 0.0)

    # Returns & drawdowns for charting
    df["NAV_gross_idx"] = df["NAV_gross"] / p.start_nav
    df["NAV_net_idx"] = df["NAV_net"] / p.start_nav
    df["DD_gross"] = df["NAV_gross"] / df["NAV_gross"].cummax() - 1.0
    df["DD_net"] = df["NAV_net"] / df["NAV_net"].cummax() - 1.0

    # Monthly aggregation (presentation-friendly)
    m = df.groupby("Month", as_index=False).agg(
        NAV_gross=("NAV_gross", "last"),
        NAV_net=("NAV_net", "last"),
        MF=("MF_Amount", "sum"),
        PF=("PF_Amount", "sum"),
        Fees=("Fees_kum_total", "last"),
    )
    m["FeeRate_MF_bps"] = (m["MF"] / m["NAV_gross"]) * 1e4
    m["FeeRate_total_bps"] = ((m["MF"] + m["PF"]) / m["NAV_gross"]) * 1e4
    df.attrs["monthly"] = m

    return df


def cagr(start: float, end: float, days: int) -> float:
    if days <= 0 or start <= 0:
        return np.nan
    years = days / 365.0
    return (end / start) ** (1 / years) - 1


def ann_vol(daily_returns: pd.Series) -> float:
    r = daily_returns.dropna()
    if len(r) < 2:
        return np.nan
    return float(r.std(ddof=1) * np.sqrt(252))


def fmt_money(x: float) -> str:
    return f"{x:,.0f}"


def fmt_pct(x: float) -> str:
    return f"{x*100:,.2f}%"


# ──────────────────────────────────────────────────────────────────────────────
# Sidebar – Inputs
# ──────────────────────────────────────────────────────────────────────────────
st.title("FeeEngine – Managed Account Fee Calculator")
st.caption("Investor-ready NAV, Fees, HWM & Fee Drag – mit CSV/XLSX Input und Exporten.")

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
        options=["daily", "monthly"],
        index=0,
        help="daily ≈ kontinuierlich (wie im Sheet-Excerpt); monthly = nur Monatsultimo (typisch MA / wikifolio-like).",
    )

    st.divider()
    st.subheader("Darstellung")
    show_gross = st.checkbox("Gross NAV anzeigen", value=True)
    show_drawdown = st.checkbox("Drawdown Chart", value=True)
    show_fee_drag = st.checkbox("Fee Drag Chart", value=True)


# ──────────────────────────────────────────────────────────────────────────────
# Compute
# ──────────────────────────────────────────────────────────────────────────────
try:
    df_prices = parse_prices(upload, date_col, price_col)
    p = FeeParams(
        mgmt_fee_pa=_to_float_pct(mgmt_fee_ui),
        perf_fee=_to_float_pct(perf_fee_ui),
        start_nav=float(start_nav),
        daycount=int(daycount),
        perf_crystallization=str(perf_mode),
        date_col=str(date_col),
        price_col=str(price_col),
    )
    df = compute_fee_engine(df_prices, p)
    m = df.attrs["monthly"]
except Exception as e:
    st.error(f"Input/Parsing Fehler: {e}")
    st.stop()


# ──────────────────────────────────────────────────────────────────────────────
# KPIs (top row)
# ──────────────────────────────────────────────────────────────────────────────
start_date = df["Date"].iloc[0]
end_date = df["Date"].iloc[-1]
total_days = int((end_date - start_date).days)

nav_gross_end = float(df["NAV_gross"].iloc[-1])
nav_net_end = float(df["NAV_net"].iloc[-1])
fees_total = float(df["Fees_kum_total"].iloc[-1])
fees_mf = float(df["MF_kum"].iloc[-1])
fees_pf = float(df["PF_kum"].iloc[-1])

gross_tr = nav_gross_end / p.start_nav - 1.0
net_tr = nav_net_end / p.start_nav - 1.0
fee_drag = gross_tr - net_tr

gross_cagr = cagr(p.start_nav, nav_gross_end, total_days)
net_cagr = cagr(p.start_nav, nav_net_end, total_days)

# “Daily returns” based on index (calendar gaps exist; keep simple for presentation)
gross_ret = df["NAV_gross"].pct_change()
net_ret = df["NAV_net"].pct_change()
gross_vol = ann_vol(gross_ret)
net_vol = ann_vol(net_ret)

c1, c2, c3, c4, c5, c6 = st.columns(6)
c1.metric("Start NAV", fmt_money(p.start_nav))
c2.metric("End NAV (Net)", fmt_money(nav_net_end), fmt_pct(net_tr))
c3.metric("CAGR (Net)", fmt_pct(net_cagr) if np.isfinite(net_cagr) else "n/a")
c4.metric("Fees Total", fmt_money(fees_total), f"{(fees_total/p.start_nav)*100:,.2f}% of Start")
c5.metric("Mgmt / Perf Fees", f"{fmt_money(fees_mf)} / {fmt_money(fees_pf)}")
c6.metric("Fee Drag (TR)", fmt_pct(fee_drag))

st.divider()


# ──────────────────────────────────────────────────────────────────────────────
# Charts – NAV & HWM
# ──────────────────────────────────────────────────────────────────────────────
left, right = st.columns([1.35, 1.0])

with left:
    fig_nav = go.Figure()

    if show_gross:
        fig_nav.add_trace(
            go.Scatter(
                x=df["Date"], y=df["NAV_gross"],
                mode="lines",
                name="NAV Gross",
                line=dict(width=2),
            )
        )

    fig_nav.add_trace(
        go.Scatter(
            x=df["Date"], y=df["NAV_net"],
            mode="lines",
            name="NAV Net",
            line=dict(width=3),
        )
    )

    fig_nav.add_trace(
        go.Scatter(
            x=df["Date"], y=df["HWM_neu"],
            mode="lines",
            name="HWM (Net)",
            line=dict(width=1, dash="dot"),
        )
    )

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
    # Fees composition chart (cumulative)
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


# ──────────────────────────────────────────────────────────────────────────────
# Charts – Monthly fees & (optional) Drawdown / Fee Drag
# ──────────────────────────────────────────────────────────────────────────────
row2a, row2b = st.columns([1.0, 1.0])

with row2a:
    m_plot = m.copy()
    m_plot["MonthStr"] = m_plot["Month"].dt.strftime("%Y-%m")
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
    # Fee rate (bps) presentation-friendly
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
            # Fee drag time series (gross idx - net idx)
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


# ──────────────────────────────────────────────────────────────────────────────
# Table + Exports
# ──────────────────────────────────────────────────────────────────────────────
st.divider()
tab1, tab2 = st.tabs(["Detail-Tabelle", "Downloads"])

with tab1:
    view_cols = [
        "Date", "Close", "Tage", "Brutto_Rendite",
        "NAV_gross", "MF_Amount", "NAV_nach_MF",
        "HWM_alt", "PF_Basis", "PF_Amount",
        "NAV_net", "HWM_neu",
        "MF_kum", "PF_kum", "Fees_kum_total",
        "T_Fee_pDay",
    ]

    display_df = df[view_cols].copy()

    # Professionelle Darstellung ohne Styler (Arrow-safe)
    round_map = {
        "Close": 2,
        "Brutto_Rendite": 6,
        "NAV_gross": 2,
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
        "T_Fee_pDay": 2,
    }
    display_df = display_df.round(round_map)

    st.dataframe(
        display_df,
        use_container_width=True,
        height=520,
    )



with tab2:
    out = df.copy()
    out.insert(0, "Mgmt_Fee_p_a", p.mgmt_fee_pa)
    out.insert(1, "Perf_Fee", p.perf_fee)
    out.insert(2, "Start_NAV", p.start_nav)
    out.insert(3, "Daycount", p.daycount)
    out.insert(4, "Perf_Mode", p.perf_crystallization)

    csv_bytes = out.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Detail (CSV)", data=csv_bytes, file_name="feeengine_detail.csv", mime="text/csv")

    monthly_csv = m.to_csv(index=False).encode("utf-8")
    st.download_button("⬇️ Download Monthly Summary (CSV)", data=monthly_csv, file_name="feeengine_monthly.csv", mime="text/csv")

    st.markdown(
        '<div class="caption">Tipp: Für Präsentationen nutze „monthly“ für saubere Monatsultimo-Crystallization; '
        '„daily“ ist näher an der kontinuierlichen HWM-Update-Logik.</div>',
        unsafe_allow_html=True,
    )


# ──────────────────────────────────────────────────────────────────────────────
# Footer note (kept short)
# ──────────────────────────────────────────────────────────────────────────────
st.caption(
    f"Berechnungsmodus: {p.perf_crystallization.upper()} | "
    f"Zeitraum: {start_date.date()} – {end_date.date()} | "
    f"Gross CAGR: {fmt_pct(gross_cagr) if np.isfinite(gross_cagr) else 'n/a'} | "
    f"Net CAGR: {fmt_pct(net_cagr) if np.isfinite(net_cagr) else 'n/a'} | "
    f"Net Vol (ann.): {fmt_pct(net_vol) if np.isfinite(net_vol) else 'n/a'}"
)
