#######################
# Foad Farahbod #######
# 945246 ##############
#######################

import os
import inspect
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
from datetime import date

#######################
# Setup ###############
#######################

st.set_page_config(
    page_title="Flight Delays Dashboard",
    page_icon="✈️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# -----------------------------
# Palette
# -----------------------------
BG_MAIN = "#0B1020"
BG_SIDEBAR = "#0A0F1D"
CARD_BG = "rgba(255,255,255,0.06)"
CARD_BORDER = "rgba(255,255,255,0.14)"
TEXT_MAIN = "rgba(255,255,255,0.95)"
TEXT_SUB = "rgba(255,255,255,0.74)"
TEXT_MUTED = "rgba(255,255,255,0.62)"
GRID = "rgba(255,255,255,0.12)"
ACCENT = "#FF4B4B"

# -----------------------------
# CSS
# -----------------------------
st.markdown(
    f"""
<style>
.block-container {{
  padding-top: 2.2rem;
  padding-bottom: 2.4rem;
  max-width: 1440px;
}}

.stApp {{
  background:
    radial-gradient(900px 520px at 18% 10%, rgba(255,75,75,0.10), transparent 60%),
    radial-gradient(900px 520px at 88% 0%, rgba(124,92,255,0.08), transparent 62%),
    {BG_MAIN};
}}

section[data-testid="stSidebar"] {{
  background:
    linear-gradient(180deg, rgba(255,255,255,0.03), rgba(255,255,255,0.00)),
    {BG_SIDEBAR};
  border-right: 1px solid rgba(255,255,255,0.12);
}}
section[data-testid="stSidebar"] * {{
  color: {TEXT_MAIN};
}}

section[data-testid="stSidebar"] hr {{
  margin: 0.20rem 0 !important;
  border-color: rgba(255,255,255,0.10) !important;
}}

.header-band {{
  background: linear-gradient(90deg, rgba(255,75,75,0.14), rgba(124,92,255,0.12));
  border: 1px solid rgba(255,255,255,0.12);
  border-bottom: 1px solid rgba(255,255,255,0.18);
  border-radius: 18px;
  padding: 18px 18px 14px 18px;
  margin-bottom: 16px;
}}
.header-title {{
  font-size: 2.30rem;
  font-weight: 900;
  line-height: 1.1;
  color: {TEXT_MAIN};
  margin: 0;
}}
.header-sub {{
  margin-top: 6px;
  color: {TEXT_SUB};
  font-size: 1.0rem;
}}

.card {{
  background: {CARD_BG};
  border: 1px solid {CARD_BORDER};
  border-radius: 16px;
  padding: 14px 14px 12px 14px;
  box-shadow: 0 12px 28px rgba(0,0,0,0.35);
}}

.kpi-label {{ font-size: 0.82rem; color: {TEXT_SUB}; }}
.kpi-value {{
  font-size: 2.05rem;
  font-weight: 950;
  color: {TEXT_MAIN};
  line-height: 1.05;
  margin-top: 2px;
}}
.kpi-note {{ font-size: 0.78rem; color: {TEXT_MUTED}; margin-top: 6px; }}

div[data-baseweb="select"] > div,
div[data-baseweb="input"] > div {{
  background: rgba(255,255,255,0.07) !important;
  border: 1px solid rgba(255,255,255,0.18) !important;
  border-radius: 12px !important;
}}

div[data-testid="stDateInput"] div[data-baseweb="input"] > div {{
  background: rgba(255,255,255,0.09) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
}}
div[data-testid="stDateInput"] input {{
  color: {TEXT_MAIN} !important;
  -webkit-text-fill-color: {TEXT_MAIN} !important;
  caret-color: {TEXT_MAIN} !important;
}}
div[data-testid="stDateInput"] input::placeholder {{
  color: rgba(255,255,255,0.70) !important;
  -webkit-text-fill-color: rgba(255,255,255,0.70) !important;
}}

section[data-testid="stSidebar"] div[data-testid="stDateInput"] {{
  margin-bottom: 0 !important;
  padding-bottom: 0 !important;
}}

section[data-testid="stSidebar"] .stButton,
section[data-testid="stSidebar"] .stButton > button {{
  width: 100% !important;
}}

section[data-testid="stSidebar"] .stButton > button {{
  box-sizing: border-box !important;
  height: 46px !important;
  min-height: 46px !important;
  padding: 0 6px !important;
  border-radius: 14px !important;

  display: flex !important;
  align-items: center !important;
  justify-content: center !important;

  white-space: nowrap !important;
  overflow: hidden !important;
  text-overflow: ellipsis !important;

  background: rgba(255,255,255,0.10) !important;
  border: 1px solid rgba(255,255,255,0.22) !important;
  color: {TEXT_MAIN} !important;
  font-weight: 800 !important;
  font-size: 0.95rem !important;
}}

.stMultiSelect [data-baseweb="tag"] {{
  background-color: rgba(255,75,75,0.18) !important;
  border: 1px solid rgba(255,75,75,0.45) !important;
  color: {TEXT_MAIN} !important;
}}

/* Make the top KPI metrics (st.metric) white */
div[data-testid="stMetric"] label,
div[data-testid="stMetric"] div[data-testid="stMetricValue"],
div[data-testid="stMetric"] div[data-testid="stMetricDelta"] {{
  color: #FFFFFF !important;
}}
div[data-testid="stMetric"] p {{
  color: #FFFFFF !important;
}}

cols_display_box div[data-testid="stMultiSelect"] label,
cols_display_box *[data-testid="stWidgetLabel"] {{
  color: #FFFFFF !important;
}}

footer {{ visibility: hidden; }}
</style>
""",
    unsafe_allow_html=True,
)


# -----------------------------
# Plotly defaults
# -----------------------------
px.defaults.template = "plotly_dark"
PLOT_LAYOUT = dict(
    paper_bgcolor="rgba(0,0,0,0)",
    plot_bgcolor="rgba(0,0,0,0)",
    font=dict(color=TEXT_MAIN),
    title=dict(font=dict(size=16, color=TEXT_MAIN)),
    margin=dict(l=10, r=10, t=55, b=10),
)
GRID_COLOR = GRID

#######################
# Helpers #############
#######################

def fmt_num(x) -> str:
    if x is None:
        return "0"
    try:
        x = float(x)
    except Exception:
        return "0"
    if np.isnan(x):
        return "0"
    ax = abs(x)
    if ax >= 1_000_000_000:
        return f"{x/1_000_000_000:.2f}B"
    if ax >= 1_000_000:
        return f"{x/1_000_000:.2f}M"
    if ax >= 1_000:
        return f"{x/1_000:.1f}K"
    return f"{x:,.0f}"

def first_existing_path(paths):
    for p in paths:
        if p and os.path.exists(p):
            return p
    return None

def safe_dt(s):
    return pd.to_datetime(s, errors="coerce")

def normalize_date_range(date_value, min_d: date, max_d: date):
    if isinstance(date_value, (tuple, list)):
        if len(date_value) == 2:
            a, b = date_value
            if a is None and b is None:
                return min_d, max_d
            if a is None:
                return b, b
            if b is None:
                return a, a
            return (a, b) if a <= b else (b, a)
        if len(date_value) == 1 and date_value[0] is not None:
            return date_value[0], date_value[0]
        return min_d, max_d
    if isinstance(date_value, date):
        return date_value, date_value
    return min_d, max_d

def supports_plotly_selection() -> bool:
    try:
        sig = inspect.signature(st.plotly_chart)
        return "on_select" in sig.parameters and "selection_mode" in sig.parameters
    except Exception:
        return False

PLOTLY_SELECT_OK = supports_plotly_selection()

def plotly_select(fig, key: str):
    cfg = {"displayModeBar": False}
    if PLOTLY_SELECT_OK:
        return st.plotly_chart(
            fig,
            use_container_width=True,
            config=cfg,
            key=key,
            on_select="rerun",
            selection_mode=("points",),
        )
    st.plotly_chart(fig, use_container_width=True, config=cfg, key=key)
    return None

def extract_clicked_value(event, prefer_field: str):
    if not event or not isinstance(event, dict):
        return None
    sel = event.get("selection")
    if not sel or not isinstance(sel, dict):
        return None
    pts = sel.get("points")
    if not pts:
        return None
    p0 = pts[0]
    if prefer_field in p0 and p0.get(prefer_field) is not None:
        return p0.get(prefer_field)
    for k in ["y", "x", "hovertext", "text", "label"]:
        if k in p0 and p0.get(k) is not None:
            return p0.get(k)
    return None

def init_state():
    defaults = {
        "sel_airline": None,
        "sel_model": None,
        "sel_weekday": None,
        "sel_origin": None,
        "sel_destination": None,
        "reset_token": 0,
    }
    for k, v in defaults.items():
        if k not in st.session_state:
            st.session_state[k] = v

def clear_chart_selections():
    st.session_state.sel_airline = None
    st.session_state.sel_model = None
    st.session_state.sel_weekday = None
    st.session_state.sel_origin = None
    st.session_state.sel_destination = None

def apply_axes_style(fig):
    fig.update_layout(**PLOT_LAYOUT)
    fig.update_xaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig.update_yaxes(showgrid=False)
    fig.update_traces(marker_line_width=0)
    return fig

#######################
# Load data ###########
#######################

@st.cache_data(show_spinner=False)
def load_data(xlsx_path: str) -> pd.DataFrame:
    xl = pd.ExcelFile(xlsx_path)
    flights = xl.parse("flights")
    aircrafts = xl.parse("aircrafts")
    airlines = xl.parse("airlines")
    airports = xl.parse("airports")

    for c in ["scheduled_departure", "departure", "scheduled_arrival", "arrival"]:
        if c in flights.columns:
            flights[c] = safe_dt(flights[c])

    df = flights.merge(airlines, on="airline_id", how="left")
    df = df.merge(
        aircrafts[["aircraft_id", "manufacturer", "model"]],
        on="aircraft_id",
        how="left",
    )

    ap = airports.copy()

    ap_origin = ap.rename(
        columns={
            "airport_code": "origin",
            "name": "origin_name",
            "latitude": "origin_latitude",
            "longitude": "origin_longitude",
        }
    )
    df = df.merge(
        ap_origin[["origin", "origin_name", "origin_latitude", "origin_longitude"]],
        on="origin",
        how="left",
    )

    ap_dest = ap.rename(
        columns={
            "airport_code": "destination",
            "name": "dest_name",
            "latitude": "dest_latitude",
            "longitude": "dest_longitude",
        }
    )
    df = df.merge(
        ap_dest[["destination", "dest_name", "dest_latitude", "dest_longitude"]],
        on="destination",
        how="left",
    )

    df["flight_date"] = df["scheduled_departure"].dt.date
    df["weekday"] = df["scheduled_departure"].dt.day_name()
    df["cancelled"] = df["departure"].isna() | df["arrival"].isna()

    df["departure_delay"] = pd.to_numeric(df.get("departure_delay"), errors="coerce")
    df["arrival_delay"] = pd.to_numeric(df.get("arrival_delay"), errors="coerce")
    df["dep_delay_pos"] = df["departure_delay"].fillna(0).clip(lower=0)
    return df

DATA_PATH = "data/flight_data.xlsx"
if not DATA_PATH:
    st.error("Could not find flight_data.xlsx. Put it in data/ or next to app.py.")
    st.stop()

df = load_data(DATA_PATH)
init_state()
rt = st.session_state.reset_token

###########################
# Main User Interaction ###
###########################

st.sidebar.header("Settings")
st.sidebar.markdown("---")

origin_all = sorted([x for x in df["origin"].dropna().unique().tolist() if str(x).strip() != ""])
common = [a for a in ["EWR", "LGA", "JFK", "SWF"] if a in origin_all]

with st.sidebar:
    st.markdown("### Origin Airport")
    cols = st.columns(2)
    selected_quick = []
    if common:
        for i, a in enumerate(common):
            with cols[i % 2]:
                if st.checkbox(a, value=True, key=f"ap_{a}_{rt}"):
                    selected_quick.append(a)

airlines_all = sorted([x for x in df["airline"].dropna().unique().tolist() if str(x).strip() != ""])
airline_sel = st.sidebar.selectbox("Airline Name", ["All"] + airlines_all, index=0, key=f"airline_sb_{rt}")

models_all = sorted([x for x in df["model"].dropna().unique().tolist() if str(x).strip() != ""])
model_sel = st.sidebar.selectbox("Airplane Model", ["All"] + models_all, index=0, key=f"model_sb_{rt}")

min_d = df["flight_date"].min()
max_d = df["flight_date"].max()

date_value = st.sidebar.date_input(
    "Date of Flight",
    value=(min_d, max_d),
    min_value=min_d,
    max_value=max_d,
    key=f"date_sb_{rt}",
)
start_d, end_d = normalize_date_range(date_value, min_d, max_d)

st.sidebar.divider()

btn_cols = st.sidebar.columns(2, gap="medium")
with btn_cols[0]:
    if st.button("Reset filters", use_container_width=True, key=f"btn_reset_filters_{rt}"):
        clear_chart_selections()
        st.rerun()
with btn_cols[1]:
    if st.button("Reset all", use_container_width=True, key=f"btn_reset_all_{rt}"):
        clear_chart_selections()
        st.session_state.reset_token += 1
        st.rerun()

########################
# Main logic  ##########
########################

base_airports = selected_quick if selected_quick else []

if st.session_state.sel_origin and st.session_state.sel_origin in origin_all:
    airport_filter = [st.session_state.sel_origin]
else:
    airport_filter = base_airports

fdf = df[df["origin"].isin(airport_filter)].copy()

if airline_sel != "All":
    fdf = fdf[fdf["airline"] == airline_sel]
if model_sel != "All":
    fdf = fdf[fdf["model"] == model_sel]

fdf = fdf[(fdf["flight_date"] >= start_d) & (fdf["flight_date"] <= end_d)]

if st.session_state.sel_airline:
    fdf = fdf[fdf["airline"] == st.session_state.sel_airline]
if st.session_state.sel_model:
    fdf = fdf[fdf["model"] == st.session_state.sel_model]
if st.session_state.sel_weekday:
    fdf = fdf[fdf["weekday"] == st.session_state.sel_weekday]
if st.session_state.sel_destination:
    fdf = fdf[fdf["destination"] == st.session_state.sel_destination]

########################
# Title  
########################

st.markdown(
    """
<div class="header-band">
  <div class="header-title">Flight Delays and Operations</div>
  <div class="header-sub">Click a bar or bubble to filter, click again to remove</div>
</div>
""",
    unsafe_allow_html=True,
)

########################
# Tabs (Dashboard / Data) below title
########################

tab_dashboard, tab_data = st.tabs(["Dashboard", "Data"])

with tab_dashboard:
    if len(fdf) == 0:
        st.warning("No data matches your filters. Select at least one origin airport.")
        st.stop()

    total_flights = int(len(fdf))
    total_delay_minutes = float(fdf["dep_delay_pos"].sum()) if total_flights else 0.0
    avg_dep_delay = float(fdf["departure_delay"].dropna().mean()) if total_flights else 0.0
    cancel_rate = float(fdf["cancelled"].mean() * 100) if total_flights else 0.0

    m1, m2, m3, m4 = st.columns(4)
    m1.metric("Total Flights", fmt_num(total_flights))
    m2.metric("Total Delay (min)", fmt_num(total_delay_minutes))
    m3.metric("Avg Departure Delay (min)", f"{avg_dep_delay:.1f}")
    m4.metric("Cancellation Rate", f"{cancel_rate:.2f}%")

    weekday_order = ["Monday", "Tuesday", "Wednesday", "Thursday", "Friday", "Saturday", "Sunday"]

    cancel_by_airline = (
        fdf.groupby("airline", dropna=False)["cancelled"]
        .mean()
        .mul(100)
        .sort_values(ascending=False)
        .reset_index(name="cancel_rate")
    )
    cancel_by_airline["airline"] = cancel_by_airline["airline"].fillna("Unknown")

    fig_cancel = px.bar(
        cancel_by_airline.head(12).sort_values("cancel_rate", ascending=True),
        x="cancel_rate",
        y="airline",
        orientation="h",
        title="Cancel Rate (%) by airline",
        labels={"cancel_rate": "Cancel Rate (%)", "airline": ""},
        color_discrete_sequence=[ACCENT],
    )
    fig_cancel.update_layout(height=360)
    apply_axes_style(fig_cancel)

    map_df = (
        fdf.groupby(["destination", "dest_name", "dest_latitude", "dest_longitude"], dropna=False)
        .size()
        .reset_index(name="flights")
    ).dropna(subset=["dest_latitude", "dest_longitude"])

    fig_geo = None
    if len(map_df) > 0:
        fig_geo = px.scatter_geo(
            map_df,
            lat="dest_latitude",
            lon="dest_longitude",
            size="flights",
            size_max=26,
            hover_name="destination",
            hover_data={"dest_name": True, "flights": True, "dest_latitude": False, "dest_longitude": False},
            projection="albers usa",
            title="Destinations map",
            color_discrete_sequence=[ACCENT],
        )
        fig_geo.update_layout(height=360, **PLOT_LAYOUT)
        fig_geo.update_geos(
            scope="usa",
            showcountries=False,
            showland=True,
            landcolor="rgba(255,255,255,0.04)",
            showlakes=True,
            lakecolor="rgba(255,255,255,0.04)",
            coastlinecolor="rgba(255,255,255,0.18)",
            bgcolor="rgba(0,0,0,0)",
        )

    usage = (
        fdf.groupby("model", dropna=False)
        .size()
        .sort_values(ascending=False)
        .head(12)
        .reset_index(name="flights")
    )
    usage["model"] = usage["model"].fillna("Unknown")

    fig_usage = px.bar(
        usage.sort_values("flights", ascending=True),
        x="flights",
        y="model",
        orientation="h",
        title="Aircraft Usage by Model",
        labels={"flights": "Number of Flights", "model": "Airplane Model"},
        color_discrete_sequence=[ACCENT],
    )
    fig_usage.update_layout(height=340)
    apply_axes_style(fig_usage)

    wdf = (
        fdf.groupby("weekday", dropna=False)
        .size()
        .reindex(weekday_order)
        .fillna(0)
        .reset_index(name="flights")
    )
    fig_week = px.bar(
        wdf,
        x="weekday",
        y="flights",
        title="Total Flights by FlightWeekday",
        labels={"weekday": "FlightWeekday", "flights": "Total Flights"},
        color_discrete_sequence=[ACCENT],
    )
    fig_week.update_layout(height=340, **PLOT_LAYOUT)
    fig_week.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig_week.update_xaxes(showgrid=False)
    fig_week.update_traces(marker_line_width=0)

    daily = (
        fdf.dropna(subset=["scheduled_departure"])
        .assign(day=lambda d: pd.to_datetime(d["scheduled_departure"]).dt.date)
        .groupby("day")
        .agg(avg_dep_delay=("departure_delay", "mean"))
        .reset_index()
    )
    fig_line = px.line(
        daily,
        x="day",
        y="avg_dep_delay",
        title="Average departure delay over time (minutes)",
        labels={"day": "", "avg_dep_delay": "Avg departure delay (min)"},
        color_discrete_sequence=[ACCENT],
    )
    fig_line.update_layout(height=320, **PLOT_LAYOUT)
    fig_line.update_yaxes(showgrid=True, gridcolor=GRID_COLOR, zeroline=False)
    fig_line.update_xaxes(showgrid=False)

    row1 = st.columns([1.05, 1.25], gap="large")
    with row1[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ev_cancel = plotly_select(fig_cancel, key=f"cancel_chart_{rt}")
        st.markdown("</div>", unsafe_allow_html=True)

    with row1[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        if fig_geo is None:
            st.info("No destination coordinates available for the current selection.")
            ev_geo = None
        else:
            ev_geo = plotly_select(fig_geo, key=f"geo_chart_{rt}")
        st.markdown("</div>", unsafe_allow_html=True)

    row2 = st.columns([1.05, 1.25], gap="large")
    with row2[0]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ev_usage = plotly_select(fig_usage, key=f"usage_chart_{rt}")
        st.markdown("</div>", unsafe_allow_html=True)

    with row2[1]:
        st.markdown('<div class="card">', unsafe_allow_html=True)
        ev_week = plotly_select(fig_week, key=f"weekday_chart_{rt}")
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown("")
    st.markdown('<div class="card">', unsafe_allow_html=True)
    st.plotly_chart(fig_line, use_container_width=True, config={"displayModeBar": True}, key=f"line_chart_{rt}")
    st.markdown("</div>", unsafe_allow_html=True)

    new_airline = extract_clicked_value(ev_cancel, prefer_field="y")
    if new_airline:
        st.session_state.sel_airline = None if st.session_state.sel_airline == new_airline else new_airline
        st.rerun()

    new_dest = extract_clicked_value(ev_geo, prefer_field="hovertext")
    if new_dest is None:
        new_dest = extract_clicked_value(ev_geo, prefer_field="text")
    if new_dest:
        st.session_state.sel_destination = None if st.session_state.sel_destination == new_dest else new_dest
        st.rerun()

    new_model = extract_clicked_value(ev_usage, prefer_field="y")
    if new_model:
        st.session_state.sel_model = None if st.session_state.sel_model == new_model else new_model
        st.rerun()

    new_weekday = extract_clicked_value(ev_week, prefer_field="x")
    if new_weekday:
        st.session_state.sel_weekday = None if st.session_state.sel_weekday == new_weekday else new_weekday
        st.rerun()

    if not PLOTLY_SELECT_OK:
        st.warning("Chart click cross-filtering is not available in your Streamlit version. Upgrade Streamlit.")

with tab_data:

    all_cols = list(fdf.columns)
    default_cols = [c for c in [
        "flight_id","flight_date","weekday","airline","manufacturer","model",
        "origin","origin_name","destination","dest_name",
        "scheduled_departure","departure","scheduled_arrival","arrival",
        "departure_delay","arrival_delay","cancelled"
    ] if c in all_cols]
    if not default_cols:
        default_cols = all_cols[:20]
    
    col_sel = st.multiselect(
    "",
    options=all_cols,
    default=default_cols,
    key=f"data_cols_{rt}",
    )

    st.markdown("</div>", unsafe_allow_html=True)


    show_df = fdf[col_sel].copy() if col_sel else fdf.copy()

    st.dataframe(show_df, use_container_width=True, height=560)
    st.markdown("</div>", unsafe_allow_html=True)

    csv_bytes = show_df.to_csv(index=False).encode("utf-8")
    st.download_button(
        label="Download filtered data as CSV",
        data=csv_bytes,
        file_name="filtered_flights.csv",
        mime="text/csv",
        use_container_width=False,
        key=f"dl_csv_{rt}",
    )
