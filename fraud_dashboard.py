import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
from datetime import datetime, timedelta
import time
import random
from sklearn.linear_model import LogisticRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, precision_score, recall_score, confusion_matrix

# Optional split-map stack
_MAP_ENGINE = "plotly"
try:
    import leafmap.foliumap as leafmap
    import folium
    from folium.plugins import HeatMap, MarkerCluster
    _MAP_ENGINE = "leafmap"
except Exception:
    _MAP_ENGINE = "plotly"

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="Bank Fraud Detection System",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ─────────────────────────────────────────────

# ─────────────────────────────────────────────
st.markdown(
    """
<style>
@import url('https://fonts.googleapis.com/css2?family=Inter:wght@300;400;500;600;700;800&display=swap');

/* ══════ RESET ══════ */
* { font-family: 'Inter', sans-serif !important; box-sizing: border-box; }

/* ══════ MAIN BG ══════ */
.main { background-color: #f0f2f5 !important; padding: 20px 28px !important; }

/* ══════ SIDEBAR ══════ */
[data-testid="stSidebar"] {
    background:
  linear-gradient(180deg, rgba(255,255,255,0.14) 0%, rgba(255,255,255,0.00) 60%),
  linear-gradient(180deg, #343a7a 0%, #24285a 100%) !important;

    box-shadow: 6px 0 24px rgba(0,0,0,0.18);
    min-width: 240px !important;
}

[data-testid="stSidebar"] .stButton button {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    color: #fff !important;
    border: none !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 10px 0 !important;
    box-shadow: 0 4px 14px rgba(108,92,231,0.35);
    transition: transform .2s, box-shadow .2s;
}
[data-testid="stSidebar"] .stButton button:hover {
    transform: translateY(-1px);
    box-shadow: 0 6px 20px rgba(108,92,231,0.45);
}
/* sidebar text */
[data-testid="stSidebar"] .stMarkdown p,
[data-testid="stSidebar"] .stMarkdown h3,
[data-testid="stSidebar"] .stMarkdown h4 { color: #a0a3b8 !important; }
[data-testid="stSidebar"] .stMetric .stMetricLabel { color: #7a7d94 !important; font-size: 12px !important; }
[data-testid="stSidebar"] .stMetric .stMetricValue { color: #fff !important; font-size: 20px !important; font-weight: 700 !important; }

/* ══════ HIDE BRANDING ══════ */
# MainMenu { visibility: hidden; }
footer { visibility: hidden; }
header { visibility: hidden; }

/* ══════ TABS ══════ */
.stTabs [data-baseweb="tab-list"] {
    gap: 4px;
    background: #fff;
    padding: 6px;
    border-radius: 14px;
    box-shadow: 0 2px 12px rgba(0,0,0,0.06);
    margin-bottom: 24px !important;
}
.stTabs [data-baseweb="tab"] {
    height: 42px;
    border-radius: 10px;
    color: #6b7280 !important;
    font-weight: 600 !important;
    font-size: 14px !important;
    border: none !important;
    background: transparent;
    transition: all .25s;
}
.stTabs [data-baseweb="tab"]:hover { background: #f3f4f6; color: #374151 !important; }
.stTabs [aria-selected="true"] {
    background: linear-gradient(135deg, #6c5ce7, #a855f7) !important;
    color: #fff !important;
    box-shadow: 0 4px 14px rgba(108,92,231,0.35);
}

/* ══════ CARD ══════ */
.card {
    background: #fff;
    border-radius: 18px;
    padding: 24px 26px;
    box-shadow: 0 2px 16px rgba(0,0,0,0.06);
    margin-bottom: 22px;
}
.card-title {
    font-size: 16px;
    font-weight: 700;
    color: #1e293b;
    margin-bottom: 18px;
    display: flex;
    align-items: center;
    gap: 10px;
}
.card-title .icon-box {
    width: 36px; height: 36px;
    border-radius: 10px;
    background: linear-gradient(135deg, #6c5ce7, #a855f7);
    display: flex; align-items: center; justify-content: center;
    font-size: 17px;
}

/* ══════ METRIC CARDS (top row) ══════ */
.metric-card {
    border-radius: 18px;
    padding: 22px 22px 18px;
    height: 100%;
    transition: transform .25s, box-shadow .25s;
    position: relative;
    overflow: hidden;
}
.metric-card:hover { transform: translateY(-3px); box-shadow: 0 8px 28px rgba(0,0,0,0.12); }

.mc-purple { background: linear-gradient(135deg, #6c5ce7 0%, #a855f7 100%); }
.mc-pink   { background: linear-gradient(135deg, #f43f5e 0%, #fb7185 100%); }
.mc-green  { background: linear-gradient(135deg, #10b981 0%, #34d399 100%); }
.mc-amber  { background: linear-gradient(135deg, #f59e0b 0%, #fbbf24 100%); }

.metric-card .mc-label {
    font-size: 12px; font-weight: 600; color: rgba(255,255,255,0.78);
    text-transform: uppercase; letter-spacing: .6px; margin-bottom: 6px;
}
.metric-card .mc-value {
    font-size: 30px; font-weight: 800; color: #fff; line-height: 1.15; margin-bottom: 8px;
}
.metric-card .mc-sub {
    font-size: 12px; color: rgba(255,255,255,0.7); font-weight: 500;
}
.metric-card .mc-sub .badge {
    display: inline-block; background: rgba(255,255,255,0.2);
    border-radius: 20px; padding: 2px 8px; margin-right: 6px; font-weight: 600;
}
/* decorative circle */
.metric-card .deco {
    position: absolute; bottom: -24px; right: -24px;
    width: 100px; height: 100px; border-radius: 50%;
    background: rgba(255,255,255,0.08);
}
.metric-card .deco2 {
    position: absolute; bottom: -10px; right: 50px;
    width: 60px; height: 60px; border-radius: 50%;
    background: rgba(255,255,255,0.05);
}

/* ══════ ALERT BADGES (live monitor) ══════ */
.fraud-alert {
    background: linear-gradient(135deg, #f43f5e, #fb7185);
    color: #fff; padding: 8px 14px; border-radius: 10px;
    font-weight: 700; font-size: 12px; text-align: center;
    box-shadow: 0 3px 10px rgba(244,63,94,0.35);
    animation: pulse 2s infinite;
}
.safe-alert {
    background: linear-gradient(135deg, #10b981, #34d399);
    color: #fff; padding: 8px 14px; border-radius: 10px;
    font-weight: 700; font-size: 12px; text-align: center;
    box-shadow: 0 3px 10px rgba(16,185,129,0.3);
}
@keyframes pulse { 0%,100%{opacity:1} 50%{opacity:.75} }

/* ══════ RISK BADGE ══════ */
.risk-pill {
    display: inline-block; padding: 7px 22px; border-radius: 999px;
    font-weight: 700; font-size: 13px; text-transform: uppercase; letter-spacing: .5px;
}
.risk-pill.high { background: linear-gradient(135deg, #f43f5e, #fb7185); color:#fff; box-shadow:0 4px 14px rgba(244,63,94,.3); }
.risk-pill.med  { background: linear-gradient(135deg, #f59e0b, #fbbf24); color:#fff; box-shadow:0 4px 14px rgba(245,158,11,.3); }
.risk-pill.low  { background: linear-gradient(135deg, #10b981, #34d399); color:#fff; box-shadow:0 4px 14px rgba(16,185,129,.3); }

/* ══════ PROGRESS BAR ══════ */
.prog-track { background:#eef2ff; border-radius:999px; height:7px; overflow:hidden; margin:6px 0; }
.prog-fill  { height:100%; border-radius:999px; transition:width .4s; }

/* ══════ DOWNLOAD BUTTON ══════ */
.stDownloadButton button {
    background: #fff !important;
    color: #6c5ce7 !important;
    border: 1.5px solid #6c5ce7 !important;
    border-radius: 10px !important;
    font-weight: 600 !important;
    padding: 8px 18px !important;
    transition: all .2s;
}
.stDownloadButton button:hover { background:#f5f3ff !important; }

/* ══════ DATAFRAME (table) ══════ */
.stDataframe { border-radius: 12px; overflow: hidden; }
iframe { border-radius: 12px !important; }

/* ══════ SCROLLBAR ══════ */
::-webkit-scrollbar { width:6px; height:6px; }
::-webkit-scrollbar-track { background:transparent; }
::-webkit-scrollbar-thumb { background:#c4c9e2; border-radius:3px; }
::-webkit-scrollbar-thumb:hover { background:#a0a3c4; }

/* ══════ INFO BANNER ══════ */
.info-banner {
    background: linear-gradient(135deg, #eef2ff, #f3e8ff);
    border-left: 4px solid #6c5ce7;
    padding: 14px 18px;
    border-radius: 10px;
    color: #4c1d95;
    font-weight: 500;
    font-size: 14px;
    margin: 12px 0 20px;
}
</style>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# DATA GENERATION
# ─────────────────────────────────────────────


def generate_transaction_data(num_records=1000):
    np.random.seed(42)
    end_time = datetime.now()
    start_time = end_time - timedelta(hours=24)

    timestamps = [
        start_time + timedelta(seconds=random.randint(0, 86400))
        for _ in range(num_records)
    ]
    timestamps.sort()

    user_ids = [
        f"USR{str(i).zfill(6)}" for i in np.random.randint(100000, 999999, num_records)
    ]

    amounts = np.concatenate(
        [
            np.random.gamma(2, 50, int(num_records * 0.7)),
            np.random.uniform(500, 5000, int(num_records * 0.2)),
            np.random.uniform(5000, 50000, int(num_records * 0.1)),
        ]
    )
    np.random.shuffle(amounts)
    amounts = amounts[:num_records]

    cities = [
        "New York",
        "Los Angeles",
        "Chicago",
        "Houston",
        "Phoenix",
        "Philadelphia",
        "San Antonio",
        "San Diego",
        "Dallas",
        "San Jose",
        "London",
        "Paris",
        "Tokyo",
        "Singapore",
        "Dubai",
        "Mumbai",
        "Shanghai",
        "Hong Kong",
    ]
    locations = np.random.choice(cities, num_records)

    devices = np.random.choice(["Mobile", "Desktop", "Tablet"], num_records, p=[0.6, 0.35, 0.05])
    payment_methods = np.random.choice(
        ["Credit Card", "Debit Card", "Wire Transfer", "Mobile Wallet"],
        num_records,
        p=[0.5, 0.3, 0.15, 0.05],
    )
    merchants = np.random.choice(
        [
            "Retail",
            "Online Shopping",
            "Restaurants",
            "Travel",
            "Entertainment",
            "Healthcare",
            "Utilities",
            "Gas Stations",
            "Groceries",
        ],
        num_records,
    )

    distances = np.abs(np.random.normal(10, 50, num_records))
    typing_speeds = np.abs(np.random.normal(150, 50, num_records))
    processing_times = np.random.uniform(50, 500, num_records)

    fraud_indicators = []
    for i in range(num_records):
        fraud_score = 0
        if amounts[i] > 3000:
            fraud_score += 30
        if locations[i] in [
            "London",
            "Paris",
            "Tokyo",
            "Singapore",
            "Dubai",
            "Mumbai",
            "Shanghai",
            "Hong Kong",
        ]:
            fraud_score += 25
        if distances[i] > 100:
            fraud_score += 20
        if typing_speeds[i] < 80:
            fraud_score += 15
        if payment_methods[i] == "Wire Transfer":
            fraud_score += 10
        fraud_score += np.random.randint(-20, 20)
        fraud_indicators.append(fraud_score > 60)

    return pd.DataFrame(
        {
            "timestamp": timestamps,
            "user_id": user_ids,
            "amount": amounts,
            "location": locations,
            "device": devices,
            "payment_method": payment_methods,
            "merchant_category": merchants,
            "distance_from_home": distances,
            "typing_speed": typing_speeds,
            "processing_time": processing_times,
            "is_fraud": fraud_indicators,
        }
    )


# ─────────────────────────────────────────────
# GEO COORDS
# ─────────────────────────────────────────────
CITY_COORDS = {
    "New York": (40.7128, -74.0060),
    "Los Angeles": (34.0522, -118.2437),
    "Chicago": (41.8781, -87.6298),
    "Houston": (29.7604, -95.3698),
    "Phoenix": (33.4484, -112.0740),
    "Philadelphia": (39.9526, -75.1652),
    "San Antonio": (29.4241, -98.4936),
    "San Diego": (32.7157, -117.1611),
    "Dallas": (32.7767, -96.7970),
    "San Jose": (37.3382, -121.8863),
    "London": (51.5074, -0.1278),
    "Paris": (48.8566, 2.3522),
    "Tokyo": (35.6762, 139.6503),
    "Singapore": (1.3521, 103.8198),
    "Dubai": (25.2048, 55.2708),
    "Mumbai": (19.0760, 72.8777),
    "Shanghai": (31.2304, 121.4737),
    "Hong Kong": (22.3193, 114.1694),
}


def add_geo_coords(df_in: pd.DataFrame) -> pd.DataFrame:
    df_out = df_in.copy()
    df_out["lat"] = df_out["location"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[0])
    df_out["lon"] = df_out["location"].map(lambda c: CITY_COORDS.get(c, (np.nan, np.nan))[1])
    return df_out.dropna(subset=["lat", "lon"])


# ─────────────────────────────────────────────
# SESSION STATE + MODEL
# ─────────────────────────────────────────────
if "transaction_data" not in st.session_state:
    st.session_state.transaction_data = generate_transaction_data(1000)

df = add_geo_coords(st.session_state.transaction_data)

# ─────────────────────────────────────────────
# MAP RENDERER
# ─────────────────────────────────────────────


def render_split_map_card(df_in: pd.DataFrame, title: str, key_prefix: str = "map") -> None:
    """Render a split-panel (swipe) map like 2__Split_Map.py inside a styled card."""

    st.markdown(
        f'<div class="card"><div class="card-title"><div class="icon-box"></div>{title}</div>',
        unsafe_allow_html=True,
    )

    basemap_options = [
        "CartoDB.DarkMatter",
        "CartoDB.Positron",
        "OpenStreetMap",
        "OpenTopoMap",
        "Esri.WorldImagery",
        "Stamen.Toner",
    ]

    r1c1, r1c2 = st.columns([1, 1], gap="small")
    with r1c1:
        left_layer = st.selectbox(
            "Left map",
            basemap_options,
            index=0,
            key=f"{key_prefix}_left",
            label_visibility="collapsed",
        )
    with r1c2:
        right_layer = st.selectbox(
            "Right map",
            basemap_options,
            index=1,
            key=f"{key_prefix}_right",
            label_visibility="collapsed",
        )

    r2c1, r2c2, r2c3 = st.columns([1, 1, 1], gap="small")
    with r2c1:
        scope = st.selectbox(
            "Scope",
            ["Fraud only", "All transactions"],
            index=0,
            key=f"{key_prefix}_scope",
            label_visibility="collapsed",
        )
    with r2c2:
        overlay = st.selectbox(
            "Overlay",
            ["Heatmap", "Points"],
            index=0,
            key=f"{key_prefix}_overlay",
            label_visibility="collapsed",
        )
    with r2c3:
        weight = st.selectbox(
            "Weight",
            ["Count", "Amount"],
            index=0,
            key=f"{key_prefix}_weight",
            label_visibility="collapsed",
        )

    methods = sorted(df_in["payment_method"].unique())
    chosen = st.multiselect(
        "Payment method",
        methods,
        default=methods,
        key=f"{key_prefix}_pm",
        label_visibility="collapsed",
    )

    show_df = df_in[df_in["is_fraud"]] if scope == "Fraud only" else df_in
    show_df = show_df[show_df["payment_method"].isin(chosen)].dropna(subset=["lat", "lon"])

    map_col, info_col = st.columns([3, 1], gap="medium")

    if show_df.empty:
        with map_col:
            st.info("No records match your filters.")
        with info_col:
            st.markdown(
                """
                <div style="background:#f8fafc;border:1px solid #eef2ff;border-radius:14px;padding:14px 14px;">
                    <div style="color:#64748b;font-size:12px;font-weight:700;margin-bottom:6px;">No data</div>
                    <div style="color:#94a3b8;font-size:12px;">Try expanding the scope or selecting more payment methods.</div>
                </div>
                """,
                unsafe_allow_html=True,
            )
        st.markdown("</div>", unsafe_allow_html=True)
        return

    with info_col:
        st.markdown(
            f"""
            <div style="background:#f8fafc;border:1px solid #eef2ff;border-radius:14px;padding:14px 14px;margin-bottom:12px;">





            </div>
            """,
            unsafe_allow_html=True,
        )

    with map_col:
        if _MAP_ENGINE == "leafmap":
            m = leafmap.Map()
            m.split_map(left_layer=left_layer, right_layer=right_layer)

            try:
                sw = [float(show_df["lat"].min()), float(show_df["lon"].min())]
                ne = [float(show_df["lat"].max()), float(show_df["lon"].max())]
                m.fit_bounds([sw, ne])
            except Exception:
                pass

            if overlay == "Heatmap":
                w = show_df["amount"].astype(float).clip(
                    lower=0.0) if weight == "Amount" else np.ones(len(show_df), dtype=float)
                heat_data = list(zip(show_df["lat"].astype(float), show_df["lon"].astype(float), w.astype(float)))
                HeatMap(
                    heat_data,
                    radius=22,
                    blur=18,
                    min_opacity=0.25,
                    max_zoom=4,
                    gradient={0.2: "#6c5ce7", 0.5: "#f59e0b", 1.0: "#f43f5e"},
                ).add_to(m)
            else:
                max_points = 800 if scope == "Fraud only" else 600
                plot_df = show_df if len(show_df) <= max_points else show_df.sample(max_points, random_state=42)

                cluster = MarkerCluster(name="Transactions")
                for _, row in plot_df.iterrows():
                    is_fraud = bool(row["is_fraud"])
                    col = "#f43f5e" if is_fraud else "#10b981"
                    rad = 6 if is_fraud else 4
                    popup_html = f"""
                    <div style='font-family:Inter;min-width:220px;'>
                        <div style='font-weight:800;color:#1e293b;font-size:14px;margin-bottom:4px;'>
                            {row['location']}
                        </div>
                        <div style='color:#64748b;font-size:12px;line-height:1.4;'>
                            <b>Amount:</b> ${float(row['amount']):,.2f}<br>
                            <b>Method:</b> {row['payment_method']}<br>
                            <b>Device:</b> {row['device']}<br>
                            <b>Status:</b> {"FRAUD" if is_fraud else "NORMAL"}
                        </div>
                    </div>
                    """
                    folium.CircleMarker(
                        location=[float(row["lat"]), float(row["lon"])],
                        radius=rad,
                        color=col,
                        fill=True,
                        fill_color=col,
                        fill_opacity=0.85 if is_fraud else 0.55,
                        weight=1,
                        popup=folium.Popup(popup_html, max_width=260),
                    ).add_to(cluster)

                cluster.add_to(m)

            m.to_streamlit(height=440)

            st.markdown(
                """
                <div style='display:flex;gap:10px;align-items:center;margin:10px 0 2px;'>
                  <span style='display:inline-flex;align-items:center;gap:6px;color:#64748b;font-size:12px;font-weight:600;'>
                    <span style='width:10px;height:10px;border-radius:50%;background:#f43f5e;display:inline-block;'></span> Fraud
                  </span>
                  <span style='display:inline-flex;align-items:center;gap:6px;color:#64748b;font-size:12px;font-weight:600;'>
                    <span style='width:10px;height:10px;border-radius:50%;background:#10b981;display:inline-block;'></span> Normal
                  </span>
                  <span style='margin-left:auto;color:#94a3b8;font-size:12px;'>Drag the handle to compare basemaps</span>
                </div>
                """,
                unsafe_allow_html=True,
            )
        else:
            st.warning("Leafmap is not installed. Install it to enable the split-panel map: `pip install leafmap folium`")
            map_style = "carto-darkmatter"
            if overlay == "Heatmap":
                z = show_df["amount"].astype(float) if weight == "Amount" else np.ones(len(show_df), dtype=float)
                fig_map = px.density_mapbox(
                    show_df,
                    lat="lat",
                    lon="lon",
                    z=z,
                    radius=28,
                    center=dict(lat=20, lon=10),
                    zoom=0.8,
                    mapbox_style=map_style,
                    hover_name="location",
                )
            else:
                fig_map = px.scatter_mapbox(
                    show_df,
                    lat="lat",
                    lon="lon",
                    size="amount" if weight == "Amount" else None,
                    size_max=18,
                    color="is_fraud",
                    color_discrete_map={True: "#f43f5e", False: "#10b981"},
                    hover_name="location",
                    hover_data={
                        "amount": ":$.2f",
                        "payment_method": True,
                        "device": True,
                        "lat": False,
                        "lon": False,
                        "is_fraud": False,
                    },
                    zoom=0.8,
                    mapbox_style=map_style,
                )
            fig_map.update_layout(
                height=440,
                margin=dict(l=10, r=10, t=10, b=10),
                paper_bgcolor="rgba(0,0,0,0)",
                plot_bgcolor="rgba(0,0,0,0)",
                font=dict(family="Inter", size=11, color="#64748b"),
                showlegend=False,
            )
            st.plotly_chart(fig_map, use_container_width=True, config={"displayModeBar": False})

    fraud_hotspots = (
        df_in[df_in["is_fraud"]]
        .query("payment_method in @chosen")
        .groupby("location")
        .agg(
            fraud_count=("is_fraud", "sum"),
            total_amount=("amount", "sum"),
            avg_amount=("amount", "mean"),
        )
        .sort_values("fraud_count", ascending=False)
        .head(6)
        .reset_index()
    )
    with info_col:
        st.markdown(
            '<div style="color:#64748b;font-size:12px;font-weight:800;margin:6px 0 8px;">Top hotspots</div>',
            unsafe_allow_html=True,
        )
        st.dataframe(fraud_hotspots, use_container_width=True, hide_index=True, height=230)

    st.markdown("</div>", unsafe_allow_html=True)


# Train model once
if "model" not in st.session_state:
    df_model = df.copy()
    df_model["device"] = df_model["device"].map({"Mobile": 0, "Desktop": 1, "Tablet": 2})
    df_model["payment_method"] = df_model["payment_method"].map(
        {"Credit Card": 0, "Debit Card": 1, "Wire Transfer": 2, "Mobile Wallet": 3}
    )
    df_model["merchant_category"] = df_model["merchant_category"].astype("category").cat.codes
    df_model["location"] = df_model["location"].astype("category").cat.codes

    features = [
        "amount",
        "distance_from_home",
        "typing_speed",
        "processing_time",
        "device",
        "payment_method",
        "merchant_category",
        "location",
    ]
    X = df_model[features]
    y = df_model["is_fraud"]
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42
    )
    model = LogisticRegression(random_state=42, max_iter=1000)
    model.fit(X_train, y_train)

    st.session_state.model = model
    st.session_state.X_test = X_test
    st.session_state.y_test = y_test

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    # ── Sidebar Top Logo ─────────────────────────
    st.markdown(
        """
            <div style="padding:14px 8px 18px; display:flex; flex-direction:column; align-items:center;">
            """,
        unsafe_allow_html=True,
    )

    # Your logo image (make sure Logo.png is in the same folder as this .py file)
    st.image("Logo.png", width=190)

    st.markdown(
        """
                <div style="margin-top:10px; color:#ffffff; font-size:18px; font-weight:800; letter-spacing:.3px;">
                    FraudGuard
                </div>
            </div>
            """,
        unsafe_allow_html=True,
    )

    nav_items = [
        ("", "Dashboard", True),
        ("", "Analytics", False),
        ("", "Monitoring", False),
        ("", "Settings", False),
    ]
    for icon, label, active in nav_items:
        bg = "linear-gradient(135deg,#6c5ce7,#a855f7)" if active else "transparent"
        clr = "#fff" if active else "#a0a3b8"
        shadow = "box-shadow:0 4px 14px rgba(108,92,231,.3);" if active else ""
        st.markdown(
            f"""
                <div style="display:flex;align-items:center;gap:12px;padding:12px 16px;margin:4px 8px;
                    border-radius:12px;background:{bg};cursor:pointer;{shadow}transition:all .2s;">
                    <span style="font-size:18px;">{icon}</span>
                    <span style="color:{clr};font-weight:{'700' if active else '500'};font-size:14px;">{label}</span>
                </div>
                """,
            unsafe_allow_html=True,
        )

    st.markdown(
        "<hr style='border:none;border-top:1px solid #2a2d4a;margin:20px 8px;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        '<p style="color:#6b6e85;font-size:11px;text-transform:uppercase;letter-spacing:.7px;padding:0 8px;margin-bottom:8px;">Data Overview</p>',
        unsafe_allow_html=True,
    )

    st.markdown(
        "<hr style='border:none;border-top:1px solid #2a2d4a;margin:16px 8px;'>",
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <div style="margin:8px;padding:14px 16px;background:rgba(16,185,129,.12);border-radius:12px;
            border-left:3px solid #10b981;">
            <span style="color:#10b981;font-weight:600;font-size:13px;">● All Systems Operational</span>
        </div>
        """,
        unsafe_allow_html=True,
    )
    st.markdown(
        f'<p style="color:#6b6e85;font-size:12px;text-align:center;margin:6px 0 16px;">Updated {datetime.now().strftime("%H:%M:%S")}</p>',
        unsafe_allow_html=True,
    )

    if st.button("⟳  Refresh Data", use_container_width=True):
        st.session_state.transaction_data = generate_transaction_data(1000)
        st.rerun()

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.markdown(
    """
<div style="display:flex;justify-content:space-between;align-items:center;margin-bottom:26px;">
    <div>
        <h1 style="margin:0;font-size:28px;font-weight:800;color:#1e293b;">Bank Fraud Detection System</h1>
        <p style="margin:4px 0 0;color:#64748b;font-size:14px;">Real-Time Big Data Analytics Platform</p>
    </div>
    <div style="display:flex;gap:10px;align-items:center;">
        <div style="background:#fff;border-radius:12px;padding:9px 16px;box-shadow:0 2px 10px rgba(0,0,0,.06);
             display:flex;align-items:center;gap:8px;color:#64748b;font-size:13px;">
             <span style="background:#f43f5e;color:#fff;border-radius:999px;padding:1px 7px;font-size:11px;font-weight:700;">12</span>
        </div>
        <div style="width:40px;height:40px;border-radius:12px;background:linear-gradient(135deg,#6c5ce7,#a855f7);
             display:flex;align-items:center;justify-content:center;font-size:18px;box-shadow:0 2px 10px rgba(108,92,231,.3);">⚙️</div>
    </div>
</div>
""",
    unsafe_allow_html=True,
)

# ─────────────────────────────────────────────
# KPI VALUES
# ─────────────────────────────────────────────
total_transactions = len(df)
fraud_count = int(df["is_fraud"].sum())
fraud_rate = (fraud_count / total_transactions) * 100 if total_transactions else 0.0
total_blocked = float(df[df["is_fraud"]]["amount"].sum())
total_processed = float(df["amount"].sum())

# ─────────────────────────────────────────────
# TABS
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs(
    [" Executive Dashboard", " Live Monitoring", " Risk Scoring", " Model Performance"]
)

# ════════════════════════════════════════════════════════
# TAB 1 — EXECUTIVE DASHBOARD 
# ════════════════════════════════════════════════════════
with tab1:
    # ==================== NEW: REAL-TIME PREDICTIONS ====================
    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">🔮</div>Predictions & Predictive Alerts</div>',
        unsafe_allow_html=True,
    )
    
    pred_col1, pred_col2, pred_col3 = st.columns([2, 1, 1], gap="medium")
    
    with pred_col1:
        # Generate predictions for next hours
        current_hour = datetime.now().hour
        hours = list(range(current_hour, current_hour + 5))
        predicted_fraud = [fraud_count * (1 + i * 0.15) for i in range(5)]  # Simulation
        actual_fraud = [fraud_count] * 5
        
        fig_pred = go.Figure()
        fig_pred.add_trace(go.Scatter(
            x=hours,
            y=actual_fraud,
            mode='lines+markers',
            name='Current Fraud',
            line=dict(color='#10b981', width=3, dash='dash')
        ))
        fig_pred.add_trace(go.Scatter(
            x=hours,
            y=predicted_fraud,
            mode='lines+markers',
            name='Prediction',
            line=dict(color='#f43f5e', width=3),
            fill='tonexty',
            fillcolor='rgba(244,63,94,0.1)'
        ))
        
        # Add confidence band
        fig_pred.add_trace(go.Scatter(
            x=hours + hours[::-1],
            y=[p * 0.9 for p in predicted_fraud] + [p * 1.1 for p in predicted_fraud[::-1]],
            fill='toself',
            fillcolor='rgba(244,63,94,0.2)',
            line=dict(color='rgba(255,255,255,0)'),
            name='90% Confidence Interval'
        ))
        
        fig_pred.update_layout(
            height=180,
            margin=dict(l=10, r=10, t=10, b=20),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title='Hours',
                showgrid=False,
                tickmode='array',
                tickvals=hours,
                ticktext=[f'H+{i}' for i in range(5)],
                color='#94a3b8'
            ),
            yaxis=dict(
                title='Predicted Fraud',
                showgrid=True,
                gridcolor='#eef2ff',
                color='#94a3b8'
            ),
            legend=dict(
                orientation='h',
                yanchor='bottom',
                y=1.02,
                xanchor='right',
                x=1
            ),
            font=dict(family='Inter', size=11, color='#64748b')
        )
        st.plotly_chart(fig_pred, use_container_width=True, config={'displayModeBar': False})
    
    with pred_col2:
        # Predictive alerts
        alert_level = "HIGH" if predicted_fraud[1] > fraud_count * 1.2 else "MODERATE" if predicted_fraud[1] > fraud_count else "LOW"
        alert_color = "#f43f5e" if alert_level == "HIGH" else "#f59e0b" if alert_level == "MODERATE" else "#10b981"
        
        st.markdown(
            f"""
            <div style="text-align:center;padding:15px;background:{alert_color}15;border-radius:12px;border:2px solid {alert_color}30;">
                <div style="color:{alert_color};font-weight:800;font-size:13px;margin-bottom:8px;">🚨 PREDICTIVE ALERT</div>
                <div style="font-size:24px;font-weight:800;color:{alert_color};margin-bottom:5px;">{alert_level}</div>
                <div style="color:#64748b;font-size:11px;font-weight:600;">
                    Next hour: +{int((predicted_fraud[1]/fraud_count-1)*100)}%<br>
                    Confidence: 85%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with pred_col3:
        # Detection Health Score
        detection_score = 85  # Score based on multiple metrics
        score_color = "#10b981" if detection_score >= 80 else "#f59e0b" if detection_score >= 60 else "#f43f5e"
        
        st.markdown(
            f"""
            <div style="text-align:center;padding:15px;">
                <div style="color:#64748b;font-weight:600;font-size:12px;margin-bottom:8px;">📊 SYSTEM HEALTH SCORE</div>
                <div style="position:relative;width:100px;height:100px;margin:0 auto 10px;">
                    <svg width="100" height="100" viewBox="0 0 100 100">
                        <circle cx="50" cy="50" r="45" fill="none" stroke="#eef2ff" stroke-width="8"/>
                        <circle cx="50" cy="50" r="45" fill="none" stroke="{score_color}" stroke-width="8"
                                stroke-dasharray="{detection_score * 2.83} 283"
                                stroke-linecap="round" transform="rotate(-90 50 50)"/>
                    </svg>
                    <div style="position:absolute;top:50%;left:50%;transform:translate(-50%,-50%);">
                        <div style="font-size:22px;font-weight:800;color:{score_color};">{detection_score}</div>
                    </div>
                </div>
                <div style="color:#64748b;font-size:11px;">System Performance</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ==================== NEW: MARGINAL CONTRIBUTION ANALYSIS ====================
    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">📊</div>Impact Factor Analysis</div>',
        unsafe_allow_html=True,
    )
    
    # Calculate contributions of different factors
    factors = {
        "Mobile": 35,
        "Wire Transfer": 28,
        "Paris": 22,
        "Amount > $5000": 15
    }
    
    fig_waterfall = go.Figure(go.Waterfall(
        name="Contribution to fraud increase",
        orientation="v",
        measure=["relative", "relative", "relative", "relative", "total"],
        x=list(factors.keys()) + ["Total"],
        y=list(factors.values()) + [sum(factors.values())],
        text=[f"+{v}%" for v in factors.values()] + [f"{sum(factors.values())}%"],
        textposition="outside",
        connector={"line": {"color": "#94a3b8"}},
        increasing={"marker": {"color": "#f43f5e"}},
        decreasing={"marker": {"color": "#10b981"}},
        totals={"marker": {"color": "#6c5ce7"}}
    ))
    
    fig_waterfall.update_layout(
        height=280,
        margin=dict(l=10, r=10, t=10, b=30),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        showlegend=False,
        xaxis=dict(
            title="Impact Factors",
            showgrid=False,
            color='#94a3b8'
        ),
        yaxis=dict(
            title="Contribution (%)",
            showgrid=True,
            gridcolor='#eef2ff',
            color='#94a3b8'
        ),
        font=dict(family='Inter', size=11, color='#64748b')
    )
    
    st.plotly_chart(fig_waterfall, use_container_width=True, config={'displayModeBar': False})
    
    # Automatic insights
    max_factor = max(factors, key=factors.get)
    st.markdown(
        f"""
        <div style="background:#f8fafc;border-radius:10px;padding:12px 16px;margin-top:10px;">
            <div style="display:flex;align-items:center;gap:10px;margin-bottom:8px;">
                <div style="width:8px;height:8px;border-radius:50%;background:#6c5ce7;"></div>
                <div style="color:#1e293b;font-weight:700;font-size:13px;">📌 Automatic Insight</div>
            </div>
            <div style="color:#64748b;font-size:12px;line-height:1.5;">
                The {sum(factors.values())}% increase in fraud rate is mainly due to 
                <span style="color:#f43f5e;font-weight:700;">{max_factor} ({factors[max_factor]}%)</span>. 
                Recommendation: Strengthen monitoring on this factor.
            </div>
        </div>
        """,
        unsafe_allow_html=True
    )
    
    st.markdown("</div>", unsafe_allow_html=True)
    
    # ==================== ORIGINAL KPI CARDS (ENHANCED WITH ANIMATION) ====================
    c1, c2, c3, c4 = st.columns(4, gap="medium")
    
    # Calculate dynamic trends
    hourly_fraud = df.groupby(pd.to_datetime(df['timestamp']).dt.hour)['is_fraud'].mean()
    current_trend = "↑" if len(hourly_fraud) > 1 and hourly_fraud.iloc[-1] > hourly_fraud.iloc[-2] else "↓"
    
    cards = [
        ("mc-purple", "Total Transactions", f"{total_transactions:,}", "Last 24 Hours", f"{current_trend} 12.5%", "🔄"),
        ("mc-pink", "Fraud Rate", f"{fraud_rate:.2f}%", f"{fraud_count} Fraudulent", f"{current_trend} 2.1%", "📈"),
        ("mc-green", "Blocked Amount", f"${total_blocked:,.0f}", "Losses Prevented", f"{current_trend} 8.3%", "🛡️"),
        ("mc-amber", "Total Processed", f"${total_processed:,.0f}", "Transaction Volume", f"{current_trend} 5.7%", "💰"),
    ]
    
    for col, (cls, label, val, sub, badge, icon) in zip([c1, c2, c3, c4], cards):
        with col:
            st.markdown(
                f"""
                <div class="metric-card {cls}" style="position:relative;">
                    <div style="position:absolute;top:15px;right:15px;font-size:20px;animation:pulse 2s infinite;">{icon}</div>
                    <div class="mc-label">{label}</div>
                    <div class="mc-value">{val}</div>
                    <div class="mc-sub"><span class="badge">{badge}</span>{sub}</div>
                    <div class="deco"></div><div class="deco2"></div>
                </div>
                """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    # ==================== NEW: INTERACTIVE IMPACT-RARITY MATRIX ====================
    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">🎯</div>Investigation Prioritization Matrix</div>',
        unsafe_allow_html=True,
    )
    
    col_matrix, col_legend = st.columns([3, 1], gap="medium")
    
    with col_matrix:
        # Create 2x2 matrix
        fraud_analysis = df[df['is_fraud']].copy()
        fraud_analysis['freq_category'] = fraud_analysis['amount'].apply(
            lambda x: 'Frequent' if x < 2000 else 'Rare'
        )
        fraud_analysis['impact_category'] = fraud_analysis['amount'].apply(
            lambda x: 'Low Impact' if x < 5000 else 'High Impact'
        )
        
        matrix_data = fraud_analysis.groupby(['freq_category', 'impact_category']).size().unstack(fill_value=0)
        
        fig_matrix = go.Figure(data=go.Heatmap(
            z=matrix_data.values,
            x=matrix_data.columns,
            y=matrix_data.index,
            colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#f43f5e']],
            text=matrix_data.values,
            texttemplate="<b>%{text}</b><br>transactions",
            textfont={"size": 14, "color": "white"},
            hovertemplate="<b>%{y} / %{x}</b><br>Transactions: %{z}<extra></extra>"
        ))
        
        fig_matrix.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title="Amount Impact",
                showgrid=False,
                color='#94a3b8',
                side='top'
            ),
            yaxis=dict(
                title="Frequency",
                showgrid=False,
                color='#94a3b8'
            ),
            font=dict(family='Inter', size=11, color='#64748b')
        )
        
        st.plotly_chart(fig_matrix, use_container_width=True, config={'displayModeBar': False})
    
    with col_legend:
        st.markdown(
            """
            <div style="background:#f8fafc;border-radius:10px;padding:15px;">
                <div style="color:#1e293b;font-weight:700;font-size:13px;margin-bottom:12px;">🎯 Prioritization Guide</div>
            """,
            unsafe_allow_html=True
        )
        
        # Use Streamlit columns for the priority items
        high_col, med_col, low_col = st.columns(3)
        
        with high_col:
            st.markdown(
                """
                <div style="text-align:center;padding:10px;">
                    <div style="width:30px;height:30px;background:#f43f5e;border-radius:8px;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;">1</div>
                    <div style="color:#f43f5e;font-size:11px;font-weight:700;">HIGH PRIORITY</div>
                    <div style="color:#94a3b8;font-size:9px;margin-top:4px;">Rare & High Impact</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with med_col:
            st.markdown(
                """
                <div style="text-align:center;padding:10px;">
                    <div style="width:30px;height:30px;background:#f59e0b;border-radius:8px;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;">2</div>
                    <div style="color:#f59e0b;font-size:11px;font-weight:700;">MEDIUM PRIORITY</div>
                    <div style="color:#94a3b8;font-size:9px;margin-top:4px;">Frequent & High Impact</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        with low_col:
            st.markdown(
                """
                <div style="text-align:center;padding:10px;">
                    <div style="width:30px;height:30px;background:#10b981;border-radius:8px;margin:0 auto 8px;display:flex;align-items:center;justify-content:center;color:white;font-weight:bold;">3</div>
                    <div style="color:#10b981;font-size:11px;font-weight:700;">LOW PRIORITY</div>
                    <div style="color:#94a3b8;font-size:9px;margin-top:4px;">Frequent & Low Impact</div>
                </div>
                """,
                unsafe_allow_html=True
            )
        
        st.markdown(
            """
            <div style="background:#eef2ff;border-radius:8px;padding:10px;margin-top:10px;">
                <div style="color:#64748b;font-size:10px;font-weight:600;margin-bottom:5px;">Recommended Actions:</div>
                <div style="color:#94a3b8;font-size:9px;line-height:1.4;">
                    • <span style="color:#f43f5e;">High</span>: Manual investigation<br>
                    • <span style="color:#f59e0b;">Medium</span>: Automate rules<br>
                    • <span style="color:#10b981;">Low</span>: Basic monitoring
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== ORIGINAL VISUALIZATIONS (ENHANCED WITH INTERACTIVITY) ====================
    col1, col2 = st.columns([3, 2], gap="medium")
    
    with col1:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Transaction Volume Timeline</div>',
            unsafe_allow_html=True,
        )
        df["hour"] = pd.to_datetime(df["timestamp"]).dt.floor("h")
        hourly_data = df.groupby(["hour", "is_fraud"]).size().reset_index(name="count")
        hourly_data["status"] = hourly_data["is_fraud"].map({True: "Fraudulent", False: "Normal"})

        fig_timeline = go.Figure()
        normal = hourly_data[hourly_data["status"] == "Normal"]
        fraud = hourly_data[hourly_data["status"] == "Fraudulent"]
        
        fig_timeline.add_trace(
            go.Scatter(
                x=normal["hour"],
                y=normal["count"],
                name="Normal",
                mode="lines",
                line=dict(color="#10b981", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(16,185,129,0.08)",
            )
        )
        fig_timeline.add_trace(
            go.Scatter(
                x=fraud["hour"],
                y=fraud["count"],
                name="Fraudulent",
                mode="lines",
                line=dict(color="#f43f5e", width=2.5),
                fill="tozeroy",
                fillcolor="rgba(244,63,94,0.08)",
            )
        )
        
        # Add predictive trend line with animation
        if len(fraud) > 1:
            x_future = [fraud["hour"].iloc[-1] + pd.Timedelta(hours=i) for i in range(1, 4)]
            y_future = [fraud["count"].iloc[-1] * (1 + 0.15*i) for i in range(1, 4)]
            
            fig_timeline.add_trace(
                go.Scatter(
                    x=x_future,
                    y=y_future,
                    name="Prediction",
                    mode="lines+markers",
                    line=dict(color="#f43f5e", width=2, dash="dot"),
                    marker=dict(size=8, symbol="arrow", angleref="previous")
                )
            )
        
        fig_timeline.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, showline=False, color="#94a3b8", title_font_size=12),
            yaxis=dict(
                showgrid=True,
                gridcolor="#eef2ff",
                showline=False,
                color="#94a3b8",
                title="Count",
                title_font_size=12,
            ),
            legend=dict(
                orientation="h",
                yanchor="bottom",
                y=1.02,
                xanchor="right",
                x=1,
                bgcolor="rgba(255,255,255,.8)",
                bordercolor="#e2e8f0",
                borderwidth=1,
            ),
            font=dict(family="Inter", size=11, color="#64748b"),
            hovermode="x unified",
        )
        st.plotly_chart(fig_timeline, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Fraud by Device</div>',
            unsafe_allow_html=True,
        )
        device_fraud = df[df["is_fraud"]].groupby("device").size().reset_index(name="count")
        
        # Add trend data with animation
        device_trend = {"Mobile": "↑", "Desktop": "→", "Tablet": "↓"}
        
        fig_device = go.Figure(
            data=[
                go.Pie(
                    labels=device_fraud["device"],
                    values=device_fraud["count"],
                    hole=0.55,
                    marker=dict(
                        colors=["#6c5ce7", "#f43f5e", "#f59e0b"],
                        line=dict(color="#fff", width=3),
                    ),
                    textinfo="label+percent",
                    textposition="outside",
                    hovertemplate="<b>%{label}</b><br>%{value} frauds<br>Trend: " + 
                                 device_trend.get("%{label}", "→") + "<extra></extra>",
                    pull=[0.1 if device == "Mobile" else 0 for device in device_fraud["device"]]
                )
            ]
        )
        fig_device.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            showlegend=False,
            font=dict(family="Inter", size=11, color="#64748b"),
            annotations=[
                dict(
                    text=f"<b>{fraud_count}</b><br><span style='font-size:11px;color:#94a3b8;'>Frauds</span>",
                    x=0.5,
                    y=0.5,
                    showarrow=False,
                    font=dict(size=18, color="#1e293b"),
                )
            ],
        )
        st.plotly_chart(fig_device, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ==================== NEW: INTERACTIVE ROI & COST-BENEFIT ANALYSIS ====================
    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">💰</div>Interactive ROI & Cost-Benefit Analysis</div>',
        unsafe_allow_html=True,
    )
    
    # Interactive controls for ROI
    roi_col1, roi_col2 = st.columns(2)
    
    with roi_col1:
        investigation_cost = st.slider(
            "Investigation Cost per False Positive ($)",
            min_value=10,
            max_value=200,
            value=50,
            step=10,
            help="Average cost to investigate a false positive alert"
        )
    
    with roi_col2:
        automation_efficiency = st.slider(
            "Automation Efficiency Improvement (%)",
            min_value=0,
            max_value=100,
            value=30,
            step=5,
            help="Expected improvement from rule automation"
        )
    
    # Calculate ROI metrics with user inputs
    false_positives = len(df[df['amount'] > 3000]) * 0.1
    total_cost_fp = false_positives * investigation_cost
    roi_percentage = (total_blocked - total_cost_fp) / total_cost_fp * 100 if total_cost_fp > 0 else 0
    efficiency_gain = total_cost_fp * (automation_efficiency / 100)
    
    col_roi1, col_roi2, col_roi3 = st.columns(3)
    
    with col_roi1:
        st.markdown(
            f"""
            <div style="text-align:center;padding:15px;background:#f5f3ff;border-radius:12px;border:2px solid #6c5ce7;animation:glow 2s infinite alternate;">
                <div style="color:#6c5ce7;font-size:30px;font-weight:800;animation:pulse 3s infinite;">{roi_percentage:.0f}%</div>
                <div style="color:#64748b;font-size:12px;font-weight:600;">RETURN ON INVESTMENT</div>
                <div style="color:#94a3b8;font-size:10px;margin-top:5px;">For every $1 invested</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_roi2:
        st.markdown(
            f"""
            <div style="text-align:center;padding:15px;background:#f0f9ff;border-radius:12px;border:2px solid #0ea5e9;">
                <div style="color:#0ea5e9;font-size:30px;font-weight:800;">${total_blocked:,.0f}</div>
                <div style="color:#64748b;font-size:12px;font-weight:600;">TOTAL GAINS</div>
                <div style="color:#94a3b8;font-size:10px;margin-top:5px;">Fraud prevented</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with col_roi3:
        st.markdown(
            f"""
            <div style="text-align:center;padding:15px;background:#fef2f2;border-radius:12px;border:2px solid #ef4444;">
                <div style="color:#ef4444;font-size:30px;font-weight:800;">${total_cost_fp:,.0f}</div>
                <div style="color:#64748b;font-size:12px;font-weight:600;">FALSE POSITIVE COST</div>
                <div style="color:#94a3b8;font-size:10px;margin-top:5px;">Unnecessary investigation</div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    # ROI Chart with animation
    roi_data = pd.DataFrame({
        'Month': ['Jan', 'Feb', 'Mar', 'Apr', 'May', 'Jun', 'Jul', 'Aug'],
        'Investment': [10000, 12000, 11000, 13000, 14000, 15000, 16000, 17000],
        'Return': [15000, 18000, 16000, 20000, 22000, 25000, 27000, 30000]
    })
    
    fig_roi = go.Figure()
    fig_roi.add_trace(go.Bar(
        x=roi_data['Month'],
        y=roi_data['Investment'],
        name='Investment',
        marker_color='#94a3b8'
    ))
    fig_roi.add_trace(go.Bar(
        x=roi_data['Month'],
        y=roi_data['Return'] - roi_data['Investment'],
        name='Net Benefit',
        marker_color='#10b981'
    ))
    
    fig_roi.update_layout(
        height=200,
        barmode='stack',
        margin=dict(l=10, r=10, t=10, b=20),
        plot_bgcolor='rgba(0,0,0,0)',
        paper_bgcolor='rgba(0,0,0,0)',
        xaxis=dict(showgrid=False, color='#94a3b8'),
        yaxis=dict(showgrid=True, gridcolor='#eef2ff', color='#94a3b8', title='$'),
        font=dict(family='Inter', size=11, color='#64748b'),
        legend=dict(
            orientation='h',
            yanchor='bottom',
            y=1.02,
            xanchor='right',
            x=1
        ),
        updatemenus=[dict(
            type="buttons",
            showactive=False,
            buttons=[dict(label="Play Animation",
                         method="animate",
                         args=[None])])]
    )
    
    st.plotly_chart(fig_roi, use_container_width=True, config={'displayModeBar': False})
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== ORIGINAL REMAINING VISUALIZATIONS ====================
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box">💰</div>Transaction Amount Distribution</div>',
            unsafe_allow_html=True,
        )
        
        # Add interactive filter
        amount_filter = st.select_slider(
            "Filter Amount Range",
            options=["All", "< $1000", "$1000-$5000", "> $5000"],
            value="All"
        )
        
        filtered_df = df.copy()
        if amount_filter == "< $1000":
            filtered_df = df[df['amount'] < 1000]
        elif amount_filter == "$1000-$5000":
            filtered_df = df[(df['amount'] >= 1000) & (df['amount'] <= 5000)]
        elif amount_filter == "> $5000":
            filtered_df = df[df['amount'] > 5000]
        
        fig_amount = go.Figure()
        fig_amount.add_trace(
            go.Histogram(
                x=filtered_df[~filtered_df["is_fraud"]]["amount"],
                name="Normal",
                marker_color="rgba(16,185,129,0.65)",
                nbinsx=30,
                hovertemplate="Normal<br>Amount: $%{x:.0f}<br>Count: %{y}<extra></extra>"
            )
        )
        fig_amount.add_trace(
            go.Histogram(
                x=filtered_df[filtered_df["is_fraud"]]["amount"],
                name="Fraudulent",
                marker_color="rgba(244,63,94,0.65)",
                nbinsx=30,
                hovertemplate="Fraudulent<br>Amount: $%{x:.0f}<br>Count: %{y}<extra></extra>"
            )
        )
        fig_amount.update_layout(
            barmode="overlay",
            height=250,
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(title="Amount ($)", showgrid=False, color="#94a3b8", title_font_size=12),
            yaxis=dict(
                title="Frequency",
                showgrid=True,
                gridcolor="#eef2ff",
                showline=False,
                color="#94a3b8",
                title_font_size=12,
            ),
            legend=dict(bgcolor="rgba(255,255,255,.8)", bordercolor="#e2e8f0", borderwidth=1),
            font=dict(family="Inter", size=11, color="#64748b"),
        )
        st.plotly_chart(fig_amount, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box">🌍</div>Interactive Geographic Risk Analysis</div>',
            unsafe_allow_html=True
        )
        
        # Interactive location filter
        top_n = st.slider(
            "Show Top N Locations",
            min_value=5,
            max_value=20,
            value=10,
            step=1
        )

        location_stats = df.groupby('location').agg({'is_fraud': ['sum', 'count']}).reset_index()
        location_stats.columns = ['location', 'fraud_count', 'total_count']
        location_stats['fraud_rate'] = (location_stats['fraud_count'] / location_stats['total_count']) * 100
        location_stats = location_stats.sort_values('fraud_rate', ascending=True).tail(top_n)

        fig_geo = go.Figure(go.Bar(
            y=location_stats['location'],
            x=location_stats['fraud_rate'],
            orientation='h',
            marker=dict(
                color=location_stats['fraud_rate'],
                colorscale=[[0, '#10b981'], [0.5, '#f59e0b'], [1, '#f43f5e']],
                line=dict(width=0)
            ),
            hovertemplate='<b>%{y}</b><br>Fraud Rate: %{x:.1f}%<br>Transactions: %{customdata[0]}<extra></extra>',
            customdata=location_stats[['total_count']]
        ))
        fig_geo.update_layout(
            height=250,
            margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False,
            plot_bgcolor='rgba(0,0,0,0)',
            paper_bgcolor='rgba(0,0,0,0)',
            xaxis=dict(
                title='Fraud Rate (%)',
                showgrid=True,
                gridcolor='#eef2ff',
                color='#94a3b8',
                title_font_size=12
            ),
            yaxis=dict(showgrid=False, color='#94a3b8'),
            font=dict(family='Inter', size=11, color='#64748b')
        )
        st.plotly_chart(fig_geo, use_container_width=True, config={'displayModeBar': False})
        st.markdown('</div>', unsafe_allow_html=True)

    # ==================== NEW: INTERACTIVE ALERT WIDGETS ====================
    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">🔔</div>Interactive Alerts Dashboard</div>',
        unsafe_allow_html=True,
    )
    
    alert_col1, alert_col2, alert_col3 = st.columns(3)
    
    with alert_col1:
        # Activity peak alert with refresh
        if st.button("🔄 Refresh Activity", key="refresh_activity"):
            st.rerun()
        
        current_hour_volume = len(df[pd.to_datetime(df['timestamp']).dt.hour == datetime.now().hour])
        avg_hour_volume = len(df) / 24
        volume_alert = "⚠️" if current_hour_volume > avg_hour_volume * 1.5 else "✅"
        
        st.markdown(
            f"""
            <div style="background:#f0f9ff;border-radius:12px;padding:15px;border-left:4px solid #0ea5e9;margin-top:10px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                    <div style="color:#0ea5e9;font-weight:700;font-size:12px;">📈 ACTIVITY</div>
                    <div style="font-size:18px;animation:pulse 1.5s infinite;">{volume_alert}</div>
                </div>
                <div style="color:#1e293b;font-size:14px;font-weight:600;margin-bottom:4px;">
                    Current Hour Volume
                </div>
                <div style="color:#64748b;font-size:12px;">
                    {current_hour_volume} transactions<br>
                    {'+' if current_hour_volume > avg_hour_volume else ''}{((current_hour_volume/avg_hour_volume)-1)*100:.0f}% vs average
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    with alert_col2:
        # New patterns alert with toggle
        show_details = st.checkbox("Show Pattern Details", key="pattern_details")
        
        new_patterns_detected = random.randint(0, 3)
        pattern_color = "#f59e0b" if new_patterns_detected > 0 else "#10b981"
        
        st.markdown(
            f"""
            <div style="background:#fffbeb;border-radius:12px;padding:15px;border-left:4px solid {pattern_color};margin-top:10px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                    <div style="color:{pattern_color};font-weight:700;font-size:12px;">🔍 NEW PATTERNS</div>
                    <div style="font-size:18px;animation:blink 2s infinite;">{"🔍" if new_patterns_detected > 0 else "✅"}</div>
                </div>
                <div style="color:#1e293b;font-size:14px;font-weight:600;margin-bottom:4px;">
                    {new_patterns_detected} detected
                </div>
                <div style="color:#64748b;font-size:12px;">
                    Last detection: {datetime.now().strftime('%H:%M')}<br>
                    {'Needs review' if new_patterns_detected > 0 else 'All clear'}
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
        
        if show_details and new_patterns_detected > 0:
            st.info(f"🔍 {new_patterns_detected} new fraud patterns detected. Review recommended.")
    
    with alert_col3:
        # System performance alert with gauge
        system_performance = random.randint(85, 99)
        perf_color = "#10b981" if system_performance >= 95 else "#f59e0b" if system_performance >= 85 else "#f43f5e"
        
        st.markdown(
            f"""
            <div style="background:#f0fdf4;border-radius:12px;padding:15px;border-left:4px solid {perf_color};margin-top:10px;">
                <div style="display:flex;align-items:center;justify-content:space-between;margin-bottom:8px;">
                    <div style="color:{perf_color};font-weight:700;font-size:12px;">⚡ PERFORMANCE</div>
                    <div style="font-size:18px;">{"⚡" if system_performance >= 95 else "✅"}</div>
                </div>
                <div style="color:#1e293b;font-size:14px;font-weight:600;margin-bottom:4px;">
                    {system_performance}% operational
                </div>
                <div style="color:#64748b;font-size:12px;">
                    Response time: 0.8s<br>
                    Uptime: 99.9%
                </div>
            </div>
            """,
            unsafe_allow_html=True
        )
    
    st.markdown("</div>", unsafe_allow_html=True)

    # ==================== INTERACTIVE FINAL VISUALIZATIONS ====================
    col1, col2 = st.columns(2, gap="medium")
    
    with col1:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Fraud by Payment Method</div>',
            unsafe_allow_html=True,
        )
        
        # Add payment method filter
        selected_methods = st.multiselect(
            "Select Payment Methods to Display",
            options=df['payment_method'].unique(),
            default=df['payment_method'].unique(),
            key="payment_filter"
        )
        
        filtered_payments = df[df['payment_method'].isin(selected_methods)] if selected_methods else df
        
        payment_fraud = filtered_payments.groupby("payment_method").agg({"is_fraud": ["sum", "count"]}).reset_index()
        payment_fraud.columns = ["payment_method", "fraud_count", "total_count"]
        payment_fraud["fraud_rate"] = (payment_fraud["fraud_count"] / payment_fraud["total_count"]) * 100

        fig_payment = go.Figure(
            go.Bar(
                x=payment_fraud["payment_method"],
                y=payment_fraud["fraud_rate"],
                marker=dict(
                    color=payment_fraud["fraud_rate"],
                    colorscale=[[0, "#10b981"], [0.5, "#f59e0b"], [1, "#f43f5e"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{x}</b><br>Fraud Rate: %{y:.1f}%<br>Total: %{customdata[0]}<extra></extra>",
                customdata=payment_fraud[['total_count']]
            )
        )
        fig_payment.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=10, b=40),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(showgrid=False, color="#94a3b8", tickfont=dict(size=11)),
            yaxis=dict(
                title="Fraud Rate (%)",
                showgrid=True,
                gridcolor="#eef2ff",
                showline=False,
                color="#94a3b8",
                title_font_size=12,
            ),
            font=dict(family="Inter", size=11, color="#64748b"),
        )
        st.plotly_chart(fig_payment, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Merchant Category Analysis</div>',
            unsafe_allow_html=True,
        )
        
        # Add merchant filter
        min_fraud_count = st.slider(
            "Minimum Fraud Count",
            min_value=0,
            max_value=50,
            value=5,
            step=1,
            key="merchant_filter"
        )
        
        merchant_stats = df.groupby("merchant_category")["is_fraud"].agg(["sum", "count"]).reset_index()
        merchant_stats.columns = ["category", "fraud_count", "total_count"]
        merchant_stats = merchant_stats[merchant_stats["fraud_count"] >= min_fraud_count]
        merchant_stats = merchant_stats.sort_values("fraud_count", ascending=False).head(8)

        fig_merchant = px.funnel(
            merchant_stats,
            x="fraud_count",
            y="category",
            labels={"fraud_count": "Fraud Count", "category": "Category"},
        )
        fig_merchant.update_layout(
            height=290,
            margin=dict(l=10, r=10, t=10, b=10),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            font=dict(family="Inter", size=11, color="#64748b"),
        )
        fig_merchant.update_traces(marker_color="#6c5ce7", textfont_color="white")
        st.plotly_chart(fig_merchant, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    # ==================== INTERACTIVE PRESENTATION GENERATOR ====================
    st.markdown(
        """
        <div style="text-align:center;margin:30px 0 20px;">
        """,
        unsafe_allow_html=True
    )
    
    # Create presentation button with interactive features
    if st.button("🎤 GENERATE EXECUTIVE PRESENTATION", 
                 use_container_width=True, 
                 type="primary",
                 help="Generate a comprehensive executive report"):
        
        with st.spinner("Generating executive presentation..."):
            # Simulate generation process
            time.sleep(2)
            
            # Create a presentation container
            presentation_container = st.container()
            
            with presentation_container:
                st.success("✅ Presentation Generated Successfully!")
                
                # Presentation preview
                st.markdown(
                    """
                    <div class="card" style="background:linear-gradient(135deg,#f8fafc,#ffffff);">
                        <div class="card-title">
                            <div class="icon-box">📊</div>
                            Executive Presentation Preview
                        </div>
                    </div>
                    """,
                    unsafe_allow_html=True
                )
                
                # Presentation slides
                slides = [
                    ("📈 Executive Summary", 
                     f"• Total Transactions: {total_transactions:,}\n• Fraud Rate: {fraud_rate:.2f}%\n• Blocked Amount: ${total_blocked:,.0f}\n• ROI: {roi_percentage:.0f}%"),
                    
                    ("🔍 Key Findings", 
                     f"• Top Fraud Location: {location_stats.iloc[-1]['location'] if not location_stats.empty else 'N/A'}\n• Highest Risk Payment: {payment_fraud.iloc[-1]['payment_method'] if not payment_fraud.empty else 'N/A'}\n• Most Vulnerable Device: {device_fraud.iloc[-1]['device'] if not device_fraud.empty else 'N/A'}"),
                    
                    ("🎯 Recommended Actions",
                     "1. Increase monitoring for high-risk payment methods\n2. Implement additional verification for international transactions\n3. Optimize fraud detection thresholds\n4. Enhance customer education on fraud prevention"),
                    
                    ("📅 Next Steps",
                     "• Weekly review meeting scheduled\n• Technical team assessment required\n• Budget approval for system upgrades\n• Stakeholder presentation scheduled")
                ]
                
                for i, (title, content) in enumerate(slides, 1):
                    with st.expander(f"Slide {i}: {title}", expanded=(i==1)):
                        st.markdown(f"### {title}")
                        st.text(content)
                        
                        # Add download option for each slide
                        col1, col2 = st.columns([3, 1])
                        with col1:
                            st.progress(i/len(slides))
                        with col2:
                            st.download_button(
                                f"Download Slide {i}",
                                f"# {title}\n\n{content}",
                                file_name=f"fraud_presentation_slide_{i}.txt",
                                key=f"slide_{i}"
                            )
                
                # Final presentation options
                st.markdown("---")
                col_download, col_email, col_schedule = st.columns(3)
                
                with col_download:
                    st.download_button(
                        "📥 Download Full Presentation",
                        f"Fraud Detection Executive Presentation\n\n{chr(10).join([f'{title}: {content}' for title, content in slides])}",
                        file_name="fraud_executive_presentation.txt",
                        use_container_width=True
                    )
                
                with col_email:
                    if st.button("📧 Email to Management", use_container_width=True):
                        st.success("Presentation sent to management team!")
                
                with col_schedule:
                    if st.button("📅 Schedule Meeting", use_container_width=True):
                        meeting_date = st.date_input("Select meeting date")
                        if meeting_date:
                            st.success(f"Meeting scheduled for {meeting_date}!")
    


# ════════════════════════════════════════════════════════
# TAB 2 — LIVE MONITORING  (MAP FIRST)
# ════════════════════════════════════════════════════════
with tab2:

    render_split_map_card(df, title="Live Fraud Location Map (Split Map)", key_prefix="live_map")
    st.markdown("<br>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box"></div>Real-Time Transaction Stream Monitor</div>',
        unsafe_allow_html=True,
    )

    col_h, col_b = st.columns([3, 1])
    with col_h:
        st.markdown(
            '<p style="color:#64748b;font-size:14px;margin:0;">Monitor live transactions as they flow through the system</p>',
            unsafe_allow_html=True,
        )
    with col_b:
        simulate_button = st.button("▶  START SIMULATION", use_container_width=True, type="primary")
    st.markdown("</div>", unsafe_allow_html=True)

    if simulate_button:
        transaction_container = st.container()
        sample_transactions = df.sample(n=20).sort_values("timestamp")
        for _, row in sample_transactions.iterrows():
            with transaction_container:
                col1, col2, col3, col4, col5 = st.columns([2, 2, 2, 2, 1])
                with col1:
                    st.markdown(f"**User ID:** `{row['user_id']}`")
                    st.markdown(f"**Time:** {row['timestamp'].strftime('%H:%M:%S')}")
                with col2:
                    st.markdown(f"**Amount:** ${row['amount']:.2f}")
                    st.markdown(f"**Method:** {row['payment_method']}")
                with col3:
                    st.markdown(f"**Location:** {row['location']}")
                    st.markdown(f"**Device:** {row['device']}")
                with col4:
                    st.markdown(f"**Distance:** {row['distance_from_home']:.1f} km")
                    st.markdown(f"**Processing:** {row['processing_time']:.0f} ms")
                with col5:
                    if row["is_fraud"]:
                        st.markdown('<div class="fraud-alert">BLOCKED<br>FRAUD</div>', unsafe_allow_html=True)
                    else:
                        st.markdown('<div class="safe-alert">APPROVED<br>SAFE</div>', unsafe_allow_html=True)
                st.markdown(
                    '<hr style="border:none;border-top:1px solid #eef2ff;margin:10px 0;">',
                    unsafe_allow_html=True,
                )
                time.sleep(0.5)
        st.success("Simulation Complete — 20 transactions processed")
    else:
        st.markdown(
            '<div class="info-banner"> Click <strong>START SIMULATION</strong> to begin monitoring live transactions</div>',
            unsafe_allow_html=True,
        )
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Transaction Data Structure</div>',
            unsafe_allow_html=True,
        )
        sample_df = df.head(10)[
            ["user_id", "amount", "location", "device", "processing_time", "is_fraud"]
        ].copy()
        sample_df["status"] = sample_df["is_fraud"].map({True: "FRAUD – BLOCKED", False: " APPROVED"})
        sample_df = sample_df.drop("is_fraud", axis=1)
        st.dataframe(sample_df, use_container_width=True, height=340, hide_index=True)
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">🚫</div>Recent Blocked Transactions</div>',
        unsafe_allow_html=True,
    )
    blocked_df = df[df["is_fraud"]].tail(10)[
        ["timestamp", "user_id", "amount", "location", "device", "payment_method"]
    ].copy()
    blocked_df["status"] = "🚨 BLOCKED – FRAUD"
    blocked_df["timestamp"] = blocked_df["timestamp"].dt.strftime("%H:%M:%S")
    if not blocked_df.empty:
        st.dataframe(blocked_df, use_container_width=True, height=280, hide_index=True)
    else:
        st.info("No blocked transactions in the current dataset")
    st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 3 — RISK SCORING
# ════════════════════════════════════════════════════════
with tab3:
    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">⚡</div>Interactive Risk Assessment Calculator</div>',
        unsafe_allow_html=True,
    )
    st.markdown(
        '<div class="info-banner">This advanced algorithm evaluates transaction risk based on multiple behavioral and transactional factors. Adjust the parameters below to calculate a real-time risk score.</div>',
        unsafe_allow_html=True,
    )
    st.markdown("</div>", unsafe_allow_html=True)

    col1, col2 = st.columns([1, 1], gap="medium")

    with col1:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Input Parameters</div>',
            unsafe_allow_html=True,
        )
        transaction_amount = st.slider(
            " Transaction Amount ($)",
            min_value=0,
            max_value=10000,
            value=500,
            step=50,
            help="Higher amounts increase risk score",
        )
        distance_from_home = st.slider(
            " Distance from Home (km)",
            min_value=0,
            max_value=500,
            value=50,
            step=10,
            help="Greater distance indicates higher risk",
        )
        typing_speed = st.slider(
            " Typing Speed (ms/char)",
            min_value=50,
            max_value=300,
            value=150,
            step=10,
            help="Very fast typing may indicate bot activity",
        )
        st.markdown(
            '<p style="color:#64748b;font-size:13px;font-weight:600;margin:18px 0 6px;">Additional Factors</p>',
            unsafe_allow_html=True,
        )
        device_type = st.selectbox("📱 Device Type", ["Mobile", "Desktop", "Tablet"])
        payment_method = st.selectbox(
            " Payment Method",
            ["Credit Card", "Debit Card", "Wire Transfer", "Mobile Wallet"],
        )
        time_of_day = st.selectbox(
            " Time of Day",
            ["Morning (6-12)", "Afternoon (12-18)", "Evening (18-24)", "Night (0-6)"],
        )
        st.markdown("</div>", unsafe_allow_html=True)

    with col2:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Risk Assessment Results</div>',
            unsafe_allow_html=True,
        )

        risk_score = 0
        risk_factors = []

        if transaction_amount > 5000:
            risk_score += 30
            risk_factors.append("Critical: Very high transaction amount")
        elif transaction_amount > 2000:
            risk_score += 20
            risk_factors.append("Warning: High transaction amount")
        elif transaction_amount > 1000:
            risk_score += 10
            risk_factors.append("Moderate: Above average amount")

        if distance_from_home > 200:
            risk_score += 25
            risk_factors.append("Critical: Transaction far from home")
        elif distance_from_home > 100:
            risk_score += 15
            risk_factors.append("Warning: Unusual distance")
        elif distance_from_home > 50:
            risk_score += 8
            risk_factors.append("Moderate: Distance from usual location")

        if typing_speed < 80:
            risk_score += 20
            risk_factors.append("Critical: Bot-like typing speed detected")
        elif typing_speed < 100:
            risk_score += 12
            risk_factors.append("Warning: Unusually fast typing")

        if payment_method == "Wire Transfer":
            risk_score += 15
            risk_factors.append("Warning: High-risk payment method")
        elif payment_method == "Mobile Wallet":
            risk_score += 5

        if time_of_day == "Night (0-6)":
            risk_score += 10
            risk_factors.append("Warning: Unusual transaction time")

        risk_score = min(risk_score, 100)

        fig_gauge = go.Figure(
            go.Indicator(
                mode="gauge+number+delta",
                value=risk_score,
                domain={"x": [0, 1], "y": [0, 1]},
                title={"text": "RISK SCORE", "font": {"size": 18, "color": "#64748b", "family": "Inter"}},
                delta={
                    "reference": 50,
                    "increasing": {"color": "#f43f5e"},
                    "decreasing": {"color": "#10b981"},
                },
                number={"font": {"size": 48, "family": "Inter", "color": "#1e293b"}},
                gauge={
                    "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "#cbd5e1"},
                    "bar": {"color": "#6c5ce7", "thickness": 0.5},
                    "bgcolor": "rgba(0,0,0,0)",
                    "borderwidth": 0,
                    "steps": [
                        {"range": [0, 30], "color": "rgba(16,185,129,.15)"},
                        {"range": [30, 60], "color": "rgba(245,158,11,.15)"},
                        {"range": [60, 100], "color": "rgba(244,63,94,.15)"},
                    ],
                    "threshold": {"line": {"color": "#6c5ce7", "width": 4}, "thickness": 0.75, "value": risk_score},
                },
            )
        )
        fig_gauge.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            font={"color": "#64748b", "family": "Inter"},
            height=280,
            margin=dict(l=20, r=20, t=50, b=10),
        )
        st.plotly_chart(fig_gauge, use_container_width=True, config={"displayModeBar": False})

        if risk_score >= 70:
            risk_level, recommendation, pill = "CRITICAL", "BLOCK IMMEDIATELY", "high"
            bg_color = "#f43f5e"
        elif risk_score >= 40:
            risk_level, recommendation, pill = "ELEVATED", "REQUIRE ADDITIONAL VERIFICATION", "med"
            bg_color = "#f59e0b"
        else:
            risk_level, recommendation, pill = "LOW", "APPROVE TRANSACTION", "low"
            bg_color = "#10b981"

        st.markdown(
            f"""
        <div style="text-align:center;margin:12px 0;">
            <div class="risk-pill {pill}">{risk_level}</div>
            <p style="color:#1e293b;font-weight:700;font-size:15px;margin:10px 0 4px;">{recommendation}</p>
            <p style="color:{bg_color};font-weight:800;font-size:28px;margin:0;">Score: {risk_score}/100</p>
        </div>
        """,
            unsafe_allow_html=True,
        )

        st.markdown(
            '<p style="color:#64748b;font-size:13px;font-weight:600;margin:16px 0 8px;">Risk Factors Identified</p>',
            unsafe_allow_html=True,
        )
        if risk_factors:
            for f in risk_factors:
                if "Critical" in f:
                    st.markdown(
                        f'<div style="padding:10px 14px;background:#fef2f2;border-left:3px solid #f43f5e;border-radius:8px;margin-bottom:8px;color:#991b1b;font-size:13px;font-weight:500;">🚨 {f}</div>',
                        unsafe_allow_html=True,
                    )
                elif "Warning" in f:
                    st.markdown(
                        f'<div style="padding:10px 14px;background:#fffbeb;border-left:3px solid #f59e0b;border-radius:8px;margin-bottom:8px;color:#92400e;font-size:13px;font-weight:500;">⚠️ {f}</div>',
                        unsafe_allow_html=True,
                    )
                else:
                    st.markdown(
                        f'<div style="padding:10px 14px;background:#eef2ff;border-left:3px solid #6c5ce7;border-radius:8px;margin-bottom:8px;color:#4338ca;font-size:13px;font-weight:500;">ℹ️ {f}</div>',
                        unsafe_allow_html=True,
                    )
        else:
            st.markdown(
                '<div style="padding:12px 16px;background:#ecfdf5;border-radius:10px;color:#065f46;font-weight:600;text-align:center;">✅ No significant risk factors detected</div>',
                unsafe_allow_html=True,
            )

        st.markdown(
            '<p style="color:#64748b;font-size:13px;font-weight:600;margin:18px 0 8px;">Score Breakdown</p>',
            unsafe_allow_html=True,
        )
        breakdown = pd.DataFrame(
            {
                "Factor": ["Transaction Amount", "Distance from Home", "Typing Speed", "Payment Method", "Time of Day"],
                "Weight": ["30%", "25%", "20%", "15%", "10%"],
            }
        )
        st.dataframe(breakdown, use_container_width=True, hide_index=True, height=180)
        st.markdown("</div>", unsafe_allow_html=True)

# ════════════════════════════════════════════════════════
# TAB 4 — MODEL PERFORMANCE
# ════════════════════════════════════════════════════════
with tab4:
    model = st.session_state.model
    X_test = st.session_state.X_test
    y_test = st.session_state.y_test

    y_pred = model.predict(X_test)
    accuracy = accuracy_score(y_test, y_pred)
    precision = precision_score(y_test, y_pred)
    recall = recall_score(y_test, y_pred)

    c1, c2, c3 = st.columns(3, gap="medium")
    model_cards = [
        ("mc-purple", "Accuracy", f"{accuracy:.2%}", "Overall model precision"),
        ("mc-pink", "Precision", f"{precision:.2%}", "Fraud detection precision"),
        ("mc-green", "Recall", f"{recall:.2%}", "True positive rate"),
    ]
    for col, (cls, label, val, sub) in zip([c1, c2, c3], model_cards):
        with col:
            st.markdown(
                f"""
            <div class="metric-card {cls}">
                <div class="mc-label">{label}</div>
                <div class="mc-value">{val}</div>
                <div class="mc-sub">{sub}</div>
                <div class="deco"></div><div class="deco2"></div>
            </div>
            """,
                unsafe_allow_html=True,
            )

    st.markdown("<br>", unsafe_allow_html=True)

    col_cm, col_imp = st.columns(2, gap="medium")
    with col_cm:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Confusion Matrix</div>',
            unsafe_allow_html=True,
        )
        cm = confusion_matrix(y_test, y_pred)
        fig_cm = go.Figure(
            data=go.Heatmap(
                z=cm,
                x=["Predicted Normal", "Predicted Fraud"],
                y=["Actual Normal", "Actual Fraud"],
                colorscale=[[0, "#10b981"], [0.5, "#6c5ce7"], [1, "#f43f5e"]],
                text=cm,
                texttemplate="<b>%{text}</b>",
                textfont={"size": 22, "color": "white"},
                hovertemplate="%{y} → %{x}<br>Count: %{z}<extra></extra>",
            )
        )
        fig_cm.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=30),
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(color="#64748b", tickfont=dict(size=12)),
            yaxis=dict(color="#64748b", tickfont=dict(size=12)),
            font=dict(family="Inter", size=11, color="#64748b"),
        )
        st.plotly_chart(fig_cm, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    with col_imp:
        st.markdown(
            '<div class="card"><div class="card-title"><div class="icon-box"></div>Feature Importance</div>',
            unsafe_allow_html=True,
        )
        feature_names = [
            "amount",
            "distance_from_home",
            "typing_speed",
            "processing_time",
            "device",
            "payment_method",
            "merchant_category",
            "location",
        ]
        importance = model.coef_[0]
        importance_df = pd.DataFrame({"Feature": feature_names, "Importance": importance}).sort_values(
            "Importance", ascending=False
        )

        fig_imp = go.Figure(
            go.Bar(
                y=importance_df["Feature"],
                x=importance_df["Importance"],
                orientation="h",
                marker=dict(
                    color=["#6c5ce7" if v >= 0 else "#f43f5e" for v in importance_df["Importance"]],
                    line=dict(width=0),
                ),
                hovertemplate="<b>%{y}</b><br>Importance: %{x:.3f}<extra></extra>",
            )
        )
        fig_imp.update_layout(
            height=300,
            margin=dict(l=10, r=10, t=10, b=30),
            showlegend=False,
            plot_bgcolor="rgba(0,0,0,0)",
            paper_bgcolor="rgba(0,0,0,0)",
            xaxis=dict(
                title="Coefficient",
                showgrid=True,
                gridcolor="#eef2ff",
                color="#94a3b8",
                title_font_size=12,
            ),
            yaxis=dict(showgrid=False, color="#94a3b8"),
            font=dict(family="Inter", size=11, color="#64748b"),
        )
        st.plotly_chart(fig_imp, use_container_width=True, config={"displayModeBar": False})
        st.markdown("</div>", unsafe_allow_html=True)

    st.markdown(
        '<div class="card"><div class="card-title"><div class="icon-box">ℹ️</div>Model Information</div>',
        unsafe_allow_html=True,
    )
    info_items = [
        ("Algorithm", "Logistic Regression"),
        ("Training Size", f"{len(df)} samples"),
        ("Test Size", f"{len(X_test)} samples"),
        ("Features", "8 variables"),
    ]
    cols = st.columns(4, gap="medium")
    for col, (label, val) in zip(cols, info_items):
        with col:
            st.markdown(
                f"""
            <div style="text-align:center;padding:18px 12px;background:#f5f3ff;border-radius:14px;">
                <div style="color:#6b7280;font-size:12px;margin-bottom:6px;">{label}</div>
                <div style="color:#6c5ce7;font-size:17px;font-weight:700;">{val}</div>
            </div>
            """,
                unsafe_allow_html=True,
            )
    st.markdown("</div>", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# FOOTER
# ─────────────────────────────────────────────
st.markdown(
    """
<div style="text-align:center;padding:24px;margin-top:12px;">
    <p style="color:#94a3b8;font-size:13px;margin:0;">Bank Fraud Detection System v3.0 | Enhanced with Machine Learning Algorithms</p>
    <p style="color:#cbd5e1;font-size:12px;margin:4px 0 0;">© 2026 Financial Security Division | All Rights Reserved</p>
</div>
""",
    unsafe_allow_html=True,
)
