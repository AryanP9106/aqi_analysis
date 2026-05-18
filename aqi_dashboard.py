"""
AQI Analysis & Prediction Dashboard
=====================================
Run with: streamlit run aqi_dashboard.py

Requirements:
    pip install streamlit plotly pandas numpy scikit-learn
"""

import pickle
import numpy as np
import pandas as pd
import streamlit as st
import plotly.express as px
import plotly.graph_objects as go
from sklearn.linear_model import LinearRegression
from sklearn.model_selection import train_test_split
from sklearn.metrics import r2_score, mean_squared_error

# ─────────────────────────────────────────────
# PAGE CONFIG
# ─────────────────────────────────────────────
st.set_page_config(
    page_title="India AQI Dashboard",
    page_icon="🌫️",
    layout="wide",
    initial_sidebar_state="expanded"
)

# ─────────────────────────────────────────────
# CUSTOM CSS
# ─────────────────────────────────────────────
st.markdown("""
<style>
    .main { background-color: #0e1117; }
    .metric-card {
        background: linear-gradient(135deg, #1e2130, #2d3250);
        border-radius: 12px;
        padding: 16px 20px;
        border-left: 4px solid #4fc3f7;
        margin-bottom: 10px;
    }
    .metric-value { font-size: 2rem; font-weight: 700; color: #4fc3f7; }
    .metric-label { font-size: 0.85rem; color: #90a4ae; margin-top: 4px; }
    .aqi-good       { color: #00e676; }
    .aqi-moderate   { color: #ffeb3b; }
    .aqi-poor       { color: #ff9800; }
    .aqi-unhealthy  { color: #f44336; }
    .aqi-very       { color: #9c27b0; }
    .aqi-hazardous  { color: #7b1fa2; }
    h1, h2, h3 { color: #e0e0e0 !important; }
</style>
""", unsafe_allow_html=True)

# ─────────────────────────────────────────────
# AQI HELPER FUNCTIONS
# ─────────────────────────────────────────────
def cal_SOi(so2):
    if so2 <= 40:   return so2 * (50/40)
    elif so2 <= 80:  return 50 + (so2-40) * (50/40)
    elif so2 <= 380: return 100 + (so2-80) * (100/300)
    elif so2 <= 800: return 200 + (so2-380) * (100/420)
    elif so2 <= 1600:return 300 + (so2-800) * (100/800)
    else:            return 400 + (so2-1600) * (100/800)

def cal_Noi(no2):
    if no2 <= 40:   return no2 * 50/40
    elif no2 <= 80:  return 50 + (no2-40) * (50/40)
    elif no2 <= 180: return 100 + (no2-80) * (100/100)
    elif no2 <= 280: return 200 + (no2-180) * (100/100)
    elif no2 <= 400: return 300 + (no2-280) * (100/120)
    else:            return 400 + (no2-400) * (100/120)

def cal_RSPMi(rspm):
    if rspm <= 100:  return rspm
    elif rspm <= 150: return 101 + (rspm-101) * ((200-101)/(150-101))
    elif rspm <= 350: return 201 + (rspm-151) * ((300-201)/(350-151))
    elif rspm <= 420: return 301 + (rspm-351) * ((400-301)/(420-351))
    else:             return 401 + (rspm-420) * ((500-401)/(420-351))

def cal_SPMi(spm):
    if spm <= 50:   return spm
    elif spm <= 100:  return 50 + (spm-50)
    elif spm <= 250:  return 100 + (spm-100) * (100/150)
    elif spm <= 350:  return 200 + (spm-250)
    elif spm <= 430:  return 300 + (spm-350) * (100/80)
    else:             return 400 + (spm-430) * (100/430)

def cal_PMi(pm):
    if pm <= 50:    return pm
    elif pm <= 100:   return 50 + (pm-50)
    elif pm <= 250:   return 100 + (pm-100) * (100/150)
    elif pm <= 350:   return 200 + (pm-250)
    elif pm <= 450:   return 300 + (pm-350)
    else:             return 400 + (pm-430) * (100/80)

def aqi_category(aqi):
    if aqi <= 50:   return "Good", "#00e676"
    elif aqi <= 100: return "Moderate", "#ffeb3b"
    elif aqi <= 200: return "Poor", "#ff9800"
    elif aqi <= 300: return "Unhealthy", "#f44336"
    elif aqi <= 400: return "Very Unhealthy", "#9c27b0"
    else:            return "Hazardous", "#7b1fa2"

# ─────────────────────────────────────────────
# LOAD & PROCESS DATA
# ─────────────────────────────────────────────
import zipfile
import os

@st.cache_data
def load_data():
    # Unzip if not already extracted
    if not os.path.exists("dataset.csv"):
        with zipfile.ZipFile("dataset.zip", "r") as z:
            z.extractall(".")
    
    df = pd.read_csv("dataset.csv", low_memory=False)
    
    # Clean
    df.replace({'state': {r'Uttaranchal': 'Uttarakhand'}}, regex=True, inplace=True)
    df = df.dropna(subset=['type', 'location', 'so2'])
    for col in ['agency', 'location_monitoring_station', 'stn_code', 'sampling_date']:
        if col in df.columns:
            del df[col]

    # Simplify type
    types = []
    for t in df['type']:
        t = str(t)
        if t[0] == 'R' and len(t) > 1 and t[1] == 'e':
            types.append('Residential')
        elif t[0] == 'I':
            types.append('Industrial')
        else:
            types.append('Other')
    df['type'] = types

    # Impute by state
    grp = df.groupby('state')
    for col in ['rspm', 'so2', 'no2', 'spm', 'pm2_5']:
        df[col] = grp[col].transform(lambda s: s.fillna(s.mean()))

    # Sub-indices & AQI
    df['SOi']   = df['so2'].apply(cal_SOi)
    df['Noi']   = df['no2'].apply(cal_Noi)
    df['RSPMi'] = df['rspm'].apply(cal_RSPMi)
    df['SPMi']  = df['spm'].apply(cal_SPMi)
    df['PMi']   = df['pm2_5'].apply(cal_PMi)
    df['AQI']   = df[['SOi', 'Noi', 'RSPMi', 'SPMi', 'PMi']].max(axis=1)
    df['AQI_Range'] = df['AQI'].apply(lambda x: aqi_category(x)[0])

    df = df.dropna(subset=['spm', 'pm2_5'])
    df['date'] = pd.to_datetime(df['date'], format='%m/%d/%Y', errors='coerce')
    df['year'] = df['date'].dt.year.fillna(0).astype(int)
    df = df[df['year'] > 0]

    return df

@st.cache_resource
def train_model(df):
    X = df[['SOi', 'Noi', 'RSPMi', 'SPMi', 'PMi']]
    y = df['AQI']
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=101
    )
    model = LinearRegression()
    model.fit(X_train, y_train)
    preds = model.predict(X_test)
    r2   = r2_score(y_test, preds)
    rmse = np.sqrt(mean_squared_error(y_test, preds))
    return model, r2, rmse

# ─────────────────────────────────────────────
# LOAD
# ─────────────────────────────────────────────
with st.spinner("Loading data..."):
    df = load_data()
    model, r2, rmse = train_model(df)

states = sorted(df['state'].unique())
years  = sorted(df['year'].unique())

# ─────────────────────────────────────────────
# SIDEBAR
# ─────────────────────────────────────────────
with st.sidebar:
    st.image("https://img.icons8.com/fluency/96/air-quality.png", width=72)
    st.title("India AQI")
    st.markdown("---")
    st.markdown("### 🔍 Filters")

    selected_states = st.multiselect(
        "Select States", states, default=states[:5]
    )
    year_range = st.slider(
        "Year Range", int(min(years)), int(max(years)),
        (int(min(years)), int(max(years)))
    )
    area_type = st.multiselect(
        "Area Type", ['Residential', 'Industrial', 'Other'],
        default=['Residential', 'Industrial', 'Other']
    )

    st.markdown("---")
    st.markdown("### 📊 Model Performance")
    st.metric("R² Score", f"{r2:.4f}")
    st.metric("RMSE", f"{rmse:.2f}")
    st.markdown("---")
    st.caption("Data: India Air Quality | ML: Linear Regression")
    st.caption("Built by Aryan Pattani")
# ─────────────────────────────────────────────
# FILTER DATA
# ─────────────────────────────────────────────
mask = (
    df['state'].isin(selected_states if selected_states else states) &
    df['year'].between(year_range[0], year_range[1]) &
    df['type'].isin(area_type if area_type else ['Residential','Industrial','Other'])
)
fdf = df[mask]

# ─────────────────────────────────────────────
# HEADER
# ─────────────────────────────────────────────
st.title("🌫️ India Air Quality Index — Analysis & Prediction")
st.markdown(f"Showing **{len(fdf):,}** records across **{fdf['state'].nunique()}** states ({year_range[0]}–{year_range[1]})")
st.markdown("---")

# ─────────────────────────────────────────────
# TAB LAYOUT
# ─────────────────────────────────────────────
tab1, tab2, tab3, tab4 = st.tabs([
    "📊 Overview", "🗺️ State Analysis", "📈 Trends", "🤖 AQI Predictor"
])

# ════════════════════════════════════════════
# TAB 1 — OVERVIEW
# ════════════════════════════════════════════
with tab1:
    # KPI metrics
    col1, col2, col3, col4, col5 = st.columns(5)
    avg_aqi = fdf['AQI'].mean()
    cat, color = aqi_category(avg_aqi)

    with col1:
        st.metric("Avg AQI", f"{avg_aqi:.1f}", help="Average AQI across filtered data")
    with col2:
        st.metric("Category", cat)
    with col3:
        st.metric("Max AQI", f"{fdf['AQI'].max():.0f}")
    with col4:
        st.metric("Min AQI", f"{fdf['AQI'].min():.0f}")
    with col5:
        st.metric("Records", f"{len(fdf):,}")

    st.markdown("---")

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("AQI Category Distribution")
        cat_counts = fdf['AQI_Range'].value_counts().reset_index()
        cat_counts.columns = ['Category', 'Count']
        color_map = {
            'Good': '#00e676', 'Moderate': '#ffeb3b', 'Poor': '#ff9800',
            'Unhealthy': '#f44336', 'Very Unhealthy': '#9c27b0', 'Hazardous': '#7b1fa2'
        }
        fig = px.pie(
            cat_counts, names='Category', values='Count',
            color='Category', color_discrete_map=color_map,
            hole=0.45
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', legend_font_color='#e0e0e0'
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("Pollutant Levels by Area Type")
        pollutants = fdf.groupby('type')[['so2','no2','rspm','spm','pm2_5']].mean().reset_index()
        fig = px.bar(
            pollutants.melt(id_vars='type', var_name='Pollutant', value_name='Avg Level'),
            x='Pollutant', y='Avg Level', color='type', barmode='group',
            color_discrete_sequence=['#4fc3f7','#81c784','#ffb74d']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', xaxis=dict(gridcolor='#2d3250'),
            yaxis=dict(gridcolor='#2d3250')
        )
        st.plotly_chart(fig, use_container_width=True)

    # Correlation heatmap
    st.subheader("Pollutant Correlation Heatmap")
    corr = fdf[['so2','no2','rspm','spm','pm2_5','AQI']].corr().round(2)
    fig = go.Figure(data=go.Heatmap(
        z=corr.values, x=corr.columns, y=corr.columns,
        colorscale='Blues', text=corr.values,
        texttemplate="%{text}", textfont={"size":11},
        hoverongaps=False
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0', height=400
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# TAB 2 — STATE ANALYSIS
# ════════════════════════════════════════════
with tab2:
    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("Top 10 Most Polluted States (Avg AQI)")
        state_aqi = (
            fdf.groupby('state')['AQI'].mean()
            .sort_values(ascending=False).head(10).reset_index()
        )
        state_aqi.columns = ['State', 'Avg AQI']
        fig = px.bar(
            state_aqi, x='Avg AQI', y='State', orientation='h',
            color='Avg AQI', color_continuous_scale='RdYlGn_r',
            text=state_aqi['Avg AQI'].round(1)
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', yaxis=dict(autorange='reversed'),
            xaxis=dict(gridcolor='#2d3250'), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("SO₂ Levels by State")
        so2_state = (
            fdf.groupby('state')['so2'].median()
            .sort_values(ascending=False).head(10).reset_index()
        )
        fig = px.bar(
            so2_state, x='so2', y='state', orientation='h',
            color='so2', color_continuous_scale='OrRd',
            text=so2_state['so2'].round(1)
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', yaxis=dict(autorange='reversed'),
            xaxis=dict(gridcolor='#2d3250'), height=400
        )
        st.plotly_chart(fig, use_container_width=True)

    # AQI heatmap by state × year
    st.subheader("AQI Heatmap — State × Year")
    pivot = fdf.pivot_table('AQI', index='state', columns='year', aggfunc='median')
    fig = go.Figure(data=go.Heatmap(
        z=pivot.values, x=pivot.columns.astype(str),
        y=pivot.index, colorscale='RdYlGn_r',
        hoverongaps=False,
        colorbar=dict(title='Median AQI')
    ))
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0', height=450,
        xaxis=dict(title='Year'), yaxis=dict(title='State')
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# TAB 3 — TRENDS
# ════════════════════════════════════════════
with tab3:
    st.subheader("AQI Trend Over Years by State")
    trend = (
        fdf.groupby(['year','state'])['AQI'].mean().reset_index()
    )
    fig = px.line(
        trend, x='year', y='AQI', color='state',
        markers=True, color_discrete_sequence=px.colors.qualitative.Set2
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0',
        xaxis=dict(title='Year', gridcolor='#2d3250'),
        yaxis=dict(title='Average AQI', gridcolor='#2d3250'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

    col_left, col_right = st.columns(2)

    with col_left:
        st.subheader("NO₂ Trend Over Years")
        no2_trend = fdf.groupby('year')['no2'].mean().reset_index()
        fig = px.area(
            no2_trend, x='year', y='no2',
            color_discrete_sequence=['#4fc3f7']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis=dict(gridcolor='#2d3250'), yaxis=dict(gridcolor='#2d3250')
        )
        st.plotly_chart(fig, use_container_width=True)

    with col_right:
        st.subheader("PM2.5 Trend Over Years")
        pm_trend = fdf.groupby('year')['pm2_5'].mean().reset_index()
        fig = px.area(
            pm_trend, x='year', y='pm2_5',
            color_discrete_sequence=['#ef9a9a']
        )
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0',
            xaxis=dict(gridcolor='#2d3250'), yaxis=dict(gridcolor='#2d3250')
        )
        st.plotly_chart(fig, use_container_width=True)

    # AQI category breakdown over years
    st.subheader("AQI Category Breakdown Over Years")
    yearly_cat = (
        fdf.groupby(['year','AQI_Range']).size().reset_index(name='count')
    )
    cat_order = ['Good','Moderate','Poor','Unhealthy','Very Unhealthy','Hazardous']
    color_map = {
        'Good': '#00e676', 'Moderate': '#ffeb3b', 'Poor': '#ff9800',
        'Unhealthy': '#f44336', 'Very Unhealthy': '#9c27b0', 'Hazardous': '#7b1fa2'
    }
    fig = px.bar(
        yearly_cat, x='year', y='count', color='AQI_Range',
        color_discrete_map=color_map,
        category_orders={'AQI_Range': cat_order}
    )
    fig.update_layout(
        paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
        font_color='#e0e0e0', barmode='stack',
        xaxis=dict(gridcolor='#2d3250'), yaxis=dict(gridcolor='#2d3250'),
        legend=dict(bgcolor='rgba(0,0,0,0)')
    )
    st.plotly_chart(fig, use_container_width=True)

# ════════════════════════════════════════════
# TAB 4 — AQI PREDICTOR
# ════════════════════════════════════════════
with tab4:
    st.subheader("🤖 Live AQI Predictor")
    st.markdown("Enter pollutant concentrations to predict AQI using the trained Linear Regression model.")

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("#### Enter Pollutant Values")
        so2_val   = st.slider("SO₂ (µg/m³)",   0.0, 200.0, 20.0, 0.1)
        no2_val   = st.slider("NO₂ (µg/m³)",   0.0, 200.0, 30.0, 0.1)
        rspm_val  = st.slider("RSPM/PM10 (µg/m³)", 0.0, 500.0, 100.0, 1.0)
        spm_val   = st.slider("SPM (µg/m³)",   0.0, 500.0, 120.0, 1.0)
        pm25_val  = st.slider("PM2.5 (µg/m³)", 0.0, 300.0, 60.0, 0.5)

        predict_btn = st.button("🔍 Predict AQI", use_container_width=True, type="primary")

    with col2:
        st.markdown("#### Prediction Result")

        # Calculate sub-indices
        soi   = cal_SOi(so2_val)
        noi   = cal_Noi(no2_val)
        rspmi = cal_RSPMi(rspm_val)
        spmi  = cal_SPMi(spm_val)
        pmi   = cal_PMi(pm25_val)

        predicted_aqi = model.predict(np.array([[soi, noi, rspmi, spmi, pmi]]))[0]
        cat, color = aqi_category(predicted_aqi)

        # AQI Gauge
        fig = go.Figure(go.Indicator(
            mode="gauge+number",
            value=round(predicted_aqi, 1),
            domain={'x': [0, 1], 'y': [0, 1]},
            title={'text': f"Predicted AQI — <b>{cat}</b>",
                   'font': {'color': color, 'size': 18}},
            gauge={
                'axis': {'range': [0, 500], 'tickcolor': '#e0e0e0',
                         'tickfont': {'color': '#e0e0e0'}},
                'bar': {'color': color},
                'bgcolor': '#1e2130',
                'bordercolor': '#2d3250',
                'steps': [
                    {'range': [0, 50],   'color': '#00251a'},
                    {'range': [50, 100], 'color': '#1b1400'},
                    {'range': [100, 200],'color': '#1a0d00'},
                    {'range': [200, 300],'color': '#1a0000'},
                    {'range': [300, 400],'color': '#12002a'},
                    {'range': [400, 500],'color': '#0d0014'},
                ],
                'threshold': {
                    'line': {'color': color, 'width': 4},
                    'thickness': 0.75,
                    'value': predicted_aqi
                }
            }
        ))
        fig.update_layout(
            paper_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', height=300
        )
        st.plotly_chart(fig, use_container_width=True)

        # Sub-index breakdown
        st.markdown("#### Sub-index Breakdown")
        sub_df = pd.DataFrame({
            'Pollutant': ['SO₂', 'NO₂', 'RSPM', 'SPM', 'PM2.5'],
            'Sub-index': [round(soi,1), round(noi,1), round(rspmi,1), round(spmi,1), round(pmi,1)]
        })
        fig2 = px.bar(
            sub_df, x='Pollutant', y='Sub-index',
            color='Sub-index', color_continuous_scale='RdYlGn_r',
            text='Sub-index'
        )
        fig2.update_layout(
            paper_bgcolor='rgba(0,0,0,0)', plot_bgcolor='rgba(0,0,0,0)',
            font_color='#e0e0e0', showlegend=False,
            xaxis=dict(gridcolor='#2d3250'),
            yaxis=dict(gridcolor='#2d3250'), height=250
        )
        st.plotly_chart(fig2, use_container_width=True)

    # AQI Reference Table
    st.markdown("---")
    st.subheader("📋 AQI Reference Scale")
    ref_data = {
        'AQI Range': ['0–50','51–100','101–200','201–300','301–400','401+'],
        'Category':  ['Good','Moderate','Poor','Unhealthy','Very Unhealthy','Hazardous'],
        'Health Impact': [
            'Minimal impact',
            'Minor breathing discomfort to sensitive people',
            'Breathing discomfort to people with lung/heart disease',
            'Breathing discomfort to most people on prolonged exposure',
            'Respiratory illness on prolonged exposure',
            'Affects healthy people; seriously impacts those with existing diseases'
        ]
    }
    st.dataframe(
        pd.DataFrame(ref_data),
        use_container_width=True,
        hide_index=True
    )
