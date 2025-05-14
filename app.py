import streamlit as st
import pandas as pd
import numpy as np
import pydeck as pdk
import plotly.express as px
from io import BytesIO
from fpdf import FPDF
from sklearn.cluster import KMeans
from statsmodels.tsa.arima.model import ARIMA
import warnings

warnings.filterwarnings("ignore")

st.set_page_config(layout="wide")
st.title("🚗 Travel Time Runs Dashboard (Updated Column Format)")

uploaded_file = st.file_uploader("📁 Upload travel time run data (CSV)", type="csv")

if uploaded_file:
    df = pd.read_csv(uploaded_file)
    df.columns = df.columns.str.strip()

    expected_cols = ['TRACK ID', 'UTC DATE', 'UTC TIME', 'LATITUDE', 'LONGITUDE', 'SPEED']
    missing_cols = [col for col in expected_cols if col not in df.columns]
    if missing_cols:
        st.error(f"❌ Missing columns in uploaded file: {missing_cols}")
        st.stop()

    df['DATETIME'] = pd.to_datetime(df['UTC DATE'] + ' ' + df['UTC TIME'], errors='coerce')
    df = df.dropna(subset=['DATETIME'])
    df.sort_values('DATETIME', inplace=True)

    st.sidebar.header("🔍 Filters")
    track_ids = df['TRACK ID'].unique()
    selected_tracks = st.sidebar.multiselect("Track ID(s)", track_ids, default=track_ids[:2])
    date_range = st.sidebar.date_input("Date Range", [df['DATETIME'].min().date(), df['DATETIME'].max().date()])

    mask = (
        df['TRACK ID'].isin(selected_tracks) &
        (df['DATETIME'].dt.date >= date_range[0]) &
        (df['DATETIME'].dt.date <= date_range[1])
    )
    filtered_df = df[mask]

    def detect_stops(data, stop_threshold_kmh=5, min_stop_duration_sec=30):
        data = data.sort_values('DATETIME').copy()
        data['is_stop'] = data['SPEED'] < stop_threshold_kmh
        data['stop_group'] = (data['is_stop'] != data['is_stop'].shift()).cumsum()
        stop_info = []
        for _, group in data.groupby(['stop_group']):
            if group['is_stop'].iloc[0]:
                duration = (group['DATETIME'].iloc[-1] - group['DATETIME'].iloc[0]).total_seconds()
                if duration >= min_stop_duration_sec:
                    stop_info.append({'start': group['DATETIME'].iloc[0],
                                      'end': group['DATETIME'].iloc[-1],
                                      'duration_sec': duration})
        return pd.DataFrame(stop_info)

    kpi_rows = []
    for track_id in selected_tracks:
        track_data = filtered_df[filtered_df['TRACK ID'] == track_id]
        travel_time = (track_data['DATETIME'].max() - track_data['DATETIME'].min()).total_seconds() / 3600
        avg_speed = track_data['SPEED'].mean()
        distance = avg_speed * travel_time
        stops_df = detect_stops(track_data)
        num_stops = len(stops_df)
        avg_stop_dur = stops_df['duration_sec'].mean() if num_stops > 0 else 0
        num_slowdowns = (track_data['SPEED'] < 15).sum()

        kpi_rows.append({
            'Track ID': track_id,
            'Avg Speed (km/h)': round(avg_speed, 2),
            'Distance (km)': round(distance, 2),
            'Travel Time (hrs)': round(travel_time, 2),
            'Num Stops': num_stops,
            'Avg Stop Duration (s)': round(avg_stop_dur, 2),
            'Num Slowdowns (<15 km/h)': num_slowdowns
        })

    st.subheader("📊 KPI Summary")
    kpi_df = pd.DataFrame(kpi_rows)
    st.dataframe(kpi_df)

    st.subheader("🗺️ Slowdown Heatmap")
    slowdown_points = filtered_df[filtered_df['SPEED'] < 15]
    st.map(slowdown_points.rename(columns={"LATITUDE": "lat", "LONGITUDE": "lon"}))

    st.subheader("📈 Speed Over Time")
    fig = px.line(filtered_df, x="DATETIME", y="SPEED", color="TRACK ID")
    st.plotly_chart(fig)

    st.subheader("🔁 Clustering Travel Patterns")
    if len(kpi_df) >= 2:
        X = kpi_df[['Avg Speed (km/h)', 'Distance (km)', 'Num Stops', 'Num Slowdowns (<15 km/h)']].fillna(0)
        kmeans = KMeans(n_clusters=2, random_state=42)
        kpi_df['Cluster'] = kmeans.fit_predict(X)
        fig2 = px.scatter_matrix(
            kpi_df,
            dimensions=['Avg Speed (km/h)', 'Distance (km)', 'Num Stops', 'Num Slowdowns (<15 km/h)'],
            color='Cluster',
            symbol='Track ID',
            title="Cluster Analysis"
        )
        st.plotly_chart(fig2)

    st.subheader("📉 Predictive Speed Forecast")
    selected_track = st.selectbox("Select Track for Forecasting", df['TRACK ID'].unique())
    track_df = df[df['TRACK ID'] == selected_track].copy()
    track_df = track_df.sort_values("DATETIME")
    track_df.set_index("DATETIME", inplace=True)

    if len(track_df) > 20:
        model = ARIMA(track_df["SPEED"], order=(2, 1, 2))
        model_fit = model.fit()
        forecast = model_fit.forecast(steps=10)
        forecast_df = forecast.to_frame(name="Forecasted Speed")
        st.line_chart(pd.concat([track_df["SPEED"], forecast_df]))
    else:
        st.warning("Not enough data points for time-series forecasting.")

    st.subheader("⬇️ Download Filtered Data")
    csv = filtered_df.to_csv(index=False).encode('utf-8')
    st.download_button("Download CSV", csv, "filtered_travel_time_data.csv", "text/csv")

    st.subheader("📄 Generate PDF Report")
    if st.button("Generate PDF"):
        pdf = FPDF()
        pdf.add_page()
        pdf.set_font("Arial", size=12)
        pdf.cell(200, 10, txt="Travel Time Summary Report", ln=True, align='C')
        for i, row in kpi_df.iterrows():
            pdf.ln(10)
            for col in kpi_df.columns:
                pdf.cell(0, 10, f"{col}: {row[col]}", ln=True)
        pdf_output = BytesIO()
        pdf.output(pdf_output)
        pdf_output.seek(0)
        st.download_button("Download PDF", pdf_output, "report.pdf", "application/pdf")
else:
    st.info("Upload a CSV file to get started.")