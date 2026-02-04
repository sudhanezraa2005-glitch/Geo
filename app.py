# app.py
import streamlit as st
import pandas as pd
import plotly.express as px

st.set_page_config(page_title="Geo Clustering Dashboard", layout="wide")

st.title("Geo Clustering Dashboard")

# 1. Upload CSV / Excel
uploaded_file = st.file_uploader(
    "Upload pincode file (CSV or Excel)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:
    if uploaded_file.name.endswith(".csv"):
        df = pd.read_csv(uploaded_file)
    else:
        df = pd.read_excel(uploaded_file)

    # Basic validation
    required_cols = ["State", "Latitude", "Longitude", "cluster"]
    if not all(col in df.columns for col in required_cols):
        st.error("Uploaded file does not contain required columns.")
        st.stop()

    # Normalize state names
    df["State"] = df["State"].astype(str).str.strip().str.title()

    # 2. State dropdown
    states = sorted(df["State"].unique())
    selected_state = st.selectbox("Select State", states)

    state_df = df[df["State"] == selected_state]

    # 3. Metrics
    col1, col2 = st.columns(2)
    col1.metric("Total Data Points", len(state_df))
    col2.metric("Number of Clusters", state_df["cluster"].nunique())

    # 4. Cluster stats (centroid + count)
    cluster_stats = (
        state_df.groupby("cluster")
        .agg(
            centroid_lat=("Latitude", "mean"),
            centroid_lon=("Longitude", "mean"),
            cluster_size=("cluster", "size")
        )
        .reset_index()
    )

    state_df = state_df.merge(cluster_stats, on="cluster", how="left")

    # 5. Map
    fig = px.scatter_mapbox(
        state_df,
        lat="Latitude",
        lon="Longitude",
        color="cluster",
        mapbox_style="open-street-map",
        zoom=6,
        center={
            "lat": state_df["Latitude"].mean(),
            "lon": state_df["Longitude"].mean()
        },
        hover_data={
            "cluster": True,
            "cluster_size": True,
            "centroid_lat": True,
            "centroid_lon": True
        }
    )

    st.plotly_chart(fig, use_container_width=True)
