import streamlit as st
import pandas as pd
import plotly.express as px
from sklearn_extra.cluster import KMedoids

st.set_page_config(page_title="Geo Clustering Dashboard", layout="wide")
st.title("Geo Clustering Dashboard (K-Medoids)")

# --------------------------------------------------
# Load pincode master (cached)
# --------------------------------------------------
@st.cache_data
def load_pincode_master():
    df = pd.read_csv("All-India-Pincode-list-with-latitude-and-longitude.csv")
    df = df.rename(columns={
        "Pincode": "pincode",
        "StateName": "State",
        "Latitude": "Latitude",
        "Longitude": "Longitude"
    })
    df["pincode"] = df["pincode"].astype(str).str.strip()
    df["State"] = df["State"].astype(str).str.strip().str.title()
    return df[["pincode", "State", "Latitude", "Longitude"]]

pincode_master = load_pincode_master()

# --------------------------------------------------
# Upload file (ONLY PINCODES)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload pincode file (CSV or Excel – must contain a 'pincode' column)",
    type=["csv", "xlsx"]
)

if uploaded_file is not None:

    # Read uploaded file
    if uploaded_file.name.endswith(".csv"):
        user_df = pd.read_csv(uploaded_file)
    else:
        user_df = pd.read_excel(uploaded_file)

    # Normalize column names
    user_df.columns = user_df.columns.str.strip().str.lower()

    if "pincode" not in user_df.columns:
        st.error("Uploaded file must contain a 'pincode' column.")
        st.stop()

    user_df["pincode"] = user_df["pincode"].astype(str).str.strip()

    # --------------------------------------------------
    # Merge with pincode master
    # --------------------------------------------------
    df = user_df.merge(pincode_master, on="pincode", how="left")

    df = df.dropna(subset=["Latitude", "Longitude", "State"])

    if df.empty:
        st.error("No valid pincodes found after mapping.")
        st.stop()

    # --------------------------------------------------
    # Clustering options
    # --------------------------------------------------
    st.sidebar.header("Clustering Options")

    scope = st.sidebar.radio(
        "Clustering Scope",
        ["All India", "State-wise"]
    )

    if scope == "State-wise":
        states = sorted(df["State"].unique())
        selected_state = st.sidebar.selectbox("Select State", states)
        df = df[df["State"] == selected_state]

    # Number of clusters
    max_k = min(10, len(df))
    k = st.sidebar.slider("Number of clusters (k)", 2, max_k, 3)

    # --------------------------------------------------
    # K-Medoids clustering
    # --------------------------------------------------
    X = df[["Latitude", "Longitude"]].values

    kmedoids = KMedoids(
        n_clusters=k,
        method="pam",
        random_state=42
    )

    df["cluster"] = kmedoids.fit_predict(X)

    # Identify medoids
    df["is_medoid"] = 0
    df.loc[kmedoids.medoid_indices_, "is_medoid"] = 1

    # --------------------------------------------------
    # Metrics
    # --------------------------------------------------
    col1, col2 = st.columns(2)
    col1.metric("Total Data Points", len(df))
    col2.metric("Number of Clusters", df["cluster"].nunique())

    # --------------------------------------------------
    # Cluster stats (centroid = medoid)
    # --------------------------------------------------
    cluster_stats = (
        df.groupby("cluster")
        .agg(
            cluster_size=("cluster", "size"),
            medoid_lat=("Latitude", "first"),
            medoid_lon=("Longitude", "first")
        )
        .reset_index()
    )

    df = df.merge(cluster_stats, on="cluster", how="left")

    # --------------------------------------------------
    # Map visualization
    # --------------------------------------------------
    fig = px.scatter_mapbox(
        df,
        lat="Latitude",
        lon="Longitude",
        color="cluster",
        mapbox_style="open-street-map",
        zoom=5,
        center={
            "lat": df["Latitude"].mean(),
            "lon": df["Longitude"].mean()
        },
        hover_data={
            "pincode": True,
            "cluster": True,
            "cluster_size": True,
            "medoid_lat": True,
            "medoid_lon": True
        }
    )

    # Highlight medoids
    fig.update_traces(marker=dict(size=6, opacity=0.7))
    fig.add_scattermapbox(
        lat=df[df["is_medoid"] == 1]["Latitude"],
        lon=df[df["is_medoid"] == 1]["Longitude"],
        mode="markers",
        marker=dict(size=16, color="black"),
        name="Medoids (Cluster Centers)"
    )

    st.plotly_chart(fig, use_container_width=True)
