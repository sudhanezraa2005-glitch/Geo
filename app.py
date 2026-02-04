import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import requests
from sklearn.cluster import KMeans

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Geo Clustering Dashboard", layout="wide")
st.set_option("client.showErrorDetails", True)
st.title("Geo Clustering Dashboard")

# --------------------------------------------------
# Pure Python K-Medoids (works everywhere)
# --------------------------------------------------
def k_medoids(X, k, dist_matrix=None, max_iter=100):
    n = X.shape[0]
    medoid_indices = np.random.choice(n, k, replace=False)

    for _ in range(max_iter):
        if dist_matrix is None:
            dists = np.linalg.norm(X[:, None, :] - X[medoid_indices][None, :, :], axis=2)
        else:
            dists = dist_matrix[:, medoid_indices]

        labels = np.argmin(dists, axis=1)
        new_medoids = medoid_indices.copy()

        for i in range(k):
            idx = np.where(labels == i)[0]
            if len(idx) == 0:
                continue

            if dist_matrix is None:
                intra = np.sum(
                    np.linalg.norm(X[idx][:, None] - X[idx][None, :], axis=2),
                    axis=1
                )
            else:
                intra = np.sum(dist_matrix[np.ix_(idx, idx)], axis=1)

            new_medoids[i] = idx[np.argmin(intra)]

        if np.all(new_medoids == medoid_indices):
            break

        medoid_indices = new_medoids

    return labels, medoid_indices


# --------------------------------------------------
# OSRM distance matrix (batched, safe)
# --------------------------------------------------
def osrm_distance_matrix(coords):
    if len(coords) > 80:
        raise ValueError("OSRM public API supports up to ~80 points per request")

    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"https://router.project-osrm.org/table/v1/driving/{coord_str}?annotations=distance"

    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["distances"])


# --------------------------------------------------
# Load pincode master
# --------------------------------------------------
@st.cache_data
def load_pincode_master():
    df = pd.read_csv("All-India-Pincode-list-with-latitude-and-longitude.csv")
    df = df.rename(columns={
        "Pincode": "pincode",
        "StateName": "State"
    })
    df["pincode"] = df["pincode"].astype(str).str.strip()
    df["State"] = df["State"].astype(str).str.strip().str.title()
    return df[["pincode", "State", "Latitude", "Longitude"]]


pincode_master = load_pincode_master()

# --------------------------------------------------
# Upload user file (ONLY PINCODE)
# --------------------------------------------------
uploaded_file = st.file_uploader(
    "Upload pincode file (CSV / Excel with a `pincode` column)",
    type=["csv", "xlsx"]
)

if uploaded_file is None:
    st.info("Please upload a file to begin.")
    st.stop()

if uploaded_file.name.endswith(".csv"):
    user_df = pd.read_csv(uploaded_file)
else:
    user_df = pd.read_excel(uploaded_file)

user_df.columns = user_df.columns.str.strip().str.lower()

if "pincode" not in user_df.columns:
    st.error("Uploaded file must contain a `pincode` column.")
    st.stop()

user_df["pincode"] = user_df["pincode"].astype(str).str.strip()

# --------------------------------------------------
# Merge geo info
# --------------------------------------------------
df = user_df.merge(pincode_master, on="pincode", how="left")
df = df.dropna(subset=["Latitude", "Longitude", "State"])

if df.shape[0] < 2:
    st.error("Not enough valid pincodes after geo-mapping.")
    st.stop()

# --------------------------------------------------
# Sidebar options
# --------------------------------------------------
st.sidebar.header("Clustering Options")

scope = st.sidebar.radio("Clustering Scope", ["All India", "State-wise"])

if scope == "State-wise":
    state = st.sidebar.selectbox("Select State", sorted(df["State"].unique()))
    df = df[df["State"] == state]

df = df.drop_duplicates(subset=["Latitude", "Longitude"]).reset_index(drop=True)

if df.shape[0] < 2:
    st.warning("Not enough points after filtering.")
    st.stop()

distance_type = st.sidebar.radio(
    "Distance Type",
    ["Euclidean (Fast)", "Real Road Distance (OSRM)"]
)

max_k = min(8, df.shape[0] - 1)
k = st.sidebar.slider("Number of clusters (k)", 2, max_k, min(3, max_k))

# --------------------------------------------------
# STAGE 1: Spatial clustering (always)
# --------------------------------------------------
X = df[["Latitude", "Longitude"]].values

kmeans = KMeans(n_clusters=k, random_state=42, n_init=10)
df["spatial_cluster"] = kmeans.fit_predict(X)

# --------------------------------------------------
# STAGE 2: K-Medoids refinement
# --------------------------------------------------
final_labels = np.zeros(len(df), dtype=int)
final_medoids = []

cluster_offset = 0

for c in sorted(df["spatial_cluster"].unique()):
    sub = df[df["spatial_cluster"] == c].copy()
    sub_idx = sub.index.to_numpy()

    X_sub = sub[["Latitude", "Longitude"]].values

    if len(sub) <= 2:
        final_labels[sub_idx] = cluster_offset
        final_medoids.append(sub_idx[0])
        cluster_offset += 1
        continue

    k_sub = min(2, len(sub) - 1)

    if distance_type.startswith("Real"):
        coords = list(zip(sub["Latitude"], sub["Longitude"]))
        dist_mat = osrm_distance_matrix(coords)
        labels, medoids = k_medoids(X_sub, k_sub, dist_matrix=dist_mat)
    else:
        labels, medoids = k_medoids(X_sub, k_sub)

    final_labels[sub_idx] = labels + cluster_offset
    final_medoids.extend(sub_idx[medoids])
    cluster_offset += k_sub

df["cluster"] = final_labels
df["is_medoid"] = 0
df.loc[final_medoids, "is_medoid"] = 1

# --------------------------------------------------
# Metrics
# --------------------------------------------------
col1, col2 = st.columns(2)
col1.metric("Total Data Points", len(df))
col2.metric("Final Clusters", df["cluster"].nunique())

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
    center={"lat": df["Latitude"].mean(), "lon": df["Longitude"].mean()},
    hover_data=["pincode", "cluster"]
)

fig.update_traces(marker=dict(size=6, opacity=0.7))

fig.add_scattermapbox(
    lat=df[df["is_medoid"] == 1]["Latitude"],
    lon=df[df["is_medoid"] == 1]["Longitude"],
    mode="markers",
    marker=dict(size=16, color="black"),
    name="Medoids"
)

st.plotly_chart(fig, use_container_width=True)
