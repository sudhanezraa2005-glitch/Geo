import streamlit as st
import pandas as pd
import numpy as np
import plotly.express as px
import plotly.graph_objects as go
import requests
from sklearn.cluster import KMeans

# --------------------------------------------------
# Streamlit config
# --------------------------------------------------
st.set_page_config(page_title="Geo Clustering Dashboard", layout="wide")
st.set_option("client.showErrorDetails", True)
st.title("Geo Clustering Dashboard – Euclidean & Road Distance (K-Medoids)")

# --------------------------------------------------
# Pure Python K-Medoids
# --------------------------------------------------
def k_medoids(X, k, dist_matrix=None, max_iter=100):
    n = len(X)
    medoids = np.random.choice(n, k, replace=False)

    for _ in range(max_iter):
        if dist_matrix is None:
            dists = np.linalg.norm(X[:, None] - X[medoids][None, :], axis=2)
        else:
            dists = dist_matrix[:, medoids]

        labels = np.argmin(dists, axis=1)
        new_medoids = medoids.copy()

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

        if np.all(new_medoids == medoids):
            break

        medoids = new_medoids

    return labels, medoids


# --------------------------------------------------
# OSRM helpers
# --------------------------------------------------
def osrm_distance_matrix(coords):
    if len(coords) > 80:
        raise ValueError("OSRM public API supports up to ~80 points per request")

    coord_str = ";".join([f"{lon},{lat}" for lat, lon in coords])
    url = f"https://router.project-osrm.org/table/v1/driving/{coord_str}?annotations=distance"
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return np.array(r.json()["distances"])


def osrm_route(src, dst):
    url = (
        f"https://router.project-osrm.org/route/v1/driving/"
        f"{src[1]},{src[0]};{dst[1]},{dst[0]}"
        "?overview=full&geometries=geojson"
    )
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    return r.json()["routes"][0]["geometry"]["coordinates"]


def split_spatially(X, max_size=70):
    n_chunks = int(np.ceil(len(X) / max_size))
    km = KMeans(n_clusters=n_chunks, random_state=42, n_init=10)
    return km.fit_predict(X)


# --------------------------------------------------
# Load pincode master
# --------------------------------------------------
@st.cache_data
def load_pincode_master():
    df = pd.read_csv("All-India-Pincode-list-with-latitude-and-longitude.csv")
    df = df.rename(columns={"Pincode": "pincode", "StateName": "State"})
    df["pincode"] = df["pincode"].astype(str).str.strip()
    df["State"] = df["State"].astype(str).str.strip().str.title()
    return df[["pincode", "State", "Latitude", "Longitude"]]


pincode_master = load_pincode_master()

# --------------------------------------------------
# Upload user file
# --------------------------------------------------
uploaded = st.file_uploader(
    "Upload CSV / Excel with a `pincode` column",
    type=["csv", "xlsx"]
)

if uploaded is None:
    st.info("Upload a file to begin.")
    st.stop()

user_df = pd.read_csv(uploaded) if uploaded.name.endswith(".csv") else pd.read_excel(uploaded)
user_df.columns = user_df.columns.str.strip().str.lower()

if "pincode" not in user_df.columns:
    st.error("File must contain a `pincode` column.")
    st.stop()

user_df["pincode"] = user_df["pincode"].astype(str).str.strip()

# --------------------------------------------------
# Merge geo
# --------------------------------------------------
df = user_df.merge(pincode_master, on="pincode", how="left")
df = df.dropna(subset=["Latitude", "Longitude", "State"])

if len(df) < 2:
    st.error("Not enough valid pincodes after mapping.")
    st.stop()

# --------------------------------------------------
# Sidebar controls
# --------------------------------------------------
st.sidebar.header("Options")

scope = st.sidebar.radio("Clustering Scope", ["All India", "State-wise"])
if scope == "State-wise":
    state = st.sidebar.selectbox("Select State", sorted(df["State"].unique()))
    df = df[df["State"] == state]

df = df.drop_duplicates(subset=["Latitude", "Longitude"]).reset_index(drop=True)
if len(df) < 2:
    st.warning("Not enough points after filtering.")
    st.stop()

distance_type = st.sidebar.radio(
    "Distance Type",
    ["Euclidean", "Real Road Distance (OSRM)"]
)

max_k = min(8, len(df) - 1)
k = st.sidebar.slider("Number of clusters (k)", 2, max_k, min(3, max_k))

# --------------------------------------------------
# Stage 1: Spatial clustering
# --------------------------------------------------
X = df[["Latitude", "Longitude"]].values
df["spatial"] = KMeans(n_clusters=k, random_state=42, n_init=10).fit_predict(X)

# --------------------------------------------------
# Stage 2: K-Medoids + OSRM
# --------------------------------------------------
final_labels = np.zeros(len(df), dtype=int)
final_medoids = []
cluster_offset = 0
road_paths = []

for c in sorted(df["spatial"].unique()):
    sub = df[df["spatial"] == c]
    idx = sub.index.to_numpy()
    X_sub = sub[["Latitude", "Longitude"]].values

    if len(sub) <= 2:
        final_labels[idx] = cluster_offset
        final_medoids.append(idx[0])
        cluster_offset += 1
        continue

    if distance_type.startswith("Real"):
        if len(sub) <= 80:
            coords = list(zip(sub["Latitude"], sub["Longitude"]))
            dist_mat = osrm_distance_matrix(coords)
            labels, medoids = k_medoids(X_sub, 2, dist_matrix=dist_mat)
        else:
            labels = np.zeros(len(sub), dtype=int)
            medoids = []
            split_lbls = split_spatially(X_sub)

            off = 0
            for sc in np.unique(split_lbls):
                sidx = np.where(split_lbls == sc)[0]
                chunk = X_sub[sidx]
                if len(chunk) < 2:
                    labels[sidx] = off
                    medoids.append(sidx[0])
                    off += 1
                    continue

                coords = list(zip(
                    sub.iloc[sidx]["Latitude"],
                    sub.iloc[sidx]["Longitude"]
                ))
                dist_mat = osrm_distance_matrix(coords)
                l, m = k_medoids(chunk, 2, dist_matrix=dist_mat)
                labels[sidx] = l + off
                medoids.extend(sidx[m])
                off += 2
    else:
        labels, medoids = k_medoids(X_sub, 2)

    final_labels[idx] = labels + cluster_offset
    final_medoids.extend(idx[medoids])
    cluster_offset += len(np.unique(labels))

df["cluster"] = final_labels
df["is_medoid"] = 0
df.loc[final_medoids, "is_medoid"] = 1

# --------------------------------------------------
# Build road paths (medoid → cluster points)
# --------------------------------------------------
if distance_type.startswith("Real"):
    for cid in df["cluster"].unique():
        cluster_df = df[df["cluster"] == cid]
        medoid = cluster_df[cluster_df["is_medoid"] == 1].iloc[0]
        src = (medoid["Latitude"], medoid["Longitude"])

        for _, row in cluster_df.iterrows():
            if row["is_medoid"] == 1:
                continue
            dst = (row["Latitude"], row["Longitude"])
            try:
                path = osrm_route(src, dst)
                road_paths.append(path)
            except:
                pass

# --------------------------------------------------
# Metrics
# --------------------------------------------------
c1, c2 = st.columns(2)
c1.metric("Total Points", len(df))
c2.metric("Clusters", df["cluster"].nunique())

# --------------------------------------------------
# Map
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

for path in road_paths:
    lons, lats = zip(*path)
    fig.add_trace(go.Scattermapbox(
        lon=lons,
        lat=lats,
        mode="lines",
        line=dict(width=2, color="black"),
        opacity=0.4,
        showlegend=False
    ))

st.plotly_chart(fig, use_container_width=True)
