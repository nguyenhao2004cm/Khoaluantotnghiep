import os
import pandas as pd
import numpy as np

from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

# ================================
# CONFIG
# ================================
LATENT_DIR = "data_processed/latent"
OUT_DIR = "data_processed/clusters"
N_CLUSTERS = 4   # theo Aithal (2023): 3–6 là hợp lý

os.makedirs(OUT_DIR, exist_ok=True)

# ================================
# LOAD & MERGE LATENT DATA
# ================================
def load_latent_data():
    frames = []

    for file in os.listdir(LATENT_DIR):
        if file.endswith("_latent.csv"):
            symbol = file.replace("_latent.csv", "")
            df = pd.read_csv(os.path.join(LATENT_DIR, file))
            df["symbol"] = symbol
            frames.append(df)

    if len(frames) == 0:
        raise ValueError(" Không tìm thấy latent files")

    return pd.concat(frames, ignore_index=True)


# ================================
# CLUSTERING
# ================================
def run_kmeans(df: pd.DataFrame):
    latent_cols = [c for c in df.columns if c.startswith("latent_")]

    X = df[latent_cols].values

    # Chuẩn hóa (Bertani, 2021)
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    kmeans = KMeans(
        n_clusters=N_CLUSTERS,
        random_state=42,
        n_init=10
    )

    df["cluster"] = kmeans.fit_predict(X_scaled)

    return df, kmeans


# ================================
# PIPELINE
# ================================
def run():
    print(" Asset clustering (K-Means)")

    df = load_latent_data()

    df_clustered, model = run_kmeans(df)

    # Lưu kết quả
    out_path = os.path.join(OUT_DIR, "asset_clusters.csv")
    df_clustered.to_csv(out_path, index=False, encoding="utf-8-sig")

    print(f" Clustering xong – lưu tại {out_path}")

    # Thống kê nhanh
    print("\n Số cổ phiếu theo cluster:")
    print(
        df_clustered[["symbol", "cluster"]]
        .drop_duplicates()
        .groupby("cluster")
        .count()
    )


# ================================
# ENTRY POINT
# ================================
if __name__ == "__main__":
    run()
