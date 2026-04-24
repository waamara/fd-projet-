import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from sklearn.decomposition import PCA

# ── K-Medoids manuel ────────────────────────────────────────────
def kmedoids(X, k, max_iter=100):
    np.random.seed(42)
    indices = np.random.choice(len(X), k, replace=False)
    medoids = X[indices]

    for _ in range(max_iter):
        distances = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X])
        labels = np.argmin(distances, axis=1)
        new_medoids = np.copy(medoids)
        for i in range(k):
            cluster_pts = X[labels == i]
            if len(cluster_pts) == 0:
                continue
            intra = np.sum([np.linalg.norm(cluster_pts - p, axis=1).sum() for p in cluster_pts])
            best = np.argmin([np.sum(np.linalg.norm(cluster_pts - p, axis=1)) for p in cluster_pts])
            new_medoids[i] = cluster_pts[best]
        if np.allclose(medoids, new_medoids):
            break
        medoids = new_medoids

    distances = np.array([[np.linalg.norm(x - m) for m in medoids] for x in X])
    labels = np.argmin(distances, axis=1)
    return labels, medoids

def show_clustering():
    st.header("🔵 Volet 2 — Clustering")

    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset dans le Volet 1.")
        return

    df = st.session_state.df.copy()
    num_cols = df.select_dtypes(include=np.number).columns.tolist()
    X = df[num_cols].dropna().values

    # ── COURBE D'ELBOW ───────────────────────────────────────────
    st.subheader("1. Courbe d'Elbow")
    max_k = st.slider("Nombre max de clusters à tester", 2, 10, 8)

    if st.button("Afficher la courbe d'Elbow"):
        inertias = []
        for k in range(1, max_k + 1):
            km = KMeans(n_clusters=k, random_state=42, n_init=10)
            km.fit(X)
            inertias.append(km.inertia_)

        fig, ax = plt.subplots()
        ax.plot(range(1, max_k + 1), inertias, marker='o', color='steelblue')
        ax.set_xlabel("Nombre de clusters (k)")
        ax.set_ylabel("Inertie")
        ax.set_title("Courbe d'Elbow")
        st.pyplot(fig)

    # ── CHOIX ALGORITHME ─────────────────────────────────────────
    st.subheader("2. Paramètres du Clustering")
    algo = st.selectbox("Algorithme", ["K-Means", "K-Medoids"])
    k = st.slider("Nombre de clusters (k)", 2, 10, 3)

    if st.button("Lancer le Clustering"):
        if algo == "K-Means":
            model = KMeans(n_clusters=k, random_state=42, n_init=10)
            labels = model.fit_predict(X)
        else:
            labels, _ = kmedoids(X, k)

        # ── SCORE SILHOUETTE ─────────────────────────────────────
        score = silhouette_score(X, labels)
        st.success(f"✅ Score de Silhouette : **{score:.4f}**")

        # ── VISUALISATION 2D ─────────────────────────────────────
        st.subheader("3. Visualisation 2D des Clusters")
        if X.shape[1] > 2:
            pca = PCA(n_components=2)
            X_2d = pca.fit_transform(X)
            st.caption("Projection PCA utilisée (données > 2 dimensions)")
        else:
            X_2d = X

        fig2, ax2 = plt.subplots()
        scatter = ax2.scatter(X_2d[:, 0], X_2d[:, 1], c=labels, cmap='tab10', alpha=0.7)
        ax2.set_title(f"Clusters ({algo}, k={k})")
        ax2.set_xlabel("Composante 1")
        ax2.set_ylabel("Composante 2")
        plt.colorbar(scatter, ax=ax2)
        st.pyplot(fig2)