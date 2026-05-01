import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.preprocessing import MinMaxScaler, StandardScaler

def show_preprocessing():
    st.header("📊 Volet 1 — Prétraitement")

    # ── 1. IMPORTATION ──────────────────────────────────────────
    st.subheader("1. Importation du Dataset")
    uploaded_file = st.file_uploader("Charger un fichier CSV ou Excel", type=["csv", "xlsx"])

    if uploaded_file is not None:
        if uploaded_file.name.endswith(".csv"):
            st.session_state.df = pd.read_csv(uploaded_file)
        else:
            st.session_state.df = pd.read_excel(uploaded_file)
        st.success("✅ Fichier chargé avec succès !")

    if st.session_state.df is None:
        st.info("Veuillez charger un fichier pour commencer.")
        return

    df = st.session_state.df.copy()
    st.dataframe(df.head())

    # ── 2. EXPLORATION ───────────────────────────────────────────
    st.subheader("2. Exploration")
    col1, col2 = st.columns(2)
    with col1:
        st.write("**Dimensions :**", df.shape)
        st.write("**Types des colonnes :**")
        st.dataframe(df.dtypes.rename("Type"))
    with col2:
        st.write("**Statistiques descriptives :**")
        st.dataframe(df.describe())

    # Mode
    st.write("**Mode de chaque colonne :**")
    st.dataframe(df.mode().iloc[0].rename("Mode"))

    # ── 3. NETTOYAGE ─────────────────────────────────────────────
    st.subheader("3. Nettoyage — Valeurs Manquantes")
    missing = df.isnull().sum()
    missing = missing[missing > 0]

    if missing.empty:
        st.success("✅ Aucune valeur manquante !")
    else:
        st.warning(f"⚠️ {missing.sum()} valeurs manquantes détectées")
        st.dataframe(missing.rename("Valeurs manquantes"))

        strategie = st.selectbox("Stratégie de nettoyage", [
            "Supprimer les lignes",
            "Remplacer par la moyenne",
            "Remplacer par la médiane",
            "Remplacer par le mode"
        ])

        if st.button("Appliquer le nettoyage"):
            num_cols = df.select_dtypes(include=np.number).columns
            if strategie == "Supprimer les lignes":
                df = df.dropna()
            elif strategie == "Remplacer par la moyenne":
                df[num_cols] = df[num_cols].fillna(df[num_cols].mean())
            elif strategie == "Remplacer par la médiane":
                df[num_cols] = df[num_cols].fillna(df[num_cols].median())
            elif strategie == "Remplacer par le mode":
                df = df.fillna(df.mode().iloc[0])
            st.session_state.df = df
            st.success("✅ Nettoyage appliqué !")

    # ── 4. NORMALISATION ─────────────────────────────────────────
    st.subheader("4. Normalisation")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    methode = st.selectbox("Méthode de normalisation", [
        "Min-Max Scaling",
        "Standardisation (Z-score)"
    ])

    if st.button("Appliquer la normalisation"):
        if methode == "Min-Max Scaling":
            scaler = MinMaxScaler()
        else:
            scaler = StandardScaler()
        df[num_cols] = scaler.fit_transform(df[num_cols])
        st.session_state.df = df
        st.success(f"✅ Normalisation '{methode}' appliquée !")
        st.dataframe(df.head())

    # ── 5. VISUALISATION ─────────────────────────────────────────
    st.subheader("5. Visualisation")
    num_cols = df.select_dtypes(include=np.number).columns.tolist()

    # Tous les boxplots sur un seul graphique
    st.write("**Boxplots de toutes les colonnes numériques**")
    fig, ax = plt.subplots(figsize=(7, 3))
    df[num_cols].boxplot(ax=ax)
    ax.set_title("Boxplots")
    ax.tick_params(axis='x', rotation=45)
    plt.tight_layout()
    st.pyplot(fig)

    # Boxplot individuel
    st.write("**Boxplot individuel**")
    col_b = st.selectbox("Colonne pour Boxplot", num_cols, key="boxplot")
    fig2, ax2 = plt.subplots(figsize=(4, 3))
    sns.boxplot(y=df[col_b], ax=ax2, color="steelblue")
    ax2.set_title(f"Boxplot — {col_b}")
    plt.tight_layout()
    st.pyplot(fig2)

    # Scatter Plot
    st.write("**Scatter Plot**")
    col_x = st.selectbox("Axe X", num_cols, key="scatter_x")
    col_y = st.selectbox("Axe Y", num_cols, key="scatter_y")
    fig3, ax3 = plt.subplots(figsize=(4, 3))
    ax3.scatter(df[col_x], df[col_y], alpha=0.6, color="coral", s=20)
    ax3.set_xlabel(col_x)
    ax3.set_ylabel(col_y)
    ax3.set_title(f"Scatter — {col_x} vs {col_y}")
    plt.tight_layout()
    st.pyplot(fig3)