import streamlit as st
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.neighbors import KNeighborsClassifier
from sklearn.tree import DecisionTreeClassifier
from sklearn.naive_bayes import GaussianNB
from sklearn.svm import SVC
from sklearn.metrics import (confusion_matrix, ConfusionMatrixDisplay,
                             accuracy_score, precision_score,
                             recall_score, f1_score)

def show_classification():
    st.header("🤖 Volet 3 — Classification")

    if st.session_state.df is None:
        st.warning("⚠️ Veuillez d'abord charger un dataset dans le Volet 1.")
        return

    df = st.session_state.df.copy()

    # ── 1. CHOIX DE LA COLONNE CIBLE ─────────────────────────────
    st.subheader("1. Partitionnement")
    target_col = st.selectbox("Colonne cible (Y)", df.columns.tolist())
    test_size = st.slider("Taille du jeu de test (%)", 10, 40, 20)

    X = df.drop(columns=[target_col])
    X = X.select_dtypes(include=np.number)
    y = df[target_col]

    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=test_size/100, random_state=42
    )

    st.info(f"Train : {len(X_train)} lignes | Test : {len(X_test)} lignes")

    # ── 2. CHOIX DU MODELE ───────────────────────────────────────
    st.subheader("2. Choix du Modèle")
    algo = st.selectbox("Algorithme", [
        "K-Nearest Neighbors (KNN)",
        "Decision Tree",
        "Naive Bayes",
        "SVM"
    ])

    # Hyperparamètres
    if algo == "K-Nearest Neighbors (KNN)":
        k = st.slider("Nombre de voisins (k)", 1, 15, 5)
        model = KNeighborsClassifier(n_neighbors=k)
    elif algo == "Decision Tree":
        max_depth = st.slider("Profondeur maximale", 1, 20, 5)
        model = DecisionTreeClassifier(max_depth=max_depth, random_state=42)
    elif algo == "Naive Bayes":
        model = GaussianNB()
    else:
        C = st.slider("Paramètre C", 0.1, 10.0, 1.0)
        model = SVC(C=C, random_state=42)

    # ── 3. ENTRAINEMENT ET EVALUATION ────────────────────────────
    if st.button("Lancer l'entraînement"):
        model.fit(X_train, y_train)
        y_pred = model.predict(X_test)

        # Métriques
        acc  = accuracy_score(y_test, y_pred)
        prec = precision_score(y_test, y_pred, average='weighted', zero_division=0)
        rec  = recall_score(y_test, y_pred, average='weighted', zero_division=0)
        f1   = f1_score(y_test, y_pred, average='weighted', zero_division=0)

        st.subheader("3. Résultats")
        col1, col2, col3, col4 = st.columns(4)
        col1.metric("Accuracy",  f"{acc:.4f}")
        col2.metric("Precision", f"{prec:.4f}")
        col3.metric("Recall",    f"{rec:.4f}")
        col4.metric("F1-Score",  f"{f1:.4f}")

        # Matrice de confusion
        st.subheader("4. Matrice de Confusion")
        cm = confusion_matrix(y_test, y_pred)
        fig, ax = plt.subplots()
        im = ax.imshow(cm, interpolation='nearest', cmap="Blues")
        ax.set_title(f"Matrice de Confusion — {algo}")
        tick_marks = np.arange(len(model.classes_))
        ax.set_xticks(tick_marks)
        ax.set_yticks(tick_marks)
        ax.set_xticklabels(model.classes_, rotation=45, ha='right')
        ax.set_yticklabels(model.classes_)
        ax.set_ylabel('Vraie classe')
        ax.set_xlabel('Classe prédite')
        for i in range(cm.shape[0]):
            for j in range(cm.shape[1]):
                ax.text(j, i, str(cm[i, j]), ha='center', va='center', color='black')
        st.pyplot(fig)