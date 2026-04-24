import streamlit as st

st.set_page_config(
    page_title="Interface Fouille de Données",
    page_icon="🔍",
    layout="wide"
)

st.title("🔍 Interface Fouille de Données")

# Initialiser le dataset dans la session
if "df" not in st.session_state:
    st.session_state.df = None

# Navigation en 3 onglets
tab1, tab2, tab3 = st.tabs([
    "📊 Volet 1 — Prétraitement",
    "🔵 Volet 2 — Clustering",
    "🤖 Volet 3 — Classification"
])

with tab1:
    from modules.preprocessing import show_preprocessing
    show_preprocessing()

with tab2:
    from modules.clustering import show_clustering
    show_clustering()

with tab3:
    from modules.classification import show_classification
    show_classification()