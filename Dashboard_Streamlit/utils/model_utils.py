import streamlit as st
from pycaret.classification import load_model


@st.cache_resource
def carregar_modelo():
    """Carrega o modelo de Machine Learning"""
    try:
        return load_model("../models/modelo_evasao")
    except:
        # Modelo antigo não encontrado - retornar None sem mostrar erro
        return None
