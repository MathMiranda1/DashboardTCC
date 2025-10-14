import streamlit as st
import database as db
from pages.homepage import show_homepage
from pages.auth import show_admin_login, safe_logout
from pages.prediction import show_prediction_form  # ← USA SEU ARQUIVO ATUAL
from pages.dashboard import show_admin_dashboard
from utils.styles import load_css
from utils.model_utils import carregar_modelo

# Configuração da página
st.set_page_config(
    page_title="Sistema de Análise de Risco de Evasão",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Ocultar navegação automática do Streamlit
st.set_option("client.showSidebarNavigation", False)

# Carregar estilos e conectar ao banco
load_css()
engine = db.get_connection()

if engine:
    db.init_db(engine)
else:
    st.error("❌ Falha na conexão com o banco de dados.")
    st.stop()

# Carregar modelo (o prediction.py vai usar o modelo correto internamente)
modelo = carregar_modelo()

# Inicializar estado
if "user_type" not in st.session_state:
    st.session_state.user_type = None


def main():
    if st.session_state.user_type is None:
        show_homepage()

    elif st.session_state.user_type == "admin_login":
        show_admin_login()

    elif st.session_state.user_type == "student":
        # ESTUDANTE - Usar prediction.py (ML puro)
        with st.sidebar:
            st.write(f"👋 **{st.session_state.name}**")
            if st.button("🏠 Voltar ao Início", use_container_width=True):
                st.session_state.user_type = None
                st.session_state.authentication_status = None
                st.session_state.name = None
                st.session_state.username = None
                st.rerun()

        # Chama a função do prediction.py
        show_prediction_form(modelo, engine)

    elif st.session_state.user_type == "admin":
        # ADMINISTRADOR
        with st.sidebar:
            st.write(f"👋 **{st.session_state.name}**")

            page = st.radio(
                "📍 Navegação:",
                [
                    "📈 Dashboard Administrativo",
                    "🎯 Testar Predição",
                ],
            )

            st.markdown("---")

            col1, col2 = st.columns(2)
            with col1:
                if st.button("🏠 Início", use_container_width=True):
                    st.session_state.user_type = None
                    st.session_state.authentication_status = None
                    st.session_state.name = None
                    st.session_state.username = None
                    st.rerun()

            with col2:
                if st.button("🚪 Logout", use_container_width=True):
                    safe_logout()
                    st.rerun()

        # Renderizar página
        if page == "🎯 Testar Predição":
            show_prediction_form(modelo, engine)
        else:  # Dashboard
            show_admin_dashboard(engine)


if __name__ == "__main__":
    main()
