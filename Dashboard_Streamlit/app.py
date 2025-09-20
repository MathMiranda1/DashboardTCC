import streamlit as st
import database as db
from pages.homepage import show_homepage
from pages.auth import show_admin_login, setup_admin_auth, safe_logout
from pages.prediction import show_prediction_form
from pages.dashboard import show_admin_dashboard
from utils.styles import load_css
from utils.model_utils import carregar_modelo

# Configuração da página
st.set_page_config(
    page_title="Sistema de Análise de Risco de Evasão",
    page_icon="🎓",
    layout="wide",
    initial_sidebar_state="expanded",  # Sidebar sempre visível
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

modelo = carregar_modelo()
if modelo is None:
    st.error("❌ Modelo de Machine Learning não encontrado.")
    st.stop()

# Inicializar estado
if "user_type" not in st.session_state:
    st.session_state.user_type = None


# Roteamento
def main():
    if st.session_state.user_type is None:
        show_homepage()

    elif st.session_state.user_type == "admin_login":
        show_admin_login()

    elif st.session_state.user_type == "student":
        with st.sidebar:
            st.write(f"👋 **{st.session_state.name}**!")
            if st.button("🏠 Início"):
                # Limpar TODAS as variáveis do session state
                st.session_state.user_type = None
                st.session_state.authentication_status = None
                st.session_state.name = None
                st.session_state.username = None
                st.rerun()
        show_prediction_form(modelo, engine)

    elif st.session_state.user_type == "admin":
        with st.sidebar:
            st.write(f"👋 **{st.session_state.name}**!")
            page = st.radio("Navegação:", ["📊 Análise", "🔑 Dashboard"])

            # Botão de logout seguro
            if st.button("🚪 Logout"):
                safe_logout()
                st.rerun()

            if st.button("🏠 Início"):
                # Limpar TODAS as variáveis do session state
                st.session_state.user_type = None
                st.session_state.authentication_status = None
                st.session_state.name = None
                st.session_state.username = None
                st.rerun()

        if page == "📊 Análise":
            show_prediction_form(modelo, engine)
        else:
            show_admin_dashboard(engine)


if __name__ == "__main__":
    main()
