import streamlit as st
import database as db
from pages.homepage import show_homepage
from pages.auth import show_admin_login, safe_logout
from pages.predicao_hibrida_streamlit import pagina_predicao_hibrida
from pages.dashboard import show_admin_dashboard
from utils.styles import load_css

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

# Inicializar estado
if "user_type" not in st.session_state:
    st.session_state.user_type = None


def main():
    if st.session_state.user_type is None:
        show_homepage()

    elif st.session_state.user_type == "admin_login":
        show_admin_login()

    elif st.session_state.user_type == "student":
        # ESTUDANTE - Mostrar sidebar mínima
        with st.sidebar:
            st.write(f"👋 **{st.session_state.name}**")
            if st.button("🏠 Voltar ao Início", use_container_width=True):
                st.session_state.user_type = None
                st.session_state.authentication_status = None
                st.session_state.name = None
                st.session_state.username = None
                st.rerun()

        # Mostrar o sistema híbrido para estudantes
        pagina_predicao_hibrida()

    elif st.session_state.user_type == "admin":
        # ADMINISTRADOR - Mostrar sidebar com navegação
        with st.sidebar:
            st.write(f"👋 **{st.session_state.name}**")

            # Menu de navegação do admin
            page = st.radio(
                "📍 Navegação:",
                [
                    "📈 Dashboard Administrativo",
                    "🎯 Sistema Híbrido",
                ],
            )

            st.markdown("---")

            # Botões de ação
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

        # Renderizar página selecionada
        if page == "🎯 Sistema Híbrido":
            pagina_predicao_hibrida()
        else:  # Dashboard Administrativo
            show_admin_dashboard(engine)


if __name__ == "__main__":
    main()
