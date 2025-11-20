import streamlit as st
import streamlit_authenticator as stauth

# Importando as funções do database.py que está na raiz
from database import get_connection, init_users_db, get_users_credentials


def setup_admin_auth():
    """Configura autenticação para administradores usando o Banco de Dados"""

    # 1. Conectar ao banco
    engine = get_connection()

    if engine is None:
        st.error("Falha na conexão com o banco de dados para autenticação.")
        # Retorna None para tratar o erro posteriormente
        return None

    # 2. Inicializar a tabela de usuários (cria admin padrão se necessário)
    init_users_db(engine)

    # 3. Buscar credenciais do banco (substitui o hardcode antigo)
    credentials = get_users_credentials(engine)

    # 4. Retornar o objeto autenticador
    return stauth.Authenticate(
        credentials,
        "evasao_admin_cookie",
        "admin_key_secure_123",
        cookie_expiry_days=30,
    )


def show_admin_login():
    """Tela de login administrativo"""
    st.markdown("## 🔐 Acesso Administrativo")

    # Limpa status anterior se houver inconsistência
    if st.session_state.get("authentication_status") == True:
        st.session_state.authentication_status = None

    authenticator = setup_admin_auth()

    # Se o banco falhou, para a execução aqui
    if authenticator is None:
        st.stop()

    name, authentication_status, username = authenticator.login("Login", "main")

    col1, col2, col3 = st.columns([1, 1, 1])
    with col2:
        if st.button("⬅️ Voltar ao Início"):
            st.session_state.user_type = None
            st.rerun()

    if authentication_status == False:
        st.error("❌ Usuário/senha incorretos")
    elif authentication_status == None:
        st.info("👋 Digite suas credenciais administrativas")
        # Removido o texto hardcoded da senha por segurança.
        # Na primeira vez, use: admin / admin456
    else:
        st.session_state.user_type = "admin"
        st.session_state.authentication_status = True
        st.session_state.name = name
        st.session_state.username = username
        st.rerun()


def safe_logout():
    """Função de logout segura que não gera erro de cookie"""
    try:
        authenticator = setup_admin_auth()
        if authenticator:
            authenticator.logout("Logout", "main")
    except (KeyError, AttributeError):
        # Se der erro no logout do cookie, apenas passamos
        pass

    # Sempre limpar o session state independentemente do cookie
    st.session_state.user_type = None
    st.session_state.authentication_status = None
    st.session_state.name = None
    st.session_state.username = None
