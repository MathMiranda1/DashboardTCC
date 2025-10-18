import streamlit as st
import streamlit_authenticator as stauth


def setup_admin_auth():
    """Configura autenticação para administradores"""
    names = ["Administrador"]
    usernames = ["admin"]
    passwords = ["admin456"]

    hashed_passwords = stauth.Hasher(passwords).generate()

    credentials = {"usernames": {}}
    for username, name, hash_password in zip(usernames, names, hashed_passwords):
        credentials["usernames"][username] = {"name": name, "password": hash_password}

    return stauth.Authenticate(
        credentials, "evasao_admin", "admin_key_12345", cookie_expiry_days=30
    )


def show_admin_login():
    """Tela de login administrativo"""
    st.markdown("## 🔐 Acesso Administrativo")

    if st.session_state.get("authentication_status") == True:
        st.session_state.authentication_status = None

    authenticator = setup_admin_auth()
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
        st.markdown("**Credenciais:** admin / admin456")
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
        authenticator.logout("Logout", "main")
    except (KeyError, AttributeError):
        # Se der erro no logout do cookie, apenas limpar o session state
        pass

    # Sempre limpar o session state independentemente do cookie
    st.session_state.user_type = None
    st.session_state.authentication_status = None
    st.session_state.name = None
    st.session_state.username = None
