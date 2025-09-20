import streamlit as st

def show_homepage():
    """Tela inicial de seleção de perfil"""
    # Header principal
    st.markdown("""
    <div style="text-align: center; padding: 2rem; background: linear-gradient(135deg, #667eea 0%, #764ba2 100%); color: white; border-radius: 15px; margin-bottom: 2rem;">
        <h1>🎓 Sistema de Análise de Risco de Evasão</h1>
        <p style="font-size: 1.2rem; opacity: 0.9;">Ferramenta inteligente para identificação precoce de estudantes com propensão à evasão universitária</p>
    </div>
    """, unsafe_allow_html=True)
    
    # Seleção de perfil
    st.markdown("### Selecione seu perfil de acesso:")
    
    col1, col2, col3 = st.columns([1, 2, 1])
    
    with col2:
        # Card do Estudante
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; text-align: center; border-left: 4px solid #27ae60;">
            <h3>👨‍🎓 Sou Estudante</h3>
            <p>Responda ao questionário de forma anônima e receba uma análise personalizada</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🎓 ACESSAR COMO ESTUDANTE", key="student_btn", use_container_width=True):
            st.session_state.user_type = "student"
            st.session_state.name = "Estudante"
            st.session_state.username = "student"
            st.session_state.authentication_status = True
            st.rerun()
        
        st.markdown("<br>", unsafe_allow_html=True)
        
        # Card do Administrador
        st.markdown("""
        <div style="background: white; padding: 2rem; border-radius: 15px; box-shadow: 0 4px 6px rgba(0,0,0,0.1); margin-bottom: 2rem; text-align: center; border-left: 4px solid #e74c3c;">
            <h3>👨‍💼 Sou Gestor/Coordenador</h3>
            <p>Acesse o dashboard administrativo com dados agregados</p>
        </div>
        """, unsafe_allow_html=True)
        
        if st.button("🔑 ACESSAR COMO ADMINISTRADOR", key="admin_btn", use_container_width=True):
            st.session_state.user_type = "admin_login"
            st.rerun()
    
    # Características do sistema
    st.markdown("---")
    st.markdown("### 🚀 Características do Sistema")
    
    col1, col2, col3, col4 = st.columns(4)
    
    with col1:
        st.markdown("**🧠 IA Avançada**\n\nModelo de ML treinado com dados reais")
    with col2:
        st.markdown("**📊 Análise Personalizada**\n\nRelatórios com percentual de risco")
    with col3:
        st.markdown("**🔒 Seguro**\n\nDados protegidos e anônimos")
    with col4:
        st.markdown("**📈 Dashboard**\n\nMétricas para gestão")