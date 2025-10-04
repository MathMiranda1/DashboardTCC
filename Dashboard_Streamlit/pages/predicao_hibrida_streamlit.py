"""
INTEGRAÇÃO DO SISTEMA HÍBRIDO COM STREAMLIT - VERSÃO COMPLETA
=============================================================

Formulário idêntico ao sistema antigo, mas com sistema híbrido por trás
"""

import streamlit as st
import pandas as pd
import plotly.graph_objects as go
import plotly.express as px
import sys
import os
from datetime import datetime

# Adicionar pasta utils ao path
current_dir = os.path.dirname(os.path.abspath(__file__))
utils_path = os.path.join(current_dir, "..", "utils")
utils_path = os.path.abspath(utils_path)

if utils_path not in sys.path:
    sys.path.insert(0, utils_path)

from sistema_hibrido_evasao import SistemaHibridoEvasao


def pagina_predicao_hibrida():
    """
    Página principal de predição híbrida - formulário idêntico ao antigo
    """
    # Inicializar sistema híbrido
    if "sistema_hibrido" not in st.session_state:
        try:
            caminho_modelo = os.path.join(
                os.path.dirname(__file__),
                "..",
                "..",
                "Dados_e_Notebook",
                "modelo_evasao_NOVO.pkl",
            )
            st.session_state.sistema_hibrido = SistemaHibridoEvasao(caminho_modelo)
            st.sidebar.success("✅ Sistema híbrido carregado!")
        except Exception as e:
            st.error(f"❌ Erro ao carregar sistema: {e}")
            return

    sistema = st.session_state.sistema_hibrido

    # Título principal
    st.title("🎓 Sistema de Análise de Risco de Evasão Universitária")
    st.markdown(
        "Esta ferramenta utiliza um **sistema híbrido** (ML 30% + Regras 70%) para identificar estudantes com maior propensão a **pensar em trancar disciplinas, período ou abandonar o curso**."
    )

    with st.form(key="formulario_evasao_hibrido"):

        # ===== INFORMAÇÕES BÁSICAS =====
        st.markdown("### 📋 Informações Básicas")

        col1, col2, col3 = st.columns(3)

        with col1:
            # Mapeamento: Frontend (Área) → Backend (Curso específico)
            areas_cursos = {
                "Ciências Exatas e da Terra": "Ciência e Tecnologia",
                "Engenharias": "Engenharia Civil",
                "Ciências Humanas": "Pedagogia",
                "Ciências Sociais Aplicadas": "Licenciatura em Computação e Informática",
                "Tecnologia da Informação": "Sistema de Informação",
                "Outro/Não informado": "",
            }

            area_selecionada = st.selectbox(
                "**Área de Conhecimento:** *(apenas informativo)*",
                list(areas_cursos.keys()),
                help="Este campo é coletado mas não influencia a predição do modelo",
            )
            curso = areas_cursos[area_selecionada]

        with col2:
            semestre_ingresso_texto = st.text_input(
                "**Semestre de Ingresso:** *(apenas informativo)*",
                value="2025.1",
                placeholder="Ex: 2025.1, 2024.2",
                help="Campo coletado mas não usado na predição",
            )

        with col3:
            identificacao_curso = st.selectbox(
                "**7. Você se identifica com a área de conhecimento do seu curso?**",
                ("Sim", "Não, mas quero concluir", "Não, não sei se concluirei"),
            )

        # ===== TRANSPORTE E LOGÍSTICA =====
        st.markdown("---")
        st.markdown("### 🚗 Transporte e Logística")

        col1, col2, col3 = st.columns(3)

        with col1:
            tipo_transporte = st.selectbox(
                "**1. Como é seu deslocamento até a universidade?**",
                ["Carro", "Moto", "Ônibus", "A pé", "Outro"],
            )

            propriedade_transporte = st.selectbox(
                "**2. Com relação ao transporte do item anterior, ele é:**",
                [
                    "Próprio",
                    "Cedido",
                    "Público(gratuito)",
                    "Particular(táxi/moto-táxi)",
                    "Não se aplica",
                ],
            )

        with col2:
            barreira_transporte = st.selectbox(
                "**3. O transporte representa uma barreira/dificuldade para frequentar a universidade?**",
                ["Sim, sempre", "Sim, às vezes", "Não, mas já foi", "Não, nunca foi"],
            )

            mora_angicos = st.selectbox(
                "**4. Você mora na cidade onde estuda?**",
                ("Sim", "Não", "Durmo nos dias de aula e atividades"),
            )

        with col3:
            tempo_deslocamento = st.text_input(
                "**5. Quanto tempo você demora para se deslocar até o campus (ida e volta):** *(apenas informativo)*",
                placeholder="Ex: 30min, 1h30min",
                help="Esta informação é coletada mas não é usada pelo modelo",
            )

            acessibilidade = st.selectbox(
                "**8. Como você considera a acessibilidade do Campus?**",
                ("Adequada", "Inadequada"),
            )

        # ===== EXPERIÊNCIAS NO CAMPUS =====
        st.markdown("---")
        st.markdown("### 🏫 Experiências no Campus")
        st.markdown(
            "**6. Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:**"
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            preconceito_cor = st.checkbox("Cor de pele")
            preconceito_financeiro = st.checkbox("Condição financeira")
            preconceito_aparencia = st.checkbox("Aparência")
        with col2:
            preconceito_deficiencia = st.checkbox("Deficiência")
            preconceito_aprendizado = st.checkbox("Dificuldade de aprendizado")
            preconceito_genero = st.checkbox("Gênero")
        with col3:
            preconceito_curso = st.checkbox("Curso")
            preconceito_idade = st.checkbox("Idade")
            preconceito_nao = st.checkbox("Não sofri preconceito")

        # ===== VIDA ACADÊMICA =====
        st.markdown("---")
        st.markdown("### 📚 Vida Acadêmica")

        tempo_estudo = st.selectbox(
            "**9. Em relação ao tempo necessário como discente para dedicar no estudo:**",
            [
                "É suficiente",
                "É insuficiente, mas desempenho a maioria das atividades",
                "É insuficiente, mas só realizo as atividades obrigatórias",
                "É insuficiente e não consigo realizar as atividades obrigatórias",
            ],
        )

        # ===== TRABALHO E VIDA PESSOAL =====
        st.markdown("---")
        st.markdown("### 💼 Trabalho e Vida Pessoal")

        col1, col2, col3 = st.columns(3)

        with col1:
            trabalha = st.selectbox(
                "**10. Você trabalha?**",
                [
                    "Sim, tenho empresa própria / sou autônomo / profissional liberal",
                    "Sim, com emprego formal",
                    "Sim, com trabalho temporário",
                    "Sim, com emprego informal",
                    "Sou bolsista da Universidade",
                    "Estou em estágio remunerado",
                    "Não trabalho ainda",
                ],
            )

        with col2:
            horarios_trabalho = st.selectbox(
                "**11. Se você trabalha, em quais horários?**",
                [
                    "Tempo integral ou dois turnos",
                    "Tempo parcial ou um turno",
                    "Horário corrido ou 6 horas diárias",
                    "Em regime de escala ou plantão",
                    "Sem dias e horários fixos",
                    "Não se aplica",
                ],
            )

        with col3:
            estado_civil = st.selectbox(
                "**12. É casado(a) / está em união estável:**", ["Não", "Sim"]
            )

        col1, col2 = st.columns(2)

        with col1:
            tem_filhos = st.selectbox("**13. Tem filhos?**", ["Não", "Sim"])

            if tem_filhos == "Sim":
                qtd_filhos = st.number_input(
                    "Quantos filhos?", min_value=1, max_value=10, value=1
                )
            else:
                qtd_filhos = 0

        with col2:
            contribuicao_financeira = st.selectbox(
                "**14. Você contribui para o sustento financeiro da família:**",
                [
                    "Sim, sou o único com renda",
                    "Sim, sou a principal",
                    "Sim, mas não sou o principal",
                    "Não contribuo",
                ],
            )

        # ===== QUESTÕES ABERTAS =====
        st.markdown("---")
        st.markdown("### ✍️ Questões Abertas")

        col1, col2 = st.columns(2)

        with col1:
            dificuldades = st.text_area(
                "**15. Cite as principais dificuldades para sua permanência no Curso:** *(apenas informativo)*",
                placeholder="Descreva as principais dificuldades...",
                help="Este campo é coletado mas não influencia a predição do modelo",
            )

        with col2:
            ja_pensou_evasao = st.text_area(
                "**🎯 16. Já pensou em trancar disciplinas, período ou abandonar o curso? Se sim, por quê?**",
                placeholder="Descreva se já pensou em evasão e os motivos...",
                help="Esta é a pergunta principal da pesquisa. Seja sincero(a) em sua resposta.",
            )

        # ===== BOTÃO DE SUBMISSÃO =====
        st.markdown("---")
        submit_button = st.form_submit_button(
            label="🔍 Analisar Risco de Evasão (Sistema Híbrido)",
            type="primary",
            use_container_width=True,
        )

    # ===== PROCESSAMENTO =====
    if submit_button:
        # Converter tempo de deslocamento para número
        import re

        tempo_numerico = 0
        if tempo_deslocamento:
            numeros = re.findall(r"\d+", tempo_deslocamento)
            if numeros:
                tempo_numerico = int(numeros[0])

        # Converter semestre para número
        try:
            if "." in semestre_ingresso_texto:
                ano, periodo = semestre_ingresso_texto.split(".")
                semestre_numerico = ((int(ano) - 2010) * 2) + int(periodo)
            else:
                semestre_numerico = 31  # Default 2025.1
        except:
            semestre_numerico = 31

        # Mapear dados do formulário para o sistema híbrido
        dados_hibrido = mapear_para_hibrido(
            tipo_transporte,
            propriedade_transporte,
            mora_angicos,
            acessibilidade,
            tempo_estudo,
            trabalha,
            horarios_trabalho,
            estado_civil,
            tem_filhos,
            contribuicao_financeira,
        )

        # Fazer predição com sistema híbrido
        with st.spinner("Analisando dados com sistema híbrido..."):
            resultado = sistema.predizer(dados_hibrido, detalhado=True)

        # SALVAR NO BANCO DE DADOS
        try:
            import database as db

            engine = db.get_connection()

            # Análise simples da resposta sobre evasão
            pensou_evasao_bool = len(ja_pensou_evasao.strip()) > 20 and any(
                palavra in ja_pensou_evasao.lower()
                for palavra in ["sim", "já", "pensei", "trancar", "abandonar"]
            )

            data_to_save = pd.DataFrame(
                [
                    {
                        "timestamp": datetime.now(),
                        "user_submitting": st.session_state.get("username", "student"),
                        "curso": curso,
                        "semestre_ingresso": semestre_numerico,
                        "identificacao_curso": identificacao_curso,
                        "tipo_transporte": tipo_transporte,
                        "propriedade_transporte": propriedade_transporte,
                        "barreira_transporte": barreira_transporte,
                        "mora_na_cidade": mora_angicos,
                        "tempo_deslocamento": tempo_numerico,
                        "acessibilidade_campus": acessibilidade,
                        "preconceito_cor": preconceito_cor,
                        "preconceito_financeiro": preconceito_financeiro,
                        "preconceito_aparencia": preconceito_aparencia,
                        "preconceito_deficiencia": preconceito_deficiencia,
                        "preconceito_aprendizado": preconceito_aprendizado,
                        "preconceito_genero": preconceito_genero,
                        "preconceito_curso": preconceito_curso,
                        "preconceito_idade": preconceito_idade,
                        "preconceito_nao": preconceito_nao,
                        "tempo_estudo": tempo_estudo,
                        "situacao_trabalho": trabalha,
                        "horarios_trabalho": horarios_trabalho,
                        "estado_civil": estado_civil == "Sim",
                        "tem_filhos": tem_filhos == "Sim",
                        "qtd_filhos": qtd_filhos,
                        "contribuicao_financeira": contribuicao_financeira,
                        "dificuldades_permanencia": dificuldades,
                        "resposta_completa_evasao": ja_pensou_evasao,
                        "pensou_evasao_real": 1 if pensou_evasao_bool else 0,
                        "prediction_label": 1 if resultado["score_final"] > 0.5 else 0,
                        "prediction_score": float(resultado["score_final"]),
                    }
                ]
            )

            if db.save_submission(engine, data_to_save):
                st.success("✅ Análise registrada no banco de dados!")
        except Exception as e:
            st.warning(f"⚠️ Dados salvos com avisos: {str(e)[:100]}")

        # Exibir resultados
        exibir_resultados_hibridos(
            resultado,
            ja_pensou_evasao,
            barreira_transporte,
            identificacao_curso,
            tempo_estudo,
            trabalha,
            horarios_trabalho,
            tem_filhos,
            contribuicao_financeira,
            preconceito_cor,
            preconceito_financeiro,
            preconceito_aparencia,
            preconceito_deficiencia,
            preconceito_aprendizado,
            preconceito_genero,
            preconceito_curso,
            preconceito_idade,
            preconceito_nao,
        )


def mapear_para_hibrido(
    tipo_transporte,
    propriedade_transporte,
    mora_angicos,
    acessibilidade,
    tempo_estudo,
    trabalha,
    horarios_trabalho,
    estado_civil,
    tem_filhos,
    contribuicao_financeira,
):
    """
    Mapeia dados do formulário para o formato do sistema híbrido
    """
    # Mapeamento de transporte
    mapa_transporte = {
        "Carro": "Carro próprio",
        "Moto": "Moto própria",
        "Ônibus": "Ônibus",
        "A pé": "A pé",
        "Outro": "Outro",
    }

    # Mapeamento de propriedade
    mapa_propriedade = {
        "Próprio": "Próprio",
        "Cedido": "Próprio",  # Considerar cedido como próprio para o sistema
        "Público(gratuito)": "Público",
        "Particular(táxi/moto-táxi)": "Dependo de terceiros",
        "Não se aplica": "Próprio",
    }

    # Mapeamento mora em Angicos
    mapa_mora = {
        "Sim": "Sim",
        "Não": "Não",
        "Durmo nos dias de aula e atividades": "Não",
    }

    # Mapeamento de acessibilidade
    mapa_acessibilidade = {"Adequada": "Boa", "Inadequada": "Ruim"}

    # Mapeamento de trabalho
    trabalha_sim_nao = "Não" if trabalha == "Não trabalho ainda" else "Sim"

    dados = {
        "Como é o seu deslocamento até a universidade?": mapa_transporte.get(
            tipo_transporte, tipo_transporte
        ),
        "Com relação ao transporte do item anterior, ele é:": mapa_propriedade.get(
            propriedade_transporte, propriedade_transporte
        ),
        "Você mora em Angicos?": mapa_mora.get(mora_angicos, mora_angicos),
        "Como você considera a acessibilidade do Campus?": mapa_acessibilidade.get(
            acessibilidade, acessibilidade
        ),
        "Em relação ao tempo necessário como discente para dedicar no estudo?": tempo_estudo,
        "Você trabalha?": trabalha_sim_nao,
        "Se você trabalha, em quais horários?": horarios_trabalho,
        "É casado(a)/está em união estável?": estado_civil,
        "Tem filhos?": tem_filhos,
        "Você contribui para o sustento financeiro da família?": contribuicao_financeira,
    }

    return dados


def exibir_resultados_hibridos(
    resultado,
    ja_pensou_evasao,
    barreira_transporte,
    identificacao_curso,
    tempo_estudo,
    trabalha,
    horarios_trabalho,
    tem_filhos,
    contribuicao_financeira,
    preconceito_cor,
    preconceito_financeiro,
    preconceito_aparencia,
    preconceito_deficiencia,
    preconceito_aprendizado,
    preconceito_genero,
    preconceito_curso,
    preconceito_idade,
    preconceito_nao,
):
    """
    Exibe os resultados do sistema híbrido
    """
    st.markdown("---")
    st.header("📊 Resultado da Análise Híbrida")

    st.info(
        "**Sistema Híbrido:** Combina Machine Learning (30%) + Regras Baseadas em Literatura (70%)"
    )

    # Card principal
    col_a, col_b, col_c = st.columns([2, 3, 3])

    with col_a:
        st.metric(label="Categoria de Risco", value=resultado["categoria_risco"])

    with col_b:
        st.metric(
            label="Probabilidade de Evasão", value=resultado["probabilidade_evasao"]
        )

    with col_c:
        if resultado["categoria_risco"] == "BAIXO":
            st.success(f"{resultado['cor']} {resultado['recomendacao']}")
        elif resultado["categoria_risco"] == "MODERADO":
            st.warning(f"{resultado['cor']} {resultado['recomendacao']}")
        elif resultado["categoria_risco"] == "ALTO":
            st.warning(f"{resultado['cor']} {resultado['recomendacao']}")
        else:  # CRÍTICO
            st.error(f"{resultado['cor']} {resultado['recomendacao']}")

    # Gauge
    fig_gauge = criar_gauge_risco(resultado["score_final"])
    st.plotly_chart(fig_gauge, use_container_width=True)

    # Decomposição
    st.subheader("🔬 Decomposição do Score")
    col_ml, col_regras = st.columns(2)

    with col_ml:
        st.metric(
            label="📊 Machine Learning",
            value=f"{resultado['score_ml']*100:.1f}%",
            help="Contribuição do modelo ML treinado (peso: 30%)",
        )

    with col_regras:
        st.metric(
            label="📖 Regras Especializadas",
            value=f"{resultado['score_regras']*100:.1f}%",
            help="Contribuição das regras baseadas em literatura científica (peso: 70%)",
        )

    # Fatores de risco do sistema híbrido
    if "fatores_risco" in resultado and resultado["fatores_risco"]:
        st.subheader("⚠️ Principais Fatores de Risco (Sistema Híbrido)")

        fatores_df = pd.DataFrame(resultado["fatores_risco"])

        fig_fatores = px.bar(
            fatores_df,
            x="severidade",
            y="fator",
            orientation="h",
            title="Intensidade dos Fatores de Risco",
            labels={"severidade": "Severidade", "fator": "Fator"},
            color="severidade",
            color_continuous_scale="Reds",
        )
        fig_fatores.update_layout(height=300)
        st.plotly_chart(fig_fatores, use_container_width=True)

        st.dataframe(
            fatores_df[["fator", "impacto"]], hide_index=True, use_container_width=True
        )

    # Outros fatores identificados
    st.markdown("---")
    st.subheader("🔍 Outros Fatores Identificados no Formulário")

    fatores_adicionais = []

    if barreira_transporte in ["Sim, sempre", "Sim, às vezes"]:
        fatores_adicionais.append("🚗 Dificuldades de transporte")
    if identificacao_curso != "Sim":
        fatores_adicionais.append("📚 Baixa identificação com o curso")
    if "insuficiente" in tempo_estudo.lower() and "não consigo" in tempo_estudo.lower():
        fatores_adicionais.append("⏰ Tempo muito insuficiente para estudos")
    elif "insuficiente" in tempo_estudo.lower():
        fatores_adicionais.append("⏰ Tempo insuficiente para estudos")
    if (
        trabalha != "Não trabalho ainda"
        and horarios_trabalho == "Tempo integral ou dois turnos"
    ):
        fatores_adicionais.append("💼 Trabalho em período integral")
    if tem_filhos == "Sim":
        fatores_adicionais.append("👶 Responsabilidades familiares (filhos)")
    if any(
        [
            preconceito_cor,
            preconceito_financeiro,
            preconceito_aparencia,
            preconceito_deficiencia,
            preconceito_aprendizado,
            preconceito_genero,
            preconceito_curso,
            preconceito_idade,
        ]
    ):
        fatores_adicionais.append("⚠️ Experiência de preconceito/violência")
    if contribuicao_financeira in [
        "Sim, sou o único com renda",
        "Sim, sou a principal",
    ]:
        fatores_adicionais.append("💰 Responsabilidade financeira principal")

    if fatores_adicionais:
        for fator in fatores_adicionais:
            st.write(f"• {fator}")
    else:
        st.info("Nenhum fator adicional significativo identificado")

    # Resposta sobre evasão
    if ja_pensou_evasao.strip():
        st.markdown("---")
        with st.expander("📝 Resposta do Estudante sobre Evasão"):
            st.write(ja_pensou_evasao)

    # Relatório completo
    with st.expander("📄 Relatório Técnico Completo"):
        try:
            relatorio = st.session_state.sistema_hibrido.gerar_relatorio(resultado)
            st.code(relatorio, language=None)
        except Exception as e:
            st.error(f"Erro ao gerar relatório: {e}")
            st.write("**Resumo alternativo:**")
            st.json(resultado)

    # Informações sobre o sistema
    with st.expander("ℹ️ Sobre o Sistema Híbrido"):
        st.markdown(
            """
        **Por que Sistema Híbrido?**
        
        Este sistema combina o melhor de dois mundos:
        
        1. **Machine Learning (30%)**: Modelo treinado com 264 casos reais da UFERSA
        2. **Regras Especializadas (70%)**: Baseadas em 40+ anos de pesquisa científica
        
        **Vantagens:**
        - ✅ Mais robusto que ML puro (dataset pequeno)
        - ✅ Explica QUAIS fatores causam risco
        - ✅ Baseado em literatura consolidada (Tinto, Bean, Cabrera)
        - ✅ Recomendações acionáveis para intervenções
        
        **Regras utilizadas:**
        1. Transporte Difícil (15%)
        2. Trabalho Conflitante (20%)
        3. Responsabilidades Familiares (18%)
        4. Falta de Tempo (17%)
        5. Distância do Campus (12%)
        6. Acessibilidade Ruim (10%)
        7. Sobrecarga Total (8%)
        """
        )


def criar_gauge_risco(score: float) -> go.Figure:
    """Cria gauge de risco"""
    fig = go.Figure(
        go.Indicator(
            mode="gauge+number",
            value=score * 100,
            domain={"x": [0, 1], "y": [0, 1]},
            title={"text": "Risco de Evasão (%)", "font": {"size": 24}},
            gauge={
                "axis": {"range": [None, 100], "tickwidth": 1, "tickcolor": "darkblue"},
                "bar": {"color": "darkblue"},
                "bgcolor": "white",
                "borderwidth": 2,
                "bordercolor": "gray",
                "steps": [
                    {"range": [0, 35], "color": "#22c55e"},
                    {"range": [35, 55], "color": "#eab308"},
                    {"range": [55, 75], "color": "#f97316"},
                    {"range": [75, 100], "color": "#ef4444"},
                ],
                "threshold": {
                    "line": {"color": "red", "width": 4},
                    "thickness": 0.75,
                    "value": score * 100,
                },
            },
        )
    )

    fig.update_layout(
        paper_bgcolor="white", height=300, font={"color": "darkblue", "family": "Arial"}
    )

    return fig


if __name__ == "__main__":
    pagina_predicao_hibrida()
