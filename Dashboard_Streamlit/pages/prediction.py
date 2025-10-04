import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
import database as db


@st.cache_resource
def load_new_model():
    """Carrega o modelo corrigido sem data leakage"""
    try:
        with open("../models/modelo_evasao_SEM_vazamento.pkl", "rb") as f:
            modelo_data = pickle.load(f)
        return modelo_data
    except FileNotFoundError:
        try:
            with open("modelo_evasao_SEM_vazamento.pkl", "rb") as f:
                modelo_data = pickle.load(f)
            return modelo_data
        except FileNotFoundError:
            st.error(
                "❌ Modelo não encontrado! Execute o script de geração do modelo primeiro."
            )
            return None
    except Exception as e:
        st.error(f"❌ Erro ao carregar modelo: {e}")
        return None


def preprocess_data_for_new_model(raw_data, model_data):
    """Preprocessa dados usando o pipeline do novo modelo"""
    try:
        # Criar DataFrame com os dados
        df = pd.DataFrame([raw_data])

        # Aplicar label encoders para variáveis categóricas
        for col, encoder in model_data["label_encoders"].items():
            if col in df.columns:
                if df[col].dtype == "object":
                    try:
                        df[col] = encoder.transform(df[col].astype(str))
                    except ValueError:
                        # Se valor não foi visto no treino, usar o mais frequente
                        df[col] = encoder.transform([encoder.classes_[0]])[0]

        # Garantir ordem correta das colunas
        df = df.reindex(columns=model_data["feature_names"], fill_value=0)

        # Aplicar imputer
        df_imputed = pd.DataFrame(
            model_data["imputer"].transform(df), columns=model_data["feature_names"]
        )

        # Aplicar scaler
        df_scaled = model_data["scaler"].transform(df_imputed)

        return df_scaled
    except Exception as e:
        st.error(f"Erro no preprocessing: {e}")
        return None


def show_prediction_form(modelo_antigo, engine):
    """Formulário de predição com campos originais, usando novo modelo"""

    # Carregar o novo modelo (ignorar o antigo passado como parâmetro)
    modelo_data = load_new_model()
    if modelo_data is None:
        return

    # Info do modelo na sidebar
    st.sidebar.success(
        f"✅ Modelo carregado: AUC = {modelo_data['metricas']['auc_test']:.3f}"
    )
    st.sidebar.info("🔬 Modelo cientificamente validado (sem data leakage)")

    st.title("🎓 Sistema de Análise de Risco de Evasão Universitária")
    st.markdown(
        "Esta ferramenta utiliza um modelo preditivo para identificar estudantes com maior propensão a **pensar em trancar disciplinas, período ou abandonar o curso**."
    )

    with st.form(key="formulario_evasao"):

        # INFORMAÇÕES BÁSICAS
        st.markdown(
            '<div class="section-header">📋 Informações Básicas</div>',
            unsafe_allow_html=True,
        )

        col1, col2, col3 = st.columns(3)
        with col1:
            # Mapeamento: Frontend (Área de Conhecimento) → Modelo (Curso Específico)
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

            # Converter área selecionada para curso específico que o modelo entende
            curso = areas_cursos[area_selecionada]

        with col2:
            semestre_ingresso_texto = st.text_input(
                "**Semestre de Ingresso:** *(apenas informativo)*",
                value="2025.1",
                placeholder="Ex: 2025.1, 2024.2",
                help="Campo coletado mas não usado na predição - Digite no formato YYYY.1 ou YYYY.2",
            )

            # Converter para número mantendo compatibilidade com o banco de dados
            try:
                if "." in semestre_ingresso_texto:
                    ano, periodo = semestre_ingresso_texto.split(".")
                    ano, periodo = int(ano), int(periodo)
                    if periodo in [1, 2] and 2010 <= ano <= 2030:
                        # Converter para sequência numérica começando de 2010 (2010.1 = 1, 2010.2 = 2, etc.)
                        semestre_ingresso = ((ano - 2010) * 2) + periodo
                    else:
                        st.warning(
                            "⚠️ Use formato YYYY.1 ou YYYY.2 (anos entre 2010-2030)"
                        )
                        semestre_ingresso = 31.0  # 2025.1 com nova base
                else:
                    # Se digitou só número, usar como está
                    semestre_ingresso = float(semestre_ingresso_texto)
            except:
                # Se houver erro na conversão, mostrar aviso e usar valor padrão
                st.warning("⚠️ Formato inválido. Usando 2025.1 como padrão.")
                semestre_ingresso = 31.0  # 2025.1 com nova base

        with col3:
            identificacao_curso = st.selectbox(
                "**7. Você se identifica com a área de conhecimento do seu curso?**",
                ("Sim", "Não, mas quero concluir", "Não, não sei se concluirei"),
            )

        # TRANSPORTE E LOGÍSTICA
        st.markdown(
            '<div class="section-header">🚗 Transporte e Logística</div>',
            unsafe_allow_html=True,
        )

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

        # EXPERIÊNCIAS NO CAMPUS
        st.markdown(
            '<div class="section-header">🏫 Experiências no Campus</div>',
            unsafe_allow_html=True,
        )

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

        # VIDA ACADÊMICA
        st.markdown(
            '<div class="section-header">📚 Vida Acadêmica</div>',
            unsafe_allow_html=True,
        )

        tempo_estudo = st.selectbox(
            "**9. Em relação ao tempo necessário como discente para dedicar no estudo:**",
            [
                "É suficiente",
                "É insuficiente, mas desempenho a maioria das atividades",
                "É insuficiente, mas só realizo as atividades obrigatórias",
                "É insuficiente e não consigo realizar as atividades obrigatórias",
            ],
        )

        # TRABALHO E VIDA PESSOAL
        st.markdown(
            '<div class="section-header">💼 Trabalho e Vida Pessoal</div>',
            unsafe_allow_html=True,
        )

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

        # QUESTÕES ABERTAS
        st.markdown(
            '<div class="section-header">✍️ Questões Abertas</div>',
            unsafe_allow_html=True,
        )

        col1, col2 = st.columns(2)
        with col1:
            dificuldades = st.text_area(
                "**15. Cite as principais dificuldades para sua permanência no Curso:** *(apenas informativo)*",
                placeholder="Descreva as principais dificuldades...",
                help="Este campo é coletado mas não influencia a predição do modelo",
            )

        with col2:
            # ===== PERGUNTA PRINCIPAL (TARGET) =====
            ja_pensou_evasao = st.text_area(
                "**🎯 16. Já pensou em trancar disciplinas, período ou abandonar o curso? Se sim, por quê?**",
                placeholder="Descreva se já pensou em evasão e os motivos...",
                help="Esta é a pergunta principal da pesquisa. Seja sincero(a) em sua resposta.",
            )

        # BOTÃO DE SUBMISSÃO
        submit_button = st.form_submit_button(label="📊 Analisar Risco de Evasão")

    # ===== PROCESSAMENTO E RESULTADOS =====
    if submit_button:
        # Processamento do tempo de deslocamento (coletado mas não usado no modelo)
        tempo_numerico = 0
        if tempo_deslocamento:
            numeros = re.findall(r"\d+", tempo_deslocamento)
            if numeros:
                tempo_numerico = int(numeros[0])

        # ===== ANÁLISE DA PERGUNTA PRINCIPAL =====
        # Determinar se o usuário pensou em evasão baseado na resposta
        resposta_evasao_lower = ja_pensou_evasao.lower().strip()

        # Palavras-chave que indicam pensamento de evasão
        palavras_positivas = [
            "sim",
            "já",
            "pensei",
            "pensando",
            "trancar",
            "abandonar",
            "desistir",
            "parar",
            "sair",
            "largar",
            "deixar",
            "quero sair",
            "vou trancar",
            "pretendo",
            "cogitei",
            "considerei",
            "às vezes",
            "algumas vezes",
            "dificuldade",
            "difícil",
            "complicado",
            "não aguento",
            "cansado",
            "estressado",
            "sobrecarregado",
        ]

        palavras_negativas = [
            "não",
            "nunca",
            "jamais",
            "nada",
            "zero",
            "nenhuma",
            "nem pensar",
            "de jeito nenhum",
            "claro que não",
            "definitivamente não",
        ]

        # Análise do texto
        pensou_evasao = False

        if not resposta_evasao_lower or len(resposta_evasao_lower) < 3:
            # Resposta muito curta ou vazia - considera como não pensou
            pensou_evasao = False
        else:
            # Verificar palavras negativas primeiro (mais específicas)
            if any(palavra in resposta_evasao_lower for palavra in palavras_negativas):
                pensou_evasao = False
            # Se não tem palavras negativas, verificar palavras positivas
            elif any(
                palavra in resposta_evasao_lower for palavra in palavras_positivas
            ):
                pensou_evasao = True
            # Se não tem palavras-chave claras, mas tem mais de 20 caracteres, considera como sim
            elif len(resposta_evasao_lower) > 20:
                pensou_evasao = True
            else:
                pensou_evasao = False

        # Conversão para valor numérico (target)
        target_value = 1 if pensou_evasao else 0

        # ===== CRIAÇÃO DOS DADOS PARA O NOVO MODELO =====
        # Apenas as 21 variáveis que o novo modelo aceita (sem data leakage)
        dados_para_modelo = {
            "Como é o seu deslocamento até a universidade?": [
                "Carro",
                "Moto",
                "Ônibus",
                "A pé",
                "Outro",
            ].index(tipo_transporte),
            "Com relação ao transporte do item anterior, ele é:": [
                "Próprio",
                "Cedido",
                "Público(gratuito)",
                "Particular(táxi/moto-táxi)",
                "Não se aplica",
            ].index(propriedade_transporte),
            "O transporte representa uma barreira/dificuldade para frequentar a universidade?": [
                "Sim, sempre",
                "Sim, às vezes",
                "Não, mas já foi",
                "Não, nunca foi",
            ].index(
                barreira_transporte
            ),
            "Você mora em Angicos?": [
                "Sim",
                "Não",
                "Durmo nos dias de aula e atividades",
            ].index(mora_angicos),
            "Você se identifica com o curso que está fazendo?": [
                "Sim",
                "Não, mas quero concluir",
                "Não, não sei se concluirei",
            ].index(identificacao_curso),
            "Como você considera a acessibilidade do Campus?": [
                "Adequada",
                "Inadequada",
            ].index(acessibilidade),
            "Em relação ao tempo necessário como discente para dedicar no estudo?": [
                "É suficiente",
                "É insuficiente, mas desempenho a maioria das atividades",
                "É insuficiente, mas só realizo as atividades obrigatórias",
                "É insuficiente e não consigo realizar as atividades obrigatórias",
            ].index(tempo_estudo),
            "Você trabalha?": [
                "Sim, tenho empresa própria / sou autônomo / profissional liberal",
                "Sim, com emprego formal",
                "Sim, com trabalho temporário",
                "Sim, com emprego informal",
                "Sou bolsista da Universidade",
                "Estou em estágio remunerado",
                "Não trabalho ainda",
            ].index(trabalha),
            "Se você trabalha, em quais horários?": [
                "Tempo integral ou dois turnos",
                "Tempo parcial ou um turno",
                "Horário corrido ou 6 horas diárias",
                "Em regime de escala ou plantão",
                "Sem dias e horários fixos",
                "Não se aplica",
            ].index(horarios_trabalho),
            "É casado(a)/está em união estável?": 1 if estado_civil == "Sim" else 0,
            "Tem filhos?": 1 if tem_filhos == "Sim" else 0,
            "Você contribui para o sustento financeiro da família?": [
                "Sim, sou o único com renda",
                "Sim, sou a principal",
                "Sim, mas não sou o principal",
                "Não contribuo",
            ].index(contribuicao_financeira),
            # Colunas de preconceito como boolean (lógica mutuamente exclusiva)
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Aparência": preconceito_aparencia
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Condição financeira": preconceito_financeiro
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Cor de pele": preconceito_cor
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Curso": preconceito_curso
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Deficiência": preconceito_deficiencia
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Dificuldade de aprendizado": preconceito_aprendizado
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Gênero": preconceito_genero
            and not preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Não": preconceito_nao,
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_idade": preconceito_idade
            and not preconceito_nao,
        }

        # Preprocessar dados
        dados_processados = preprocess_data_for_new_model(
            dados_para_modelo, modelo_data
        )

        if dados_processados is not None:
            try:
                # Fazer predição
                probabilidade = modelo_data["modelo"].predict_proba(dados_processados)[
                    0
                ][1]
                predicao_label = modelo_data["modelo"].predict(dados_processados)[0]

                # ===== SALVAR NO BANCO DE DADOS (com todos os campos, inclusive informativos) =====
                data_to_save = pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.now(),
                            "user_submitting": st.session_state.get(
                                "username", "anonymous"
                            ),
                            "curso": curso,  # Campo informativo
                            "semestre_ingresso": semestre_ingresso,  # Campo informativo
                            "identificacao_curso": identificacao_curso,
                            "tipo_transporte": tipo_transporte,
                            "propriedade_transporte": propriedade_transporte,
                            "barreira_transporte": barreira_transporte,
                            "mora_na_cidade": mora_angicos,
                            "tempo_deslocamento": tempo_numerico,  # Campo informativo
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
                            "estado_civil": True if estado_civil == "Sim" else False,
                            "tem_filhos": True if tem_filhos == "Sim" else False,
                            "qtd_filhos": qtd_filhos,
                            "contribuicao_financeira": contribuicao_financeira,
                            "dificuldades_permanencia": dificuldades,  # Campo informativo
                            "resposta_completa_evasao": ja_pensou_evasao,
                            "pensou_evasao_real": target_value,
                            "prediction_label": int(predicao_label),
                            "prediction_score": float(probabilidade),
                        }
                    ]
                )

                # Salvar dados
                if db.save_submission(engine, data_to_save):
                    st.success("✅ Análise registrada com sucesso no banco de dados!")

                # ===== EXIBIÇÃO DOS RESULTADOS =====
                st.markdown("---")
                st.header("📊 Resultado da Análise Preditiva")

                # Mostrar a resposta real primeiro
                resposta_analise = "SIM" if pensou_evasao else "NÃO"
                st.info(
                    f"**Análise da resposta:** O estudante **{resposta_analise}** pensou em evasão"
                )

                if ja_pensou_evasao.strip():
                    with st.expander("📝 Resposta completa do estudante"):
                        st.write(ja_pensou_evasao)

                col1, col2 = st.columns(2)

                with col1:
                    if predicao_label == 1:
                        st.markdown(
                            '<div class="error-card"><h2>⚠️ ALTO RISCO</h2><p>Estudante tem alta probabilidade de <strong>pensar em trancar disciplinas, período ou abandonar o curso</strong></p></div>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Probabilidade de Pensar em Evasão",
                            f"{probabilidade*100:.1f}%",
                        )
                    else:
                        st.markdown(
                            '<div class="success-card"><h2>✅ BAIXO RISCO</h2><p>Estudante tem baixa probabilidade de pensar em evasão</p></div>',
                            unsafe_allow_html=True,
                        )
                        st.metric(
                            "Probabilidade de Permanência",
                            f"{(1-probabilidade)*100:.1f}%",
                        )

                with col2:
                    if predicao_label == 1:
                        st.subheader("🚨 Ações Recomendadas")
                        st.error(
                            """
                        **Intervenção Imediata:**
                        - Contato proativo com o estudante
                        - Análise das dificuldades específicas mencionadas
                        - Encaminhamento para suporte acadêmico/psicológico
                        - Verificação de auxílios estudantis disponíveis
                        - Acompanhamento sistemático do desempenho
                        """
                        )
                    else:
                        st.subheader("👍 Recomendações")
                        st.success(
                            """
                        **Acompanhamento Preventivo:**
                        - Monitoramento acadêmico regular
                        - Incentivo à participação em atividades extracurriculares
                        - Estímulo ao engajamento em projetos de pesquisa/extensão
                        - Manutenção do canal de comunicação aberto
                        """
                        )

                # Comparação: Predição vs Realidade
                if pensou_evasao and predicao_label == 1:
                    st.success(
                        "✅ **Predição CORRETA**: O modelo identificou corretamente o risco de evasão!"
                    )
                elif not pensou_evasao and predicao_label == 0:
                    st.success(
                        "✅ **Predição CORRETA**: O modelo identificou corretamente a baixa propensão à evasão!"
                    )
                elif pensou_evasao and predicao_label == 0:
                    st.warning(
                        "⚠️ **Falso Negativo**: O estudante PENSOU em evasão, mas o modelo não detectou alto risco."
                    )
                else:
                    st.warning(
                        "⚠️ **Falso Positivo**: O modelo detectou alto risco, mas o estudante NÃO pensou em evasão."
                    )

                # Informações do modelo
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric(
                        "Acurácia (AUC)", f"{modelo_data['metricas']['auc_test']:.3f}"
                    )
                with col2:
                    st.metric("Tipo de Modelo", modelo_data["metricas"]["modelo_tipo"])
                with col3:
                    st.metric("Features Utilizadas", len(modelo_data["feature_names"]))

                # Fatores de risco
                st.markdown("---")
                st.subheader("🔍 Fatores de Risco Identificados")

                fatores_risco = []

                if barreira_transporte in ["Sim, sempre", "Sim, às vezes"]:
                    fatores_risco.append("Dificuldades de transporte")
                if identificacao_curso != "Sim":
                    fatores_risco.append(
                        "Baixa identificação com a área de conhecimento do curso"
                    )
                if tempo_estudo in [
                    "É insuficiente e não consigo realizar as atividades obrigatórias",
                    "É insuficiente, mas só realizo as atividades obrigatórias",
                ]:
                    fatores_risco.append("Tempo insuficiente para estudos")
                if trabalha in [
                    "Sim, com emprego formal",
                    "Sim, tenho empresa própria / sou autônomo / profissional liberal",
                ]:
                    fatores_risco.append("Trabalho que pode interferir nos estudos")
                if horarios_trabalho == "Tempo integral ou dois turnos":
                    fatores_risco.append("Trabalho em período integral")
                if tem_filhos == "Sim":
                    fatores_risco.append("Responsabilidades familiares (filhos)")
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
                    fatores_risco.append("Experiência de preconceito/violência")
                if contribuicao_financeira in [
                    "Sim, sou o único com renda",
                    "Sim, sou a principal",
                ]:
                    fatores_risco.append("Responsabilidade financeira familiar")

                if fatores_risco:
                    for fator in fatores_risco:
                        st.write(f"• {fator}")
                else:
                    st.write("• Nenhum fator de risco específico identificado")

                # Mostrar diferenças do modelo anterior
                with st.expander("🔬 Melhorias do Novo Modelo"):
                    st.markdown(
                        """
                    **Modelo Anterior (com problemas):**
                    - ❌ AUC = 1.0000 (falso, por data leakage)
                    - ❌ Intercept = -14.33 (extremo)
                    - ❌ Usava Curso, Semestre e Dificuldades (vazavam informação)
                    
                    **Modelo Atual (corrigido):**
                    - ✅ AUC = 0.8681 (real e confiável)
                    - ✅ Intercept = 0.37 (normalizado)
                    - ✅ Apenas preditores legítimos (21 variáveis)
                    - ✅ Cientificamente válido
                    
                    **Campos informativos coletados mas não usados na predição:**
                    - 📝 Área de Conhecimento (Curso)
                    - 📝 Semestre de Ingresso
                    - 📝 Dificuldades de Permanência
                    - 📝 Tempo de Deslocamento
                    """
                    )

            except Exception as e:
                st.error(f"❌ Erro na predição: {e}")
                st.write("Dados enviados para o modelo:", dados_para_modelo)
