import streamlit as st
import pandas as pd
import numpy as np
import pickle
import re
from datetime import datetime
import database as db


@st.cache_resource
def load_corrected_model():
    """Carrega o modelo CORRIGIDO com codificação consistente"""
    try:
        # Tentar carregar do diretório models
        with open("../models/modelo_evasao_CORRIGIDO.pkl", "rb") as f:
            modelo_completo = pickle.load(f)
    except FileNotFoundError:
        try:
            # Tentar carregar do diretório atual
            with open("modelo_evasao_CORRIGIDO.pkl", "rb") as f:
                modelo_completo = pickle.load(f)
        except FileNotFoundError:
            st.error(
                "❌ Modelo não encontrado! Execute o script de treinamento primeiro."
            )
            return None

    # Extrair componentes
    modelo_data = {
        "modelo": modelo_completo["modelo"],
        "feature_names": modelo_completo["feature_names"],
        "metricas": modelo_completo["metricas"],
        "label_encoders": modelo_completo["preprocessors"].get("label_encoders", {}),
        "imputer": modelo_completo["preprocessors"]["imputer"],
        "scaler": modelo_completo["preprocessors"]["scaler"],
        "metadata": modelo_completo.get("metadata", {}),
    }

    return modelo_data


def codificar_resposta_evasao(resposta):
    """
    Codifica a resposta de evasão de forma CONSISTENTE com o treinamento

    Retorna:
        0 = NÃO pensou em evasão (baixo risco)
        1 = SIM pensou em evasão (alto risco)
    """
    if not resposta or len(resposta.strip()) < 3:
        return 0  # Resposta vazia = não pensou

    resposta_lower = resposta.lower().strip()
    inicio = resposta_lower[:30]  # Primeiras palavras são mais importantes

    # Palavras que indicam SIM pensou em evasão
    palavras_sim = [
        "sim",
        "já",
        "pensei",
        "pensando",
        "vou",
        "quero",
        "pretendo",
        "planejo",
        "talvez",
        "possivelmente",
    ]

    # Palavras que indicam NÃO pensou em evasão
    palavras_nao = [
        "não",
        "nunca",
        "jamais",
        "feliz",
        "gosto",
        "adoro",
        "amo",
        "satisfeito",
        "realizado",
    ]

    # Palavras de evasão forte
    palavras_evasao = [
        "trancar",
        "abandonar",
        "desistir",
        "parar",
        "sair",
        "largar",
        "deixar",
    ]

    # Prioridade 1: Início da resposta
    if any(palavra in inicio for palavra in palavras_sim):
        return 1
    if any(palavra in inicio for palavra in palavras_nao):
        return 0

    # Prioridade 2: Palavras de evasão em qualquer lugar
    if any(palavra in resposta_lower for palavra in palavras_evasao):
        return 1

    # Padrão: se não encontrou indicadores claros, assumir que não pensou
    return 0


def preprocess_input(raw_data, model_data):
    """Preprocessa os dados de entrada usando o pipeline do modelo"""
    try:
        # Criar DataFrame
        df = pd.DataFrame([raw_data])

        # Garantir ordem correta das features
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
        import traceback

        st.code(traceback.format_exc())
        return None


def show_prediction_form(modelo_antigo, engine):
    """Formulário principal com predição usando modelo corrigido"""

    # Carregar modelo corrigido
    modelo_data = load_corrected_model()
    if modelo_data is None:
        return

    # Informações do modelo
    st.sidebar.success(f"✅ Modelo: AUC = {modelo_data['metricas']['test_auc']:.3f}")
    st.sidebar.info("🔬 Modelo cientificamente validado")

    if modelo_data.get("metadata"):
        st.sidebar.caption(f"Labels: 0=Não pensou | 1=Pensou em evasão")

    st.title("🎓 Sistema de Análise de Risco de Evasão Universitária")
    st.markdown(
        "Esta ferramenta utiliza IA para identificar estudantes com maior propensão a **pensar em trancar disciplinas, período ou abandonar o curso**."
    )

    with st.form(key="formulario_evasao"):

        # ===== SEÇÃO 1: INFORMAÇÕES BÁSICAS =====
        st.markdown("### 📋 Informações Básicas")

        col1, col2, col3 = st.columns(3)
        with col1:
            areas_cursos = {
                "Ciências Exatas e da Terra": "Ciência e Tecnologia",
                "Engenharias": "Engenharia Civil",
                "Ciências Humanas": "Pedagogia",
                "Ciências Sociais Aplicadas": "Licenciatura em Computação",
                "Tecnologia da Informação": "Sistema de Informação",
                "Outro": "",
            }
            area = st.selectbox(
                "Área de Conhecimento *(informativo)*", list(areas_cursos.keys())
            )
            curso = areas_cursos[area]

        with col2:
            semestre_texto = st.text_input(
                "Semestre de Ingresso *(informativo)*",
                value="2025.1",
                placeholder="Ex: 2025.1",
            )

        with col3:
            identificacao_curso = st.selectbox(
                "**Você se identifica com o curso?**",
                ("Sim", "Não, mas quero concluir", "Não, não sei se concluirei"),
            )

        # ===== SEÇÃO 2: TRANSPORTE =====
        st.markdown("### 🚗 Transporte e Logística")

        col1, col2, col3 = st.columns(3)
        with col1:
            tipo_transporte = st.selectbox(
                "**Como é seu deslocamento?**",
                ["Carro", "Moto", "Ônibus", "A pé", "Outro"],
            )

            propriedade = st.selectbox(
                "**O transporte é:**",
                [
                    "Próprio",
                    "Cedido",
                    "Público(gratuito)",
                    "Particular(táxi)",
                    "Não se aplica",
                ],
            )

        with col2:
            barreira = st.selectbox(
                "**O transporte é uma barreira?**",
                ["Sim, sempre", "Sim, às vezes", "Não, mas já foi", "Não, nunca foi"],
            )

            mora_cidade = st.selectbox(
                "**Mora na cidade onde estuda?**",
                ("Sim", "Não", "Durmo nos dias de aula"),
            )

        with col3:
            tempo_deslocamento = st.text_input(
                "**Tempo de deslocamento *(informativo)*:**",
                placeholder="Ex: 30min, 1h",
            )

            acessibilidade = st.selectbox(
                "**Acessibilidade do campus:**",
                ("Adequada", "Inadequada"),
            )

        # ===== SEÇÃO 3: EXPERIÊNCIAS NO CAMPUS =====
        st.markdown("### 🏫 Experiências no Campus")
        st.markdown("**Sofreu algum preconceito/violência relativo a:**")

        col1, col2, col3 = st.columns(3)
        with col1:
            prec_cor = st.checkbox("Cor de pele")
            prec_financeiro = st.checkbox("Condição financeira")
            prec_aparencia = st.checkbox("Aparência")
        with col2:
            prec_deficiencia = st.checkbox("Deficiência")
            prec_aprendizado = st.checkbox("Dificuldade de aprendizado")
            prec_genero = st.checkbox("Gênero")
        with col3:
            prec_curso = st.checkbox("Curso")
            prec_idade = st.checkbox("Idade")
            prec_nao = st.checkbox("Não sofri preconceito")

        # ===== SEÇÃO 4: VIDA ACADÊMICA =====
        st.markdown("### 📚 Vida Acadêmica")

        tempo_estudo = st.selectbox(
            "**Tempo para dedicar aos estudos:**",
            [
                "É suficiente",
                "É insuficiente, mas desempenho a maioria das atividades",
                "É insuficiente, mas só realizo as atividades obrigatórias",
                "É insuficiente e não consigo realizar as atividades obrigatórias",
            ],
        )

        # ===== SEÇÃO 5: TRABALHO E FAMÍLIA =====
        st.markdown("### 💼 Trabalho e Vida Pessoal")

        col1, col2, col3 = st.columns(3)
        with col1:
            trabalha = st.selectbox(
                "**Você trabalha?**",
                [
                    "Sim, empresa própria/autônomo",
                    "Sim, emprego formal",
                    "Sim, trabalho temporário",
                    "Sim, emprego informal",
                    "Sou bolsista",
                    "Estágio remunerado",
                    "Não trabalho",
                ],
            )

        with col2:
            horarios = st.selectbox(
                "**Horários de trabalho:**",
                [
                    "Tempo integral ou dois turnos",
                    "Tempo parcial ou um turno",
                    "Horário corrido 6h",
                    "Escala ou plantão",
                    "Sem horários fixos",
                    "Não se aplica",
                ],
            )

        with col3:
            estado_civil = st.selectbox(
                "**Estado civil:**", ["Não casado", "Casado/união estável"]
            )

        col1, col2 = st.columns(2)
        with col1:
            tem_filhos = st.selectbox("**Tem filhos?**", ["Não", "Sim"])
            qtd_filhos = 0
            if tem_filhos == "Sim":
                qtd_filhos = st.number_input(
                    "Quantos?", min_value=1, max_value=10, value=1
                )

        with col2:
            contribuicao = st.selectbox(
                "**Contribuição financeira familiar:**",
                [
                    "Sim, sou o único com renda",
                    "Sim, sou a principal",
                    "Sim, mas não sou o principal",
                    "Não contribuo",
                ],
            )

        # ===== SEÇÃO 6: QUESTÕES ABERTAS =====
        st.markdown("### ✍️ Questões Abertas")

        col1, col2 = st.columns(2)
        with col1:
            dificuldades = st.text_area(
                "**Principais dificuldades *(informativo)*:**",
                placeholder="Descreva suas dificuldades...",
            )

        with col2:
            resposta_evasao = st.text_area(
                "**🎯 Já pensou em trancar ou abandonar o curso? Por quê?**",
                placeholder="Seja sincero(a)...",
                help="Esta é a pergunta principal.",
            )

        submit = st.form_submit_button("📊 Analisar Risco de Evasão")

    # ===== PROCESSAMENTO DA SUBMISSÃO =====
    if submit:

        # 1. CODIFICAR A RESPOSTA DE EVASÃO (ground truth)
        target_real = codificar_resposta_evasao(resposta_evasao)
        pensou_text = "SIM" if target_real == 1 else "NÃO"

        # 2. PREPARAR FEATURES NUMÉRICAS
        # Mapeamentos exatos do treinamento
        dados_numericos = {
            "Como é o seu deslocamento até a universidade?": {
                "A pé": 0,
                "Carro": 1,
                "Moto": 2,
                "Outro": 3,
                "Ônibus": 4,
            }[tipo_transporte],
            "Com relação ao transporte do item anterior, ele é:": {
                "Cedido": 0,
                "Não se aplica": 1,
                "Particular(táxi)": 2,
                "Próprio": 3,
                "Público(gratuito)": 4,
            }[propriedade],
            "O transporte representa uma barreira/dificuldade para frequentar a universidade?": {
                "Não, mas já foi": 0,
                "Não, nunca foi": 1,
                "Sim, sempre": 2,
                "Sim, às vezes": 3,
            }[
                barreira
            ],
            "Você mora em Angicos?": {
                "Durmo nos dias de aula": 0,
                "Não": 1,
                "Sim": 2,
            }[mora_cidade],
            "Você se identifica com o curso que está fazendo?": {
                "Não, mas quero concluir": 0,
                "Não, não sei se concluirei": 1,
                "Sim": 2,
            }[identificacao_curso],
            "Como você considera a acessibilidade do Campus?": {
                "Adequada": 0,
                "Inadequada": 1,
            }[acessibilidade],
            "Em relação ao tempo necessário como discente para dedicar no estudo?": {
                "É insuficiente e não consigo realizar as atividades obrigatórias": 0,
                "É insuficiente, mas desempenho a maioria das atividades": 1,
                "É insuficiente, mas só realizo as atividades obrigatórias": 2,
                "É suficiente": 3,
            }[tempo_estudo],
            "Você trabalha?": {
                "Estágio remunerado": 0,
                "Não trabalho": 1,
                "Sim, emprego formal": 2,
                "Sim, emprego informal": 3,
                "Sim, trabalho temporário": 4,
                "Sim, empresa própria/autônomo": 5,
                "Sou bolsista": 6,
            }[trabalha],
            "Se você trabalha, em quais horários?": {
                "Escala ou plantão": 0,
                "Horário corrido 6h": 1,
                "Não se aplica": 2,
                "Sem horários fixos": 3,
                "Tempo integral ou dois turnos": 4,
                "Tempo parcial ou um turno": 5,
            }[horarios],
            "É casado(a)/está em união estável?": {
                "Não casado": 0,
                "Casado/união estável": 1,
            }[estado_civil],
            "Tem filhos?": {"Não": 0, "Sim": 1}[tem_filhos],
            "Você contribui para o sustento financeiro da família?": {
                "Não contribuo": 0,
                "Sim, mas não sou o principal": 1,
                "Sim, sou a principal": 2,
                "Sim, sou o único com renda": 3,
            }[contribuicao],
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Aparência": int(
                prec_aparencia and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Condição financeira": int(
                prec_financeiro and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Cor de pele": int(
                prec_cor and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Curso": int(
                prec_curso and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Deficiência": int(
                prec_deficiencia and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Dificuldade de aprendizado": int(
                prec_aprendizado and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Gênero": int(
                prec_genero and not prec_nao
            ),
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_idade": int(
                prec_idade and not prec_nao
            ),
        }

        # 3. PREPROCESSAR
        dados_processados = preprocess_input(dados_numericos, modelo_data)

        if dados_processados is not None:
            try:
                # 4. PREDIÇÃO
                # No CSV original: 0=Pensou em evasão, 1=Não pensou
                # Como não invertemos mais, a classe 0 do modelo = Pensou em evasão
                probabilidades = modelo_data["modelo"].predict_proba(dados_processados)[
                    0
                ]

                # Com o modelo corrigido:
                # Classe 0 = NÃO pensou em evasão (baixo risco)
                # Classe 1 = SIM pensou em evasão (alto risco)
                probabilidade_evasao = probabilidades[
                    1
                ]  # Usar a probabilidade da CLASSE 1
                predicao = 1 if probabilidade_evasao > 0.5 else 0
                # 5. SALVAR NO BANCO
                data_to_save = pd.DataFrame(
                    [
                        {
                            "timestamp": datetime.now(),
                            "user_submitting": st.session_state.get("username", "anon"),
                            "curso": curso,
                            "tipo_transporte": tipo_transporte,
                            "propriedade_transporte": propriedade,
                            "barreira_transporte": barreira,
                            "mora_na_cidade": mora_cidade,
                            "identificacao_curso": identificacao_curso,
                            "acessibilidade_campus": acessibilidade,
                            "preconceito_cor": prec_cor,
                            "preconceito_financeiro": prec_financeiro,
                            "preconceito_aparencia": prec_aparencia,
                            "preconceito_deficiencia": prec_deficiencia,
                            "preconceito_aprendizado": prec_aprendizado,
                            "preconceito_genero": prec_genero,
                            "preconceito_curso": prec_curso,
                            "preconceito_idade": prec_idade,
                            "preconceito_nao": prec_nao,
                            "tempo_estudo": tempo_estudo,
                            "situacao_trabalho": trabalha,
                            "horarios_trabalho": horarios,
                            "estado_civil": estado_civil == "Casado/união estável",
                            "tem_filhos": tem_filhos == "Sim",
                            "qtd_filhos": qtd_filhos,
                            "contribuicao_financeira": contribuicao,
                            "dificuldades_permanencia": dificuldades,
                            "resposta_completa_evasao": resposta_evasao,
                            "pensou_evasao_real": int(target_real),
                            "prediction_label": int(predicao),
                            "prediction_score": float(probabilidade_evasao),
                        }
                    ]
                )

                if db.save_submission(engine, data_to_save):
                    st.success("✅ Análise registrada no banco de dados!")

                # 6. EXIBIR RESULTADOS
                st.markdown("---")
                st.header("📊 Resultado da Análise")

                st.info(
                    f"**Análise da resposta:** O estudante **{pensou_text}** pensou em evasão"
                )

                if resposta_evasao.strip():
                    with st.expander("📝 Resposta completa"):
                        st.write(resposta_evasao)

                col1, col2 = st.columns(2)

                with col1:
                    if probabilidade_evasao > 0.5:
                        st.error("### ⚠️ ALTO RISCO")
                        st.metric(
                            "Probabilidade de Pensar em Evasão",
                            f"{probabilidade_evasao*100:.1f}%",
                        )
                    else:
                        st.success("### ✅ BAIXO RISCO")
                        st.metric(
                            "Probabilidade de Risco de Evasão",  # Ou mantenha "de Permanência", mas a lógica muda
                            f"{probabilidade_evasao*100:.1f}%",
                        )

                        # E, para maior clareza, você pode exibir a probabilidade de permanência explicitamente:
                        st.write(
                            f"Probabilidade de Permanência: {(1-probabilidade_evasao)*100:.1f}%"
                        )

                with col2:
                    st.subheader("📌 Ações Recomendadas")
                    if probabilidade_evasao > 0.5:
                        st.error(
                            """
                        **Intervenção Imediata:**
                        - Contato proativo com o estudante
                        - Análise das dificuldades específicas
                        - Encaminhamento para suporte
                        - Verificação de auxílios disponíveis
                        """
                        )
                    else:
                        st.success(
                            """
                        **Acompanhamento Preventivo:**
                        - Monitoramento regular
                        - Incentivo à participação em atividades
                        - Canal de comunicação aberto
                        """
                        )

                # 7. VALIDAÇÃO DA PREDIÇÃO
                if target_real == predicao:
                    st.success(
                        "✅ **Predição CORRETA**: O modelo acertou a classificação!"
                    )
                else:
                    if target_real == 1 and predicao == 0:
                        st.warning(
                            "⚠️ **Falso Negativo**: Estudante pensou em evasão, mas modelo não detectou."
                        )
                    else:
                        st.warning(
                            "⚠️ **Falso Positivo**: Modelo detectou risco, mas estudante não pensou em evasão."
                        )

                # 8. MÉTRICAS DO MODELO
                col1, col2, col3 = st.columns(3)
                with col1:
                    st.metric("AUC", f"{modelo_data['metricas']['test_auc']:.3f}")
                with col2:
                    st.metric("Tipo", modelo_data["metricas"]["nome"])
                with col3:
                    st.metric("Features", len(modelo_data["feature_names"]))

                # 9. FATORES DE RISCO
                st.markdown("---")
                st.subheader("🔍 Fatores de Risco Identificados")

                fatores = []
                if barreira in ["Sim, sempre", "Sim, às vezes"]:
                    fatores.append("Dificuldades de transporte")
                if identificacao_curso != "Sim":
                    fatores.append("Baixa identificação com o curso")
                if "insuficiente" in tempo_estudo.lower():
                    fatores.append("Tempo insuficiente para estudos")
                if trabalha in ["Sim, emprego formal", "Sim, empresa própria/autônomo"]:
                    fatores.append("Trabalho pode interferir nos estudos")
                if horarios == "Tempo integral ou dois turnos":
                    fatores.append("Trabalho em período integral")
                if tem_filhos == "Sim":
                    fatores.append("Responsabilidades familiares")
                if any([prec_cor, prec_financeiro, prec_aparencia, prec_deficiencia]):
                    fatores.append("Experiência de preconceito/violência")
                if contribuicao in [
                    "Sim, sou o único com renda",
                    "Sim, sou a principal",
                ]:
                    fatores.append("Responsabilidade financeira familiar")

                if fatores:
                    for fator in fatores:
                        st.write(f"• {fator}")
                else:
                    st.write("• Nenhum fator de risco específico identificado")

            except Exception as e:
                st.error(f"❌ Erro na predição: {e}")
                import traceback

                st.code(traceback.format_exc())
