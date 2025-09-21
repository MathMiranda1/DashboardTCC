import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import database as db


def show_admin_dashboard(engine):
    """Dashboard administrativo com métricas e análises completas para TCC"""
    st.title("🔑 Dashboard Administrativo")

    # Carregar dados do banco
    df = db.load_data(engine)
    stats = db.get_stats(engine)

    if df.empty:
        st.warning("📊 Ainda não há dados de submissão para exibir.")
        st.info(
            "Execute algumas análises na página 'Análise de Risco Individual' para popular o dashboard."
        )
        return

    # ===== SISTEMA DE ALERTAS CRÍTICOS (RF Crítico) =====
    st.markdown("## 🚨 Sistema de Alertas Críticos")

    # Identificar casos críticos (alto risco + alta confiança)
    casos_criticos = df[(df["prediction_label"] == 1) & (df["prediction_score"] >= 0.8)]
    casos_urgentes = df[(df["prediction_label"] == 1) & (df["prediction_score"] >= 0.9)]

    if len(casos_urgentes) > 0:
        st.error(
            f"🚨 **ALERTA MÁXIMO**: {len(casos_urgentes)} casos com risco superior a 90%"
        )
        with st.expander("Ver casos urgentes"):
            urgentes_display = casos_urgentes[
                ["timestamp", "curso", "prediction_score", "resposta_completa_evasao"]
            ].copy()
            urgentes_display["prediction_score"] = urgentes_display[
                "prediction_score"
            ].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(urgentes_display, use_container_width=True)

    if len(casos_criticos) > 0:
        st.warning(
            f"⚠️ **{len(casos_criticos)} casos críticos** detectados (score ≥ 80%)"
        )
        with st.expander("Ver todos os casos críticos"):
            criticos_display = casos_criticos[
                ["timestamp", "curso", "prediction_score", "resposta_completa_evasao"]
            ].copy()
            criticos_display["prediction_score"] = criticos_display[
                "prediction_score"
            ].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(criticos_display, use_container_width=True)

    if len(casos_criticos) == 0:
        st.success("✅ Nenhum caso crítico identificado no momento")

    st.divider()

    # ===== MÉTRICAS PRINCIPAIS + PRECISÃO DO MODELO =====
    st.markdown("## 📈 Métricas Gerais e Performance do Modelo")

    # Primeira linha - métricas básicas
    col1, col2, col3, col4 = st.columns(4)
    with col1:
        st.metric("Total de Análises", stats["total_submissions"])
    with col2:
        st.metric(
            "Alto Risco",
            stats["alto_risco_count"],
            delta=f"{stats['percentual_alto_risco']:.1f}%",
        )
    with col3:
        st.metric("Baixo Risco", stats["baixo_risco_count"])
    with col4:
        if stats["total_submissions"] > 0:
            acuracia = (
                len(df[df["prediction_label"] == df["pensou_evasao_real"]])
                / len(df)
                * 100
            )
            st.metric("Acurácia Geral", f"{acuracia:.1f}%")

    # Segunda linha - métricas de precisão do modelo ML
    st.markdown("### 🎯 Métricas de Precisão do Modelo (Validação Científica)")

    if len(df) > 0:
        # Calcular matriz de confusão
        tp = len(
            df[(df["prediction_label"] == 1) & (df["pensou_evasao_real"] == 1)]
        )  # Verdadeiros Positivos
        fp = len(
            df[(df["prediction_label"] == 1) & (df["pensou_evasao_real"] == 0)]
        )  # Falsos Positivos
        tn = len(
            df[(df["prediction_label"] == 0) & (df["pensou_evasao_real"] == 0)]
        )  # Verdadeiros Negativos
        fn = len(
            df[(df["prediction_label"] == 0) & (df["pensou_evasao_real"] == 1)]
        )  # Falsos Negativos

        # Calcular métricas
        precisao = (tp / (tp + fp)) if (tp + fp) > 0 else 0
        recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
        f1_score = (
            (2 * precisao * recall / (precisao + recall))
            if (precisao + recall) > 0
            else 0
        )
        especificidade = (tn / (tn + fp)) if (tn + fp) > 0 else 0

        col1, col2, col3, col4, col5 = st.columns(5)
        with col1:
            st.metric("Precisão", f"{precisao:.3f}")
            st.caption("TP/(TP+FP)")
        with col2:
            st.metric("Recall (Sensibilidade)", f"{recall:.3f}")
            st.caption("TP/(TP+FN)")
        with col3:
            st.metric("F1-Score", f"{f1_score:.3f}")
            st.caption("Média harmônica")
        with col4:
            st.metric("Especificidade", f"{especificidade:.3f}")
            st.caption("TN/(TN+FP)")
        with col5:
            st.metric("Casos Críticos", len(casos_criticos))
            st.caption("Score ≥ 80%")

        # Matriz de confusão visual
        col1, col2 = st.columns([1, 2])
        with col1:
            st.markdown("**Matriz de Confusão:**")
            matriz_df = pd.DataFrame(
                {
                    "Predito\\Real": ["Baixo Risco", "Alto Risco"],
                    "Baixo Risco": [tn, fn],
                    "Alto Risco": [fp, tp],
                }
            )
            st.dataframe(matriz_df, use_container_width=True)

        with col2:
            st.markdown("**Interpretação Clínica:**")
            if precisao >= 0.8:
                st.success(
                    f"✅ Alta precisão ({precisao:.1%}) - Poucos falsos positivos"
                )
            elif precisao >= 0.6:
                st.warning(
                    f"⚠️ Precisão moderada ({precisao:.1%}) - Alguns falsos positivos"
                )
            else:
                st.error(
                    f"❌ Baixa precisão ({precisao:.1%}) - Muitos falsos positivos"
                )

            if recall >= 0.8:
                st.success(
                    f"✅ Alto recall ({recall:.1%}) - Detecta bem casos de risco"
                )
            elif recall >= 0.6:
                st.warning(f"⚠️ Recall moderado ({recall:.1%}) - Perde alguns casos")
            else:
                st.error(
                    f"❌ Baixo recall ({recall:.1%}) - Muitos casos não detectados"
                )

    st.divider()

    # ===== ANÁLISE DE FATORES DE RISCO (Objetivo Específico do TCC) =====
    st.markdown(
        "## ⚠️ Análise de Fatores de Risco (Identificação dos Principais Influenciadores)"
    )

    # Analisar fatores de risco nos dados
    fatores_risco = {}

    for _, row in df.iterrows():
        # Analisar cada fator potencial
        if row.get("barreira_transporte") in ["Sim, sempre", "Sim, às vezes"]:
            fatores_risco["Dificuldades de Transporte"] = (
                fatores_risco.get("Dificuldades de Transporte", 0) + 1
            )

        if row.get("identificacao_curso") != "Sim":
            fatores_risco["Baixa Identificação com o Curso"] = (
                fatores_risco.get("Baixa Identificação com o Curso", 0) + 1
            )

        if row.get("tempo_estudo", "").startswith("É insuficiente"):
            fatores_risco["Tempo Insuficiente para Estudos"] = (
                fatores_risco.get("Tempo Insuficiente para Estudos", 0) + 1
            )

        if row.get("situacao_trabalho") in [
            "Sim, com emprego formal",
            "Sim, tenho empresa própria / sou autônomo / profissional liberal",
        ]:
            fatores_risco["Trabalho que Interfere nos Estudos"] = (
                fatores_risco.get("Trabalho que Interfere nos Estudos", 0) + 1
            )

        if row.get("horarios_trabalho") == "Tempo integral ou dois turnos":
            fatores_risco["Trabalho em Período Integral"] = (
                fatores_risco.get("Trabalho em Período Integral", 0) + 1
            )

        if row.get("tem_filhos"):
            fatores_risco["Responsabilidades Familiares (Filhos)"] = (
                fatores_risco.get("Responsabilidades Familiares (Filhos)", 0) + 1
            )

        if row.get("contribuicao_financeira") in [
            "Sim, sou o único com renda",
            "Sim, sou a principal",
        ]:
            fatores_risco["Responsabilidade Financeira Principal"] = (
                fatores_risco.get("Responsabilidade Financeira Principal", 0) + 1
            )

        # Verificar preconceitos
        preconceitos = any(
            [
                row.get("preconceito_cor"),
                row.get("preconceito_financeiro"),
                row.get("preconceito_aparencia"),
                row.get("preconceito_deficiencia"),
                row.get("preconceito_aprendizado"),
                row.get("preconceito_genero"),
                row.get("preconceito_curso"),
                row.get("preconceito_idade"),
            ]
        )
        if preconceitos:
            fatores_risco["Experiência de Preconceito/Violência"] = (
                fatores_risco.get("Experiência de Preconceito/Violência", 0) + 1
            )

    # Exibir ranking de fatores
    if fatores_risco:
        st.markdown("### 📊 Ranking dos Principais Fatores de Risco Identificados")

        # Ordenar fatores por frequência
        fatores_ordenados = sorted(
            fatores_risco.items(), key=lambda x: x[1], reverse=True
        )

        col1, col2 = st.columns([2, 1])

        with col1:
            # Criar DataFrame para gráfico
            fatores_df = pd.DataFrame(
                fatores_ordenados, columns=["Fator", "Frequência"]
            )

            chart = (
                alt.Chart(fatores_df)
                .mark_bar(color="#ff6b6b")
                .encode(
                    x=alt.X("Frequência:Q", title="Número de Casos"),
                    y=alt.Y("Fator:N", sort="-x", title="Fatores de Risco"),
                    tooltip=["Fator:N", "Frequência:Q"],
                )
                .properties(title="Frequência dos Fatores de Risco", height=300)
            )

            st.altair_chart(chart, use_container_width=True)

        with col2:
            st.markdown("**Top 5 Fatores:**")
            for i, (fator, freq) in enumerate(fatores_ordenados[:5], 1):
                percentual = (freq / len(df)) * 100
                st.metric(f"{i}º {fator}", f"{freq} casos", delta=f"{percentual:.1f}%")
    else:
        st.info("Nenhum fator de risco específico identificado nos dados atuais.")

    st.divider()

    # ===== FILTROS =====
    st.markdown("## 🔍 Filtros")
    col1, col2, col3 = st.columns(3)

    with col1:
        cursos_unicos = ["Todos"] + list(df["curso"].unique())
        curso_filtro = st.selectbox("Filtrar por Curso:", cursos_unicos)

    with col2:
        risco_filtro = st.selectbox(
            "Filtrar por Risco:", ["Todos", "Alto Risco", "Baixo Risco"]
        )

    with col3:
        periodo_filtro = st.selectbox(
            "Período:", ["Últimos 30 dias", "Últimos 7 dias", "Hoje", "Todos"]
        )

    # Aplicar filtros
    df_filtrado = df.copy()

    if curso_filtro != "Todos":
        df_filtrado = df_filtrado[df_filtrado["curso"] == curso_filtro]

    if risco_filtro == "Alto Risco":
        df_filtrado = df_filtrado[df_filtrado["prediction_label"] == 1]
    elif risco_filtro == "Baixo Risco":
        df_filtrado = df_filtrado[df_filtrado["prediction_label"] == 0]

    # Filtro temporal
    if periodo_filtro != "Todos":
        agora = datetime.now()
        if periodo_filtro == "Hoje":
            df_filtrado = df_filtrado[
                pd.to_datetime(df_filtrado["timestamp"]).dt.date == agora.date()
            ]
        elif periodo_filtro == "Últimos 7 dias":
            df_filtrado = df_filtrado[
                pd.to_datetime(df_filtrado["timestamp"]) >= agora - timedelta(days=7)
            ]
        elif periodo_filtro == "Últimos 30 dias":
            df_filtrado = df_filtrado[
                pd.to_datetime(df_filtrado["timestamp"]) >= agora - timedelta(days=30)
            ]

    # ===== VISUALIZAÇÕES =====
    st.markdown("## 📊 Visualizações")

    if not df_filtrado.empty:
        col1, col2 = st.columns(2)

        with col1:
            # Gráfico de risco por curso
            risco_por_curso = (
                df_filtrado.groupby(["curso", "prediction_label"])
                .size()
                .reset_index(name="count")
            )
            risco_por_curso["risco"] = risco_por_curso["prediction_label"].map(
                {0: "Baixo Risco", 1: "Alto Risco"}
            )

            chart_curso = (
                alt.Chart(risco_por_curso)
                .mark_bar()
                .encode(
                    x="curso:N",
                    y="count:Q",
                    color=alt.Color(
                        "risco:N", scale=alt.Scale(range=["#2ecc71", "#e74c3c"])
                    ),
                    tooltip=["curso:N", "risco:N", "count:Q"],
                )
                .properties(
                    title="Distribuição de Risco por Curso", width=400, height=300
                )
            )

            st.altair_chart(chart_curso, use_container_width=True)

        with col2:
            # Gráfico temporal
            df_filtrado["data"] = pd.to_datetime(df_filtrado["timestamp"]).dt.date
            temporal = (
                df_filtrado.groupby(["data", "prediction_label"])
                .size()
                .reset_index(name="count")
            )
            temporal["risco"] = temporal["prediction_label"].map(
                {0: "Baixo Risco", 1: "Alto Risco"}
            )

            chart_temporal = (
                alt.Chart(temporal)
                .mark_line(point=True)
                .encode(
                    x="data:T",
                    y="count:Q",
                    color=alt.Color(
                        "risco:N", scale=alt.Scale(range=["#2ecc71", "#e74c3c"])
                    ),
                    tooltip=["data:T", "risco:N", "count:Q"],
                )
                .properties(title="Evolução Temporal dos Riscos", width=400, height=300)
            )

            st.altair_chart(chart_temporal, use_container_width=True)

    # ===== RELATÓRIOS INSTITUCIONAIS (RF-008) =====
    st.markdown("## 📋 Dados e Relatórios Institucionais")

    # Seleção de colunas para exibir
    colunas_exibir = st.multiselect(
        "Selecione as colunas para exibir:",
        df.columns.tolist(),
        default=[
            "timestamp",
            "user_submitting",
            "curso",
            "prediction_label",
            "prediction_score",
            "pensou_evasao_real",
        ],
    )

    if colunas_exibir:
        df_exibir = df_filtrado[colunas_exibir].copy()
        if "prediction_label" in df_exibir.columns:
            df_exibir["prediction_label"] = df_exibir["prediction_label"].map(
                {0: "Baixo Risco", 1: "Alto Risco"}
            )

        # Destacar casos de alto risco
        def highlight_alto_risco(row):
            if (
                "prediction_label" in row.index
                and row["prediction_label"] == "Alto Risco"
            ):
                return ["background-color: #ffebee"] * len(row)
            return [""] * len(row)

        st.dataframe(
            df_exibir.style.apply(highlight_alto_risco, axis=1),
            use_container_width=True,
        )

        # ===== SISTEMA DE RELATÓRIOS EXPANDIDO =====
        st.markdown("### 📄 Sistema de Relatórios Institucionais")

        col1, col2, col3, col4 = st.columns(4)

        with col1:
            # Download CSV básico
            csv = df_filtrado.to_csv(index=False)
            st.download_button(
                label="📥 Dados Filtrados (CSV)",
                data=csv,
                file_name=f"dados_evasao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                mime="text/csv",
            )

        with col2:
            # Download apenas casos de alto risco
            alto_risco_df = df_filtrado[df_filtrado["prediction_label"] == 1]
            if len(alto_risco_df) > 0:
                csv_alto_risco = alto_risco_df.to_csv(index=False)
                st.download_button(
                    label="⚠️ Apenas Alto Risco (CSV)",
                    data=csv_alto_risco,
                    file_name=f"alto_risco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
                    mime="text/csv",
                )

        with col3:
            # Relatório Executivo Estruturado
            if st.button("📊 Relatório Executivo"):
                relatorio_executivo = f"""
# RELATÓRIO EXECUTIVO - ANÁLISE DE EVASÃO UNIVERSITÁRIA
**Data de Geração:** {datetime.now().strftime('%d/%m/%Y às %H:%M')}
**Período Analisado:** {periodo_filtro}
**Curso(s):** {curso_filtro}

## RESUMO EXECUTIVO
- Total de análises realizadas: {len(df_filtrado)}
- Estudantes classificados como alto risco: {len(df_filtrado[df_filtrado['prediction_label'] == 1])} ({(len(df_filtrado[df_filtrado['prediction_label'] == 1])/len(df_filtrado)*100):.1f}%)
- Casos críticos identificados (score ≥ 80%): {len(casos_criticos)}
- Acurácia do modelo preditivo: {acuracia:.1f}%

## MÉTRICAS DE PERFORMANCE DO MODELO
- Precisão: {precisao:.3f} (Confiabilidade das predições positivas)
- Recall/Sensibilidade: {recall:.3f} (Capacidade de detectar casos reais)
- F1-Score: {f1_score:.3f} (Equilíbrio entre precisão e recall)
- Especificidade: {especificidade:.3f} (Capacidade de detectar casos negativos)

## PRINCIPAIS FATORES DE RISCO IDENTIFICADOS
{chr(10).join([f"- {fator}: {freq} casos ({freq/len(df)*100:.1f}%)" for fator, freq in sorted(fatores_risco.items(), key=lambda x: x[1], reverse=True)[:5]])}

## DISTRIBUIÇÃO POR CURSO
{df_filtrado['curso'].value_counts().to_string()}

## RECOMENDAÇÕES ESTRATÉGICAS
1. **INTERVENÇÃO IMEDIATA:** Contatar os {len(casos_criticos)} casos críticos identificados
2. **MONITORAMENTO:** Acompanhar sistematicamente estudantes com score > 60%
3. **POLÍTICAS PREVENTIVAS:** Focar nos principais fatores de risco identificados
4. **VALIDAÇÃO CONTÍNUA:** Monitorar a performance do modelo (precisão atual: {precisao:.1%})

## ANEXOS
- Matriz de confusão: TP={tp}, FP={fp}, TN={tn}, FN={fn}
- Casos urgentes (score ≥ 90%): {len(casos_urgentes)}
- Filtros aplicados: Curso={curso_filtro}, Risco={risco_filtro}, Período={periodo_filtro}

---
Relatório gerado automaticamente pelo Sistema de Análise de Evasão Universitária
"""

                st.download_button(
                    label="📄 Baixar Relatório Executivo",
                    data=relatorio_executivo,
                    file_name=f"relatorio_executivo_evasao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

        with col4:
            # Relatório Técnico Detalhado
            if st.button("🔬 Relatório Técnico"):
                relatorio_tecnico = f"""
# RELATÓRIO TÉCNICO - VALIDAÇÃO DO MODELO PREDITIVO DE EVASÃO
**Gerado em:** {datetime.now().strftime('%d/%m/%Y às %H:%M')}

## DADOS DA AMOSTRA
- Tamanho da amostra: {len(df_filtrado)} observações
- Variáveis preditoras: 25+ features (transporte, identificação, trabalho, etc.)
- Variável target: Intenção de evasão (binária)
- Modelo utilizado: Regressão Logística

## MATRIZ DE CONFUSÃO DETALHADA
```
                    VALORES REAIS
                 Baixo Risco  Alto Risco
PREDIÇÕES  Baixo     {tn}        {fn}
           Alto      {fp}        {tp}
```

## MÉTRICAS DE VALIDAÇÃO CRUZADA
- Acurácia: {acuracia:.3f} ({acuracia:.1f}%)
- Precisão: {precisao:.3f} ({precisao:.1f}%)
- Recall (Sensibilidade): {recall:.3f} ({recall:.1f}%)
- Especificidade: {especificidade:.3f} ({especificidade:.1f}%)
- F1-Score: {f1_score:.3f}

## INTERPRETAÇÃO ESTATÍSTICA
- Taxa de Verdadeiros Positivos: {tp}/{tp+fn} = {recall:.1%}
- Taxa de Falsos Positivos: {fp}/{fp+tn} = {fp/(fp+tn):.1%}
- Valor Preditivo Positivo: {tp}/{tp+fp} = {precisao:.1%}
- Valor Preditivo Negativo: {tn}/{tn+fn} = {tn/(tn+fn):.1%}

## DISTRIBUIÇÃO DE SCORES
- Score médio casos positivos: {df[df['prediction_label']==1]['prediction_score'].mean():.3f}
- Score médio casos negativos: {df[df['prediction_label']==0]['prediction_score'].mean():.3f}
- Desvio padrão geral: {df['prediction_score'].std():.3f}

## ANÁLISE DE FATORES DE RISCO (FREQUÊNCIAS)
{chr(10).join([f"- {fator}: {freq} casos" for fator, freq in sorted(fatores_risco.items(), key=lambda x: x[1], reverse=True)])}

## LIMITAÇÕES E CONSIDERAÇÕES
- Modelo baseado em dados auto-declarados
- Amostra de {len(df)} casos para validação
- Recomenda-se retreinamento periódico
- Validação externa necessária para generalização

---
Análise técnica gerada pelo Sistema de ML para Predição de Evasão
"""

                st.download_button(
                    label="🔬 Baixar Relatório Técnico",
                    data=relatorio_tecnico,
                    file_name=f"relatorio_tecnico_evasao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt",
                    mime="text/plain",
                )

        # Métricas finais
        st.markdown("### 📊 Resumo dos Dados Filtrados")
        col1, col2, col3 = st.columns(3)
        with col1:
            st.metric("Registros Exibidos", len(df_filtrado))
        with col2:
            st.metric(
                "Alto Risco Filtrado",
                len(df_filtrado[df_filtrado["prediction_label"] == 1]),
            )
        with col3:
            st.metric(
                "Taxa de Risco",
                f"{len(df_filtrado[df_filtrado['prediction_label'] == 1])/len(df_filtrado)*100:.1f}%",
            )

    else:
        st.warning("Selecione pelo menos uma coluna para exibir os dados.")
