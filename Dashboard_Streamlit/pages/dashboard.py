import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime
import database as db


def show_admin_dashboard(engine):
    """Dashboard administrativo com métricas e visualizações"""
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

    # ===== MÉTRICAS PRINCIPAIS =====
    st.markdown("### 📈 Métricas Gerais")
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
            st.metric("Acurácia do Modelo", f"{acuracia:.1f}%")

    # ===== FILTROS =====
    st.markdown("### 🔍 Filtros")
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

    # ===== GRÁFICOS =====
    st.markdown("### 📊 Visualizações")

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
                    color="risco:N",
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
                    color="risco:N",
                    tooltip=["data:T", "risco:N", "count:Q"],
                )
                .properties(title="Evolução Temporal dos Riscos", width=400, height=300)
            )

            st.altair_chart(chart_temporal, use_container_width=True)

    # ===== TABELA DE DADOS =====
    st.markdown("### 📋 Dados Detalhados")

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
        st.dataframe(df_exibir, use_container_width=True)

        # Botão para download
        csv = df_filtrado.to_csv(index=False)
        st.download_button(
            label="📥 Baixar dados filtrados (CSV)",
            data=csv,
            file_name=f"dados_evasao_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )
