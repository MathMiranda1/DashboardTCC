import streamlit as st
import pandas as pd
import altair as alt
from datetime import datetime, timedelta
import database as db
import plotly.express as px
import plotly.graph_objects as go


def show_admin_dashboard(engine):
    """Dashboard administrativo para o modelo Random Forest"""
    st.title("🔑 Dashboard Administrativo - Modelo Random Forest")

    # Carregar dados
    df = db.load_data(engine)
    stats = db.get_stats(engine)

    if df.empty:
        st.warning("📊 Ainda não há dados de submissão para exibir.")
        st.info("Execute algumas análises para popular o dashboard.")
        return

    # ===== CLASSIFICAR POR CATEGORIAS =====
    def classificar_risco(score):
        """Classifica baseado no score do Random Forest"""
        if score < 0.30:
            return "BAIXO"
        elif score < 0.50:
            return "MODERADO"
        elif score < 0.70:
            return "ALTO"
        else:
            return "CRÍTICO"

    df["categoria_risco"] = df["prediction_score"].apply(classificar_risco)

    # ===== ALERTAS CRÍTICOS =====
    st.markdown("## 🚨 Sistema de Alertas")

    casos_criticos = df[df["prediction_score"] >= 0.70]  # CRÍTICO: ≥70%
    casos_altos = df[
        (df["prediction_score"] >= 0.50) & (df["prediction_score"] < 0.70)
    ]  # ALTO: 50-70%
    casos_moderados = df[
        (df["prediction_score"] >= 0.30) & (df["prediction_score"] < 0.50)
    ]  # MODERADO: 30-50%
    casos_baixos = df[df["prediction_score"] < 0.30]  # BAIXO: <30%

    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("🔴 CRÍTICO", len(casos_criticos), help="Score ≥ 70%")
    with col2:
        st.metric("🟠 ALTO", len(casos_altos), help="50-70%")
    with col3:
        st.metric("🟡 MODERADO", len(casos_moderados), help="30-50%")
    with col4:
        st.metric("🟢 BAIXO", len(casos_baixos), help="< 30%")

    if len(casos_criticos) > 0:
        st.error(f"🚨 **ALERTA MÁXIMO**: {len(casos_criticos)} casos CRÍTICOS (≥70%)")
        with st.expander("Ver casos urgentes"):
            urgentes_display = casos_criticos[
                ["timestamp", "curso", "prediction_score", "resposta_completa_evasao"]
            ].copy()
            urgentes_display["prediction_score"] = urgentes_display[
                "prediction_score"
            ].apply(lambda x: f"{x*100:.1f}%")
            st.dataframe(urgentes_display, use_container_width=True)

    st.divider()

    # ===== MÉTRICAS PRINCIPAIS =====
    st.markdown("## 📊 Métricas do Modelo")

    col1, col2, col3 = st.columns(3)

    with col1:
        st.metric("Total de Análises", len(df))
        st.caption("Todas as predições realizadas")

    with col2:
        score_medio = df["prediction_score"].mean()
        st.metric("Score Médio", f"{score_medio*100:.1f}%")
        st.caption("Probabilidade média de evasão")

    with col3:
        acuracia = (
            len(df[df["prediction_label"] == df["pensou_evasao_real"]]) / len(df) * 100
        )
        st.metric("Acurácia", f"{acuracia:.1f}%")
        st.caption("Concordância com respostas reais")

    # ===== DISTRIBUIÇÃO POR CATEGORIA =====
    st.markdown("### 📈 Distribuição por Categoria de Risco")

    dist_categorias = df["categoria_risco"].value_counts()

    fig_pizza = go.Figure(
        data=[
            go.Pie(
                labels=dist_categorias.index,
                values=dist_categorias.values,
                marker=dict(colors=["#22c55e", "#eab308", "#f97316", "#ef4444"]),
                hole=0.3,
            )
        ]
    )
    fig_pizza.update_layout(title="Distribuição de Risco na População", height=400)
    st.plotly_chart(fig_pizza, use_container_width=True)

    # Tabela de distribuição
    col1, col2 = st.columns(2)

    with col1:
        st.markdown("**Contagem por Categoria:**")
        for cat in ["BAIXO", "MODERADO", "ALTO", "CRÍTICO"]:
            count = len(df[df["categoria_risco"] == cat])
            pct = (count / len(df)) * 100 if len(df) > 0 else 0
            st.write(f"- {cat}: {count} casos ({pct:.1f}%)")

    with col2:
        st.markdown("**Estatísticas de Score:**")
        st.write(f"- Mínimo: {df['prediction_score'].min()*100:.1f}%")
        st.write(f"- Máximo: {df['prediction_score'].max()*100:.1f}%")
        st.write(f"- Mediana: {df['prediction_score'].median()*100:.1f}%")
        st.write(f"- Desvio Padrão: {df['prediction_score'].std()*100:.1f}%")

    st.divider()

    # ===== MATRIZ DE CONFUSÃO =====
    st.markdown("### 🎯 Matriz de Confusão e Métricas")

    # Calcular matriz de confusão
    tp = len(df[(df["prediction_label"] == 1) & (df["pensou_evasao_real"] == 1)])
    fp = len(df[(df["prediction_label"] == 1) & (df["pensou_evasao_real"] == 0)])
    tn = len(df[(df["prediction_label"] == 0) & (df["pensou_evasao_real"] == 0)])
    fn = len(df[(df["prediction_label"] == 0) & (df["pensou_evasao_real"] == 1)])

    # Calcular métricas (evitando divisão por zero)
    precisao = (tp / (tp + fp)) if (tp + fp) > 0 else 0
    recall = (tp / (tp + fn)) if (tp + fn) > 0 else 0
    f1_score = (
        (2 * precisao * recall / (precisao + recall)) if (precisao + recall) > 0 else 0
    )
    especificidade = (tn / (tn + fp)) if (tn + fp) > 0 else 0

    # Verificar se há dados suficientes
    if tp + fp == 0:
        st.warning(
            "⚠️ Nenhum caso classificado como alto risco. Matriz de confusão não pode ser calculada."
        )
        st.info(
            "💡 Faça mais testes incluindo cenários de alto risco (score > 50%) para popular as métricas."
        )

    # Exibir métricas
    col1, col2, col3, col4 = st.columns(4)

    with col1:
        st.metric("Precisão", f"{precisao:.3f}" if precisao > 0 else "N/A")
        st.caption("TP/(TP+FP)")
    with col2:
        st.metric("Recall (Sensibilidade)", f"{recall:.3f}" if recall > 0 else "N/A")
        st.caption("TP/(TP+FN)")
    with col3:
        st.metric("F1-Score", f"{f1_score:.3f}" if f1_score > 0 else "N/A")
        st.caption("Média harmônica")
    with col4:
        st.metric(
            "Especificidade", f"{especificidade:.3f}" if especificidade > 0 else "N/A"
        )
        st.caption("TN/(TN+FP)")

    # Matriz visual
    if tp + fp > 0:
        col1, col2 = st.columns([1, 2])

        with col1:
            matriz_df = pd.DataFrame(
                {
                    "Predito\\Real": ["Baixo Risco", "Alto Risco"],
                    "Baixo Risco": [tn, fn],
                    "Alto Risco": [fp, tp],
                }
            )
            st.dataframe(matriz_df, use_container_width=True)

        with col2:
            # Avaliar desempenho
            if precisao >= 0.8:
                st.success(f"✅ Alta precisão ({precisao:.1%})")
            elif precisao >= 0.6:
                st.warning(f"⚠️ Precisão moderada ({precisao:.1%})")
            else:
                st.error(f"❌ Baixa precisão ({precisao:.1%})")

            if recall >= 0.8:
                st.success(f"✅ Alto recall ({recall:.1%})")
            elif recall >= 0.6:
                st.warning(f"⚠️ Recall moderado ({recall:.1%})")
            else:
                st.error(f"❌ Baixo recall ({recall:.1%})")

    st.divider()

    # ===== ANÁLISE DE FATORES DE RISCO =====
    st.markdown("## ⚠️ Análise de Fatores de Risco")

    fatores_risco = {}

    for _, row in df.iterrows():
        if row.get("barreira_transporte") in ["Sim, sempre", "Sim, às vezes"]:
            fatores_risco["Dificuldades de Transporte"] = (
                fatores_risco.get("Dificuldades de Transporte", 0) + 1
            )

        if row.get("identificacao_curso") != "Sim":
            fatores_risco["Baixa Identificação com Curso"] = (
                fatores_risco.get("Baixa Identificação com Curso", 0) + 1
            )

        if "insuficiente" in str(row.get("tempo_estudo", "")).lower():
            fatores_risco["Tempo Insuficiente"] = (
                fatores_risco.get("Tempo Insuficiente", 0) + 1
            )

        if row.get("horarios_trabalho") == "Tempo integral ou dois turnos":
            fatores_risco["Trabalho Integral"] = (
                fatores_risco.get("Trabalho Integral", 0) + 1
            )

        if row.get("tem_filhos"):
            fatores_risco["Responsabilidades Familiares"] = (
                fatores_risco.get("Responsabilidades Familiares", 0) + 1
            )

        if row.get("mora_na_cidade") == "Não":
            fatores_risco["Distância do Campus"] = (
                fatores_risco.get("Distância do Campus", 0) + 1
            )

        # Verificar se sofreu algum preconceito
        colunas_preconceito = [
            col
            for col in df.columns
            if "preconceito" in col.lower() and col != "preconceito_nao"
        ]
        if any(row.get(col, False) for col in colunas_preconceito):
            fatores_risco["Experiência de Preconceito/Violência"] = (
                fatores_risco.get("Experiência de Preconceito/Violência", 0) + 1
            )

    if fatores_risco:
        fatores_ordenados = sorted(
            fatores_risco.items(), key=lambda x: x[1], reverse=True
        )
        fatores_df = pd.DataFrame(fatores_ordenados, columns=["Fator", "Frequência"])

        fig_fatores = px.bar(
            fatores_df,
            x="Frequência",
            y="Fator",
            orientation="h",
            title="Frequência dos Fatores de Risco",
            color="Frequência",
            color_continuous_scale="Reds",
        )
        fig_fatores.update_layout(height=400)
        st.plotly_chart(fig_fatores, use_container_width=True)
    else:
        st.info("Nenhum fator de risco específico identificado nos dados.")

    st.divider()

    # ===== EVOLUÇÃO TEMPORAL =====
    st.markdown("## 📈 Evolução Temporal")

    df["data"] = pd.to_datetime(df["timestamp"]).dt.date

    temporal = df.groupby(["data", "categoria_risco"]).size().reset_index(name="count")

    fig_temporal = px.line(
        temporal,
        x="data",
        y="count",
        color="categoria_risco",
        title="Evolução das Categorias de Risco ao Longo do Tempo",
        color_discrete_map={
            "BAIXO": "#22c55e",
            "MODERADO": "#eab308",
            "ALTO": "#f97316",
            "CRÍTICO": "#ef4444",
        },
    )
    fig_temporal.update_layout(height=400)
    st.plotly_chart(fig_temporal, use_container_width=True)

    st.divider()

    # ===== ANÁLISE DETALHADA DE SCORES =====
    st.markdown("## 📊 Análise Detalhada de Scores")

    col1, col2 = st.columns(2)

    with col1:
        # Histograma de distribuição
        fig_hist = px.histogram(
            df,
            x="prediction_score",
            nbins=20,
            title="Distribuição de Scores de Risco",
            labels={"prediction_score": "Score", "count": "Frequência"},
            color_discrete_sequence=["#667eea"],
        )
        fig_hist.add_vline(
            x=0.30, line_dash="dash", line_color="green", annotation_text="Baixo"
        )
        fig_hist.add_vline(
            x=0.50, line_dash="dash", line_color="orange", annotation_text="Threshold"
        )
        fig_hist.add_vline(
            x=0.70, line_dash="dash", line_color="red", annotation_text="Crítico"
        )
        fig_hist.update_layout(height=350)
        st.plotly_chart(fig_hist, use_container_width=True)

    with col2:
        # Box plot por categoria
        fig_box = px.box(
            df,
            x="categoria_risco",
            y="prediction_score",
            title="Distribuição de Scores por Categoria",
            color="categoria_risco",
            color_discrete_map={
                "BAIXO": "#22c55e",
                "MODERADO": "#eab308",
                "ALTO": "#f97316",
                "CRÍTICO": "#ef4444",
            },
        )
        fig_box.update_layout(height=350, showlegend=False)
        st.plotly_chart(fig_box, use_container_width=True)

    # Comparação por curso (se houver mais de um)
    if df["curso"].nunique() > 1:
        st.markdown("### 📚 Distribuição de Risco por Curso")

        curso_risco = (
            df.groupby(["curso", "categoria_risco"]).size().reset_index(name="count")
        )

        fig_curso = px.bar(
            curso_risco,
            x="curso",
            y="count",
            color="categoria_risco",
            title="Distribuição de Categorias de Risco por Curso",
            color_discrete_map={
                "BAIXO": "#22c55e",
                "MODERADO": "#eab308",
                "ALTO": "#f97316",
                "CRÍTICO": "#ef4444",
            },
            barmode="stack",
        )
        fig_curso.update_layout(height=400)
        st.plotly_chart(fig_curso, use_container_width=True)

    st.divider()

    # ===== ANÁLISE DE CONCORDÂNCIA =====
    st.markdown("## 🎯 Análise de Concordância (Modelo vs Resposta Real)")

    # Criar tabela de concordância
    concordancia = pd.crosstab(
        df["pensou_evasao_real"],
        df["prediction_label"],
        rownames=["Real"],
        colnames=["Predito"],
        margins=True,
    )

    col1, col2 = st.columns([1, 1])

    with col1:
        st.markdown("**Tabela de Concordância:**")
        st.dataframe(concordancia, use_container_width=True)

    with col2:
        # Gráfico de concordância
        df_temp = df.copy()
        df_temp["real_label"] = df_temp["pensou_evasao_real"].map(
            {0: "Não pensou", 1: "Pensou"}
        )
        df_temp["pred_label"] = df_temp["prediction_label"].map(
            {0: "Não pensou", 1: "Pensou"}
        )

        concordancia_count = (
            df_temp.groupby(["real_label", "pred_label"])
            .size()
            .reset_index(name="count")
        )

        fig_concordancia = px.bar(
            concordancia_count,
            x="real_label",
            y="count",
            color="pred_label",
            title="Concordância: Real vs Predito",
            barmode="group",
            color_discrete_map={"Não pensou": "#22c55e", "Pensou": "#ef4444"},
        )
        st.plotly_chart(fig_concordancia, use_container_width=True)

    st.divider()

    # ===== TABELA RESUMO COMPLETA =====
    st.markdown("## 📋 Tabela Resumo de Casos")

    # Criar tabela resumo
    colunas_resumo = [
        "id",
        "timestamp",
        "curso",
        "categoria_risco",
        "prediction_score",
        "identificacao_curso",
        "mora_na_cidade",
        "tem_filhos",
        "horarios_trabalho",
    ]

    # Verificar quais colunas existem no DataFrame
    colunas_existentes = [col for col in colunas_resumo if col in df.columns]

    df_resumo = df[colunas_existentes].copy()

    # Só formatar se não estiver vazio
    if not df_resumo.empty:
        df_resumo["timestamp"] = pd.to_datetime(df_resumo["timestamp"]).dt.strftime(
            "%d/%m/%Y %H:%M"
        )
        df_resumo["prediction_score"] = df_resumo["prediction_score"].apply(
            lambda x: f"{x*100:.1f}%"
        )

        # Renomear colunas
        rename_dict = {
            "id": "ID",
            "timestamp": "Data/Hora",
            "curso": "Curso",
            "categoria_risco": "Categoria",
            "prediction_score": "Score",
            "identificacao_curso": "Identifica com Curso",
            "mora_na_cidade": "Mora na Cidade",
            "tem_filhos": "Tem Filhos",
            "horarios_trabalho": "Horário Trabalho",
        }

        df_resumo = df_resumo.rename(
            columns={k: v for k, v in rename_dict.items() if k in df_resumo.columns}
        )

        # ⭐ NOVA FUNCIONALIDADE: SELEÇÃO INTERATIVA
        st.info("💡 **Clique em uma linha da tabela abaixo para ver detalhes do caso**")

        evento = st.dataframe(
            df_resumo,
            use_container_width=True,
            height=400,
            on_select="rerun",
            selection_mode="single-row",
        )

        # ⭐ MOSTRAR DETALHES QUANDO LINHA FOR SELECIONADA
        if evento.selection.rows:
            linha_selecionada = evento.selection.rows[0]
            caso_id = df_resumo.iloc[linha_selecionada]["ID"]

            # Buscar dados completos do caso
            caso_completo = df[df["id"] == caso_id].iloc[0]

            st.markdown("---")
            st.markdown(f"## 🔍 **Detalhes do Caso #{caso_id}**")

            col1, col2 = st.columns(2)

            with col1:
                st.markdown("### 📌 Ações Recomendadas")
                acoes = caso_completo.get("acoes_recomendadas", "Não disponível")

                # Dividir ações por ponto e vírgula se estiver concatenado
                if ";" in str(acoes):
                    for acao in acoes.split(";"):
                        st.write(f"• {acao.strip()}")
                else:
                    st.write(f"• {acoes}")

            with col2:
                st.markdown("### 🔍 Fatores de Risco")
                fatores = caso_completo.get(
                    "fatores_risco", "Nenhum fator identificado"
                )

                # Dividir fatores por ponto e vírgula
                if ";" in str(fatores):
                    for fator in fatores.split(";"):
                        st.write(f"• {fator.strip()}")
                else:
                    st.write(f"• {fatores}")

            # Informações adicionais
            st.markdown("### 📄 Informações do Caso")
            col1, col2, col3 = st.columns(3)

            with col1:
                st.metric(
                    "Score de Risco", f"{caso_completo['prediction_score']*100:.1f}%"
                )
            with col2:
                st.metric("Categoria", caso_completo.get("categoria_risco", "N/A"))
            with col3:
                real_label = (
                    "Pensou em evasão"
                    if caso_completo.get("pensou_evasao_real", 0) == 1
                    else "Não pensou"
                )
                st.metric("Resposta Real", real_label)

            # Resposta completa se disponível
            if caso_completo.get("resposta_completa_evasao"):
                with st.expander("📝 Ver resposta completa sobre evasão"):
                    st.write(caso_completo["resposta_completa_evasao"])

    else:
        st.warning("📊 Ainda não há dados para exibir na tabela.")

    st.divider()

    # ===== RELATÓRIOS =====
    st.markdown("## 📄 Relatórios e Exportação")

    col1, col2, col3 = st.columns(3)

    with col1:
        csv = df.to_csv(index=False)
        st.download_button(
            label="📥 Baixar Todos os Dados (CSV)",
            data=csv,
            file_name=f"dados_completos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with col2:
        criticos_csv = df[df["prediction_score"] >= 0.70].to_csv(index=False)
        st.download_button(
            label="🔴 Apenas Casos Críticos (CSV)",
            data=criticos_csv,
            file_name=f"casos_criticos_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )

    with col3:
        alto_risco_csv = df[df["prediction_score"] >= 0.50].to_csv(index=False)
        st.download_button(
            label="🟠 Alto + Crítico (CSV)",
            data=alto_risco_csv,
            file_name=f"alto_risco_{datetime.now().strftime('%Y%m%d_%H%M%S')}.csv",
            mime="text/csv",
        )


if __name__ == "__main__":
    show_admin_dashboard()
