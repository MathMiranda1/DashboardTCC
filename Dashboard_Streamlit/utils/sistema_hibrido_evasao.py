"""
SISTEMA HÍBRIDO DE PREDIÇÃO DE EVASÃO UNIVERSITÁRIA
====================================================

Combina Machine Learning (30%) + Regras Especializadas (70%)
Baseado em literatura sobre retenção e evasão estudantil

Autor: Sistema Híbrido para TCC
Data: Outubro 2025
"""

import pickle
import numpy as np
import pandas as pd
from typing import Dict, List, Tuple


class SistemaHibridoEvasao:
    """
    Sistema híbrido que combina ML com regras de negócio para predição de evasão.

    Fatores de risco baseados em literatura:
    - Vincent Tinto (1975): Teoria da Integração Estudantil
    - Bean & Metzner (1985): Modelo de Evasão para Estudantes Não-Tradicionais
    - Cabrera et al. (1993): Fatores financeiros e familiares
    """

    def __init__(self, caminho_modelo="modelo_evasao_NOVO.pkl"):
        """Inicializa o sistema híbrido"""
        self.modelo_ml = None
        self.preprocessors = None
        self.feature_names = None
        self.pesos = {
            "ml": 0.30,  # 30% modelo ML
            "regras": 0.70,  # 70% regras especializadas
        }

        # Carregar modelo ML
        self._carregar_modelo(caminho_modelo)

        # Definir pesos das regras (baseado em meta-análises de evasão)
        self.pesos_regras = {
            "transporte_dificil": 0.15,
            "trabalho_conflitante": 0.20,
            "responsabilidades_familiares": 0.18,
            "falta_tempo_estudo": 0.17,
            "distancia_campus": 0.12,
            "acessibilidade_ruim": 0.10,
            "sobrecarga_total": 0.08,
        }

    def _carregar_modelo(self, caminho):
        """Carrega o modelo ML treinado"""
        try:
            with open(caminho, "rb") as f:
                modelo_completo = pickle.load(f)

            self.modelo_ml = modelo_completo["modelo"]
            self.preprocessors = modelo_completo["preprocessors"]
            self.feature_names = modelo_completo["feature_names"]

            print(f"✅ Modelo ML carregado: {len(self.feature_names)} features")
        except Exception as e:
            print(f"⚠️ Erro ao carregar modelo ML: {e}")
            print("Sistema continuará apenas com regras")

    def _calcular_score_regras(self, dados: Dict) -> Tuple[float, Dict]:
        """
        Calcula score baseado em regras especializadas

        Args:
            dados: Dicionário com as respostas do estudante

        Returns:
            (score_total, detalhamento_por_regra)
        """
        scores_regras = {}

        # REGRA 1: Transporte Difícil
        transporte = str(
            dados.get("Como é o seu deslocamento até a universidade?", "")
        ).lower()
        tipo_transporte = str(
            dados.get("Com relação ao transporte do item anterior, ele é:", "")
        ).lower()

        score_transporte = 0.0
        if "ônibus" in transporte or "público" in tipo_transporte:
            score_transporte = 0.6
        elif "carona" in transporte or "dependo" in tipo_transporte:
            score_transporte = 0.7
        elif "a pé" in transporte or "caminhando" in transporte:
            score_transporte = 0.5

        scores_regras["transporte_dificil"] = score_transporte

        # REGRA 2: Trabalho em Horários Conflitantes
        trabalha = str(dados.get("Você trabalha?", "")).lower()
        horario_trabalho = str(
            dados.get("Se você trabalha, em quais horários?", "")
        ).lower()

        score_trabalho = 0.0
        if "sim" in trabalha or trabalha == "1":
            if "integral" in horario_trabalho or "tempo integral" in horario_trabalho:
                score_trabalho = 0.8
            elif "manhã" in horario_trabalho or "tarde" in horario_trabalho:
                score_trabalho = 0.6
            elif "noite" in horario_trabalho:
                score_trabalho = 0.4
            else:
                score_trabalho = 0.5

        scores_regras["trabalho_conflitante"] = score_trabalho

        # REGRA 3: Responsabilidades Familiares
        casado = str(dados.get("É casado(a)/está em união estável?", "")).lower()
        tem_filhos = str(dados.get("Tem filhos?", "")).lower()
        sustenta_familia = str(
            dados.get("Você contribui para o sustento financeiro da família?", "")
        ).lower()

        score_familia = 0.0
        contador_responsabilidades = 0

        if "sim" in casado or casado == "1":
            contador_responsabilidades += 1
        if "sim" in tem_filhos or tem_filhos == "1":
            contador_responsabilidades += 2
        if "sim" in sustenta_familia or sustenta_familia == "1":
            contador_responsabilidades += 1

        score_familia = min(contador_responsabilidades / 4, 1.0)
        scores_regras["responsabilidades_familiares"] = score_familia

        # REGRA 4: Falta de Tempo para Estudo
        tempo_estudo = str(
            dados.get(
                "Em relação ao tempo necessário como discente para dedicar no estudo?",
                "",
            )
        ).lower()

        score_tempo = 0.0
        if "insuficiente" in tempo_estudo or "pouco" in tempo_estudo:
            score_tempo = 0.8
        elif "razoável" in tempo_estudo or "adequado" in tempo_estudo:
            score_tempo = 0.3
        elif "suficiente" in tempo_estudo or "muito" in tempo_estudo:
            score_tempo = 0.1

        scores_regras["falta_tempo_estudo"] = score_tempo

        # REGRA 5: Distância do Campus
        mora_angicos = str(dados.get("Você mora em Angicos?", "")).lower()

        score_distancia = 0.0
        if "não" in mora_angicos or mora_angicos == "0":
            score_distancia = 0.7

        scores_regras["distancia_campus"] = score_distancia

        # REGRA 6: Acessibilidade Ruim
        acessibilidade = str(
            dados.get("Como você considera a acessibilidade do Campus?", "")
        ).lower()

        score_acessibilidade = 0.0
        if "ruim" in acessibilidade or "péssima" in acessibilidade:
            score_acessibilidade = 0.8
        elif "regular" in acessibilidade or "razoável" in acessibilidade:
            score_acessibilidade = 0.5
        elif "boa" in acessibilidade or "excelente" in acessibilidade:
            score_acessibilidade = 0.2

        scores_regras["acessibilidade_ruim"] = score_acessibilidade

        # REGRA 7: Sobrecarga Total
        fatores_alto_risco = sum(
            [1 if score > 0.6 else 0 for score in scores_regras.values()]
        )

        score_sobrecarga = 0.0
        if fatores_alto_risco >= 4:
            score_sobrecarga = 0.9
        elif fatores_alto_risco >= 3:
            score_sobrecarga = 0.7
        elif fatores_alto_risco >= 2:
            score_sobrecarga = 0.5

        scores_regras["sobrecarga_total"] = score_sobrecarga

        # Calcular score ponderado final
        score_final = sum(
            scores_regras[regra] * self.pesos_regras[regra]
            for regra in self.pesos_regras.keys()
        )

        return score_final, scores_regras

    def _calcular_score_ml(self, dados: Dict) -> float:

        if self.modelo_ml is None:
            return 0.5

        try:
            # Criar mapeamento numérico dos dados de texto
            dados_numericos = {}

            # Mapeamento de transporte
            mapa_transporte_num = {
                "carro próprio": 0,
                "carro": 0,
                "moto própria": 1,
                "moto": 1,
                "ônibus": 2,
                "a pé": 3,
                "outro": 4,
            }

            # Mapeamento de propriedade
            mapa_propriedade_num = {
                "próprio": 0,
                "público": 1,
                "dependo de terceiros": 2,
                "cedido": 0,
                "particular": 2,
            }

            # Mapeamento mora em Angicos
            mapa_mora_num = {"sim": 0, "não": 1}

            # Mapeamento acessibilidade
            mapa_acess_num = {
                "boa": 0,
                "excelente": 0,
                "adequada": 0,
                "ruim": 1,
                "péssima": 1,
                "regular": 1,
            }

            # Processar cada campo
            transporte_val = str(
                dados.get("Como é o seu deslocamento até a universidade?", "")
            ).lower()
            dados_numericos["transporte"] = mapa_transporte_num.get(transporte_val, 0)

            tipo_val = str(
                dados.get("Com relação ao transporte do item anterior, ele é:", "")
            ).lower()
            dados_numericos["tipo_transporte"] = mapa_propriedade_num.get(tipo_val, 0)

            mora_val = str(dados.get("Você mora em Angicos?", "")).lower()
            dados_numericos["mora_angicos"] = mapa_mora_num.get(mora_val, 0)

            acess_val = str(
                dados.get("Como você considera a acessibilidade do Campus?", "")
            ).lower()
            dados_numericos["acessibilidade"] = mapa_acess_num.get(acess_val, 0)

            # Tempo de estudo (categórico 0-3)
            tempo_val = str(
                dados.get(
                    "Em relação ao tempo necessário como discente para dedicar no estudo?",
                    "",
                )
            ).lower()
            if "suficiente" in tempo_val and "in" not in tempo_val.replace(
                "suficiente", ""
            ):
                dados_numericos["tempo_estudo"] = 0
            elif "insuficiente" in tempo_val and "não consigo" not in tempo_val:
                dados_numericos["tempo_estudo"] = 1
            elif "razoável" in tempo_val:
                dados_numericos["tempo_estudo"] = 1
            else:
                dados_numericos["tempo_estudo"] = 2

            # Trabalha (binário)
            trabalha_val = str(dados.get("Você trabalha?", "")).lower()
            dados_numericos["trabalha"] = 1 if "sim" in trabalha_val else 0

            # Horário trabalho (categórico)
            horario_val = str(
                dados.get("Se você trabalha, em quais horários?", "")
            ).lower()
            if "integral" in horario_val:
                dados_numericos["horario_trabalho"] = 0
            elif "parcial" in horario_val or "turno" in horario_val:
                dados_numericos["horario_trabalho"] = 1
            else:
                dados_numericos["horario_trabalho"] = 2

            # Casado (binário)
            casado_val = str(
                dados.get("É casado(a)/está em união estável?", "")
            ).lower()
            dados_numericos["casado"] = 1 if "sim" in casado_val else 0

            # Filhos (binário)
            filhos_val = str(dados.get("Tem filhos?", "")).lower()
            dados_numericos["filhos"] = 1 if "sim" in filhos_val else 0

            # Sustento financeiro (categórico)
            sustento_val = str(
                dados.get("Você contribui para o sustento financeiro da família?", "")
            ).lower()
            if "único" in sustento_val:
                dados_numericos["sustento"] = 0
            elif "principal" in sustento_val:
                dados_numericos["sustento"] = 1
            elif "não" in sustento_val:
                dados_numericos["sustento"] = 3
            else:
                dados_numericos["sustento"] = 2

            # Criar DataFrame com valores numéricos
            X = pd.DataFrame([dados_numericos])

            # Garantir que todas as features estão presentes
            for feature in self.feature_names:
                if feature not in X.columns:
                    X[feature] = 0

            # Reordenar colunas
            X = X[self.feature_names]

            # Aplicar imputer
            if self.preprocessors and "imputer" in self.preprocessors:
                X_imputed = self.preprocessors["imputer"].transform(X)
            else:
                X_imputed = X.values

            # Aplicar scaler
            if self.preprocessors and "scaler" in self.preprocessors:
                X_scaled = self.preprocessors["scaler"].transform(X_imputed)
            else:
                X_scaled = X_imputed

            # Predição
            prob_evasao = self.modelo_ml.predict_proba(X_scaled)[0][1]

            return prob_evasao

        except Exception as e:
            print(f"⚠️ Erro no modelo ML: {e}")
            import traceback

            traceback.print_exc()
            return 0.5

    def predizer(self, dados: Dict, detalhado: bool = True) -> Dict:
        """
        Realiza predição híbrida combinando ML e regras

        Args:
            dados: Dicionário com respostas do estudante
            detalhado: Se True, retorna explicação detalhada

        Returns:
            Dicionário com resultado da predição
        """
        score_ml = self._calcular_score_ml(dados)
        score_regras, detalhes_regras = self._calcular_score_regras(dados)
        score_final = self.pesos["ml"] * score_ml + self.pesos["regras"] * score_regras

        if score_final < 0.35:
            categoria = "BAIXO"
            cor = "🟢"
            recomendacao = "Estudante com boas condições de permanência"
        elif score_final < 0.55:
            categoria = "MODERADO"
            cor = "🟡"
            recomendacao = "Monitorar e oferecer suporte preventivo"
        elif score_final < 0.75:
            categoria = "ALTO"
            cor = "🟠"
            recomendacao = "Intervenção recomendada - acompanhamento próximo"
        else:
            categoria = "CRÍTICO"
            cor = "🔴"
            recomendacao = "Intervenção urgente necessária"

        resultado = {
            "score_final": score_final,
            "probabilidade_evasao": f"{score_final*100:.1f}%",
            "categoria_risco": categoria,
            "cor": cor,
            "recomendacao": recomendacao,
            "score_ml": score_ml,
            "score_regras": score_regras,
        }

        if detalhado:
            fatores_risco = sorted(
                detalhes_regras.items(), key=lambda x: x[1], reverse=True
            )

            principais_fatores = [
                {
                    "fator": self._traduzir_fator(fator),
                    "severidade": severidade,
                    "impacto": f"{severidade * self.pesos_regras[fator] * 100:.0f}%",
                }
                for fator, severidade in fatores_risco
                if severidade > 0.3
            ]

            resultado["fatores_risco"] = principais_fatores
            resultado["detalhes_regras"] = detalhes_regras
            resultado["confianca_ml"] = (
                "Baixa" if score_ml > 0.4 and score_ml < 0.6 else "Moderada"
            )

        return resultado

    def _traduzir_fator(self, fator: str) -> str:
        """Traduz nome técnico do fator para descrição amigável"""
        traducoes = {
            "transporte_dificil": "Dificuldades com Transporte",
            "trabalho_conflitante": "Trabalho em Horário Conflitante",
            "responsabilidades_familiares": "Responsabilidades Familiares",
            "falta_tempo_estudo": "Tempo Insuficiente para Estudo",
            "distancia_campus": "Distância do Campus",
            "acessibilidade_ruim": "Problemas de Acessibilidade",
            "sobrecarga_total": "Sobrecarga de Múltiplos Fatores",
        }
        return traducoes.get(fator, fator)

    def gerar_relatorio(self, resultado: Dict) -> str:
        """
        Gera relatório textual da predição

        Args:
            resultado: Resultado da predição

        Returns:
            String formatada com o relatório
        """
        relatorio = f"""
{'='*60}
RELATÓRIO DE ANÁLISE DE RISCO DE EVASÃO
{'='*60}

{resultado['cor']} RISCO: {resultado['categoria_risco']}
Probabilidade de Evasão: {resultado['probabilidade_evasao']}

COMPOSIÇÃO DO SCORE:
  • Machine Learning:     {resultado['score_ml']*100:5.1f}% (peso 30%)
  • Regras Especializadas: {resultado['score_regras']*100:5.1f}% (peso 70%)
  • Score Final:           {resultado['score_final']*100:5.1f}%

RECOMENDAÇÃO:
{resultado['recomendacao']}
"""

        if "fatores_risco" in resultado and resultado["fatores_risco"]:
            relatorio += f"\nPRINCIPAIS FATORES DE RISCO IDENTIFICADOS:\n"
            for i, fator in enumerate(resultado["fatores_risco"][:5], 1):
                relatorio += (
                    f"  {i}. {fator['fator']:<35} (Impacto: {fator['impacto']})\n"
                )
        else:
            relatorio += f"\nNenhum fator de risco significativo identificado.\n"

        relatorio += f"\n{'='*60}\n"
        relatorio += (
            "Nota: Sistema híbrido combina ML com regras baseadas em literatura\n"
        )
        relatorio += (
            "sobre retenção estudantil (Tinto, Bean & Metzner, Cabrera et al.)\n"
        )
        relatorio += f"{'='*60}\n"

        return relatorio


if __name__ == "__main__":
    print("Sistema Híbrido de Evasão - Pronto para uso")
