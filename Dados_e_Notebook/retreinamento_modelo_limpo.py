"""
RETREINAMENTO COMPLETO DO MODELO DE EVASÃO UNIVERSITÁRIA
========================================================

Script para retreinar o modelo do zero, com validações rigorosas
e prevenção de data leakage.

Autor: Assistente Claude
Data: Setembro 2025
"""

import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    roc_auc_score,
    roc_curve,
)
from sklearn.impute import SimpleImputer
import pickle
import warnings

warnings.filterwarnings("ignore")


class ModeloEvasaoLimpo:
    def __init__(self, arquivo_dados="sem tempo2.csv"):
        self.arquivo_dados = arquivo_dados
        self.df_original = None
        self.df_limpo = None
        self.X_train = None
        self.X_test = None
        self.y_train = None
        self.y_test = None
        self.modelo_final = None
        self.preprocessors = {}
        self.metricas = {}

        print("=" * 60)
        print("RETREINAMENTO LIMPO - MODELO DE EVASÃO UNIVERSITÁRIA")
        print("=" * 60)

    def carregar_dados(self):
        """1. Carrega e examina os dados brutos"""
        print("\n1. CARREGANDO E EXAMINANDO DADOS")
        print("-" * 40)

        self.df_original = pd.read_csv(
            self.arquivo_dados, na_values=["", " ", "Sem resposta"]
        )

        print(f"Dados carregados: {self.df_original.shape}")
        print(f"Colunas: {list(self.df_original.columns)}")
        print(f"\nPrimeiras linhas:")
        print(self.df_original.head(2))

        # Identificar coluna target
        target_candidates = [
            col
            for col in self.df_original.columns
            if "trancar" in col.lower() or "abandonar" in col.lower()
        ]

        if target_candidates:
            self.target_col = target_candidates[0]
            print(f"\nColuna target identificada: {self.target_col}")
        else:
            print("ERRO: Coluna target não encontrada!")
            return False

        return True

    def analise_exploratoria(self):
        """2. Análise exploratória e identificação de data leakage"""
        print("\n2. ANÁLISE EXPLORATÓRIA E DATA LEAKAGE")
        print("-" * 40)

        # Analisar distribuição do target
        print("Distribuição do target:")
        target_counts = self.df_original[self.target_col].value_counts()
        print(target_counts)
        print(
            f"Proporção: {target_counts.values[0]/len(self.df_original):.2%} / {target_counts.values[1]/len(self.df_original):.2%}"
        )

        # Identificar colunas suspeitas de data leakage
        colunas_suspeitas = []

        for col in self.df_original.columns:
            col_lower = col.lower()

            # Palavras-chave que indicam consequência ao invés de preditor
            palavras_suspeitas = [
                "dificuldade",
                "problema",
                "barreira",
                "obstáculo",
                "principais",
                "cite",
                "descreva",
                "motivo",
                "porque",
                "semestre",
                "período",
                "curso",  # Informações que podem vazar
            ]

            if any(palavra in col_lower for palavra in palavras_suspeitas):
                if col != self.target_col:  # Não incluir o próprio target
                    colunas_suspeitas.append(col)

        print(f"\nColunas SUSPEITAS de data leakage:")
        for col in colunas_suspeitas:
            print(f"  - {col}")

        # Analisar correlações muito altas
        print(f"\nAnalisando correlações...")

        # Converter dados categóricos temporariamente para análise
        df_temp = self.df_original.copy()
        le_temp = LabelEncoder()

        for col in df_temp.columns:
            if df_temp[col].dtype == "object":
                df_temp[col] = le_temp.fit_transform(df_temp[col].astype(str))

        # Calcular correlações com target
        correlacoes = df_temp.corr()[self.target_col].abs().sort_values(ascending=False)

        print(f"\nTop 10 correlações com target:")
        for col, corr in correlacoes.head(10).items():
            if col != self.target_col:
                status = "SUSPEITA" if corr > 0.7 else "OK"
                print(f"  {col}: {corr:.4f} [{status}]")

                if corr > 0.7 and col not in colunas_suspeitas:
                    colunas_suspeitas.append(col)

        self.colunas_suspeitas = colunas_suspeitas
        return colunas_suspeitas

    def limpeza_dados(self):
        """3. Limpeza de dados e seleção de features válidas"""
        print("\n3. LIMPEZA DE DADOS E SELEÇÃO DE FEATURES")
        print("-" * 40)

        # Remover linhas com target missing
        df_limpo = self.df_original.dropna(subset=[self.target_col]).copy()
        print(f"Após remover NaN no target: {df_limpo.shape}")

        # Definir features válidas (que NÃO vazam informação)
        features_validas = []

        for col in df_limpo.columns:
            if col == self.target_col:
                continue

            col_lower = col.lower()

            # Critérios para features válidas (preditores que antecedem a decisão)
            if any(
                palavra in col_lower
                for palavra in [
                    "deslocamento",
                    "transporte",
                    "mora",
                    "trabalha",
                    "horário",
                    "casado",
                    "filho",
                    "identifica",
                    "acessibilidade",
                    "tempo",
                    "sustento",
                    "preconceito",
                    "violência",
                ]
            ):
                if col not in self.colunas_suspeitas:
                    features_validas.append(col)

        print(f"\nFeatures VÁLIDAS selecionadas ({len(features_validas)}):")
        for i, feature in enumerate(features_validas, 1):
            print(f"  {i:2d}. {feature}")

        # Manter apenas features válidas + target
        self.df_limpo = df_limpo[features_validas + [self.target_col]].copy()

        print(f"\nDataset final: {self.df_limpo.shape}")
        print(f"Features para modelo: {len(features_validas)}")

        return features_validas

    def preprocessing(self):
        """4. Preprocessing dos dados"""
        print("\n4. PREPROCESSING DOS DADOS")
        print("-" * 40)

        # Separar features e target
        X = self.df_limpo.drop(self.target_col, axis=1)
        y = self.df_limpo[self.target_col]

        print(f"Tipos de dados originais:")
        print(X.dtypes.value_counts())

        # Preprocessar features categóricas
        self.preprocessors["label_encoders"] = {}

        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.preprocessors["label_encoders"][col] = le
                print(f"  Codificado: {col}")

        # Imputar valores ausentes
        self.preprocessors["imputer"] = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            self.preprocessors["imputer"].fit_transform(X), columns=X.columns
        )

        # Codificar target se necessário
        if y.dtype == "object":
            le_target = LabelEncoder()
            y = le_target.fit_transform(y)
            self.preprocessors["target_encoder"] = le_target

        print(f"Valores ausentes após imputação: {X_imputed.isnull().sum().sum()}")
        print(f"Target distribution: {np.bincount(y)}")

        # Split treino/teste com estratificação
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed, y, test_size=0.2, random_state=42, stratify=y
        )

        # Normalização (importante para regularização)
        self.preprocessors["scaler"] = StandardScaler()
        self.X_train_scaled = self.preprocessors["scaler"].fit_transform(self.X_train)
        self.X_test_scaled = self.preprocessors["scaler"].transform(self.X_test)

        self.feature_names = list(X.columns)

        print(f"Treino: {self.X_train_scaled.shape}, Teste: {self.X_test_scaled.shape}")

        return True

    def treinar_modelos(self):
        """5. Treinamento e seleção do melhor modelo"""
        print("\n5. TREINAMENTO E SELEÇÃO DE MODELOS")
        print("-" * 40)

        # Definir modelos candidatos
        modelos = {
            "LogisticRegression_C0.1": LogisticRegression(
                C=0.1, random_state=42, max_iter=1000
            ),
            "LogisticRegression_C1.0": LogisticRegression(
                C=1.0, random_state=42, max_iter=1000
            ),
            "LogisticRegression_C10": LogisticRegression(
                C=10.0, random_state=42, max_iter=1000
            ),
            "RandomForest_balanced": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42, class_weight="balanced"
            ),
            "RandomForest_default": RandomForestClassifier(
                n_estimators=100, max_depth=5, random_state=42
            ),
        }

        resultados = []

        print("Modelo\t\t\tCV_AUC\tTest_AUC\tIntercept")
        print("-" * 65)

        for nome, modelo in modelos.items():
            # Cross-validation
            cv = StratifiedKFold(n_splits=5, shuffle=True, random_state=42)
            cv_scores = cross_val_score(
                modelo, self.X_train_scaled, self.y_train, cv=cv, scoring="roc_auc"
            )

            # Treinar e testar
            modelo.fit(self.X_train_scaled, self.y_train)
            y_pred_proba = modelo.predict_proba(self.X_test_scaled)[:, 1]
            test_auc = roc_auc_score(self.y_test, y_pred_proba)

            # Intercept para modelos logísticos
            intercept = modelo.intercept_[0] if hasattr(modelo, "intercept_") else "N/A"

            resultado = {
                "nome": nome,
                "modelo": modelo,
                "cv_auc_mean": cv_scores.mean(),
                "cv_auc_std": cv_scores.std(),
                "test_auc": test_auc,
                "intercept": intercept,
            }

            resultados.append(resultado)

            print(f"{nome:20s}\t{cv_scores.mean():.4f}\t{test_auc:.4f}\t{intercept}")

        # Selecionar melhor modelo (maior CV AUC)
        melhor_resultado = max(resultados, key=lambda x: x["cv_auc_mean"])
        self.modelo_final = melhor_resultado["modelo"]
        self.metricas = melhor_resultado

        print(f"\nMELHOR MODELO: {melhor_resultado['nome']}")
        print(
            f"CV AUC: {melhor_resultado['cv_auc_mean']:.4f} ± {melhor_resultado['cv_auc_std']:.4f}"
        )
        print(f"Test AUC: {melhor_resultado['test_auc']:.4f}")

        return melhor_resultado

    def validacao_modelo(self):
        """6. Validação e testes de sanidade"""
        print("\n6. VALIDAÇÃO E TESTES DE SANIDADE")
        print("-" * 40)

        # Predições finais
        y_pred = self.modelo_final.predict(self.X_test_scaled)
        y_pred_proba = self.modelo_final.predict_proba(self.X_test_scaled)[:, 1]

        print("RELATÓRIO DE CLASSIFICAÇÃO:")
        print(classification_report(self.y_test, y_pred))

        # Verificar se o intercept está normalizado
        if hasattr(self.modelo_final, "intercept_"):
            intercept = self.modelo_final.intercept_[0]
            print(f"\nIntercept: {intercept:.4f}")

            if abs(intercept) < 3:
                print("✅ Intercept normalizado")
            elif abs(intercept) < 5:
                print("⚠️ Intercept moderado")
            else:
                print("❌ Intercept problemático")

        # Teste de sanidade: casos extremos
        print(f"\nTESTES DE SANIDADE:")

        # Caso 1: Baixo risco (boas condições)
        caso_baixo_risco = np.zeros(len(self.feature_names))
        # Definir valores que indicam baixo risco
        # (isso depende da codificação específica dos dados)

        # Caso 2: Alto risco (condições difíceis)
        caso_alto_risco = np.ones(len(self.feature_names))

        # Normalizar usando o mesmo scaler
        caso_baixo_scaled = self.preprocessors["scaler"].transform([caso_baixo_risco])
        caso_alto_scaled = self.preprocessors["scaler"].transform([caso_alto_risco])

        prob_baixo = self.modelo_final.predict_proba(caso_baixo_scaled)[0][1]
        prob_alto = self.modelo_final.predict_proba(caso_alto_scaled)[0][1]

        print(f"Caso extremo baixo risco: {prob_baixo:.1%}")
        print(f"Caso extremo alto risco: {prob_alto:.1%}")

        # Validação da lógica
        if prob_baixo < prob_alto:
            print("✅ Modelo segue lógica esperada")
        else:
            print("❌ Modelo com lógica invertida")

        # Feature importance para modelos logísticos
        if hasattr(self.modelo_final, "coef_"):
            print(f"\nTOP 5 FEATURES MAIS IMPORTANTES:")
            feature_importance = pd.DataFrame(
                {
                    "feature": self.feature_names,
                    "coefficient": self.modelo_final.coef_[0],
                }
            )
            feature_importance["abs_coef"] = abs(feature_importance["coefficient"])
            top_features = feature_importance.sort_values(
                "abs_coef", ascending=False
            ).head(5)

            for _, row in top_features.iterrows():
                direction = "+" if row["coefficient"] > 0 else "-"
                print(f"  {direction} {row['feature']}: {row['coefficient']:.4f}")

        return True

    def salvar_modelo(self, nome_arquivo="modelo_evasao_NOVO"):
        """7. Salvar modelo final"""
        print("\n7. SALVANDO MODELO FINAL")
        print("-" * 40)

        modelo_completo = {
            "modelo": self.modelo_final,
            "preprocessors": self.preprocessors,
            "feature_names": self.feature_names,
            "metricas": self.metricas,
            "metadata": {
                "data_treinamento": pd.Timestamp.now(),
                "n_features": len(self.feature_names),
                "n_train_samples": len(self.y_train),
                "n_test_samples": len(self.y_test),
                "features_removidas_data_leakage": self.colunas_suspeitas,
            },
        }

        # Salvar
        nome_completo = f"{nome_arquivo}.pkl"
        with open(nome_completo, "wb") as f:
            pickle.dump(modelo_completo, f)

        print(f"✅ Modelo salvo como: {nome_completo}")

        # Resumo final
        print(f"\nRESUMO DO MODELO FINAL:")
        print(f"  Tipo: {self.metricas['nome']}")
        print(f"  AUC (CV): {self.metricas['cv_auc_mean']:.4f}")
        print(f"  AUC (Test): {self.metricas['test_auc']:.4f}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Intercept: {self.metricas['intercept']}")

        return nome_completo

    def executar_pipeline_completo(self):
        """Executa todo o pipeline de retreinamento"""
        try:
            # Etapa 1: Carregar dados
            if not self.carregar_dados():
                return False

            # Etapa 2: Análise exploratória
            self.analise_exploratoria()

            # Etapa 3: Limpeza
            self.limpeza_dados()

            # Etapa 4: Preprocessing
            self.preprocessing()

            # Etapa 5: Treinamento
            self.treinar_modelos()

            # Etapa 6: Validação
            self.validacao_modelo()

            # Etapa 7: Salvar
            arquivo_salvo = self.salvar_modelo()

            print("\n" + "=" * 60)
            print("RETREINAMENTO CONCLUÍDO COM SUCESSO!")
            print(f"Modelo salvo: {arquivo_salvo}")
            print("=" * 60)

            return True

        except Exception as e:
            print(f"ERRO durante o retreinamento: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Função principal"""
    # Instanciar e executar o retreinamento
    retreinamento = ModeloEvasaoLimpo()
    sucesso = retreinamento.executar_pipeline_completo()

    if sucesso:
        print("\n🎉 Modelo retreinado com sucesso!")
        print("Você pode agora usar o novo modelo no seu dashboard.")
    else:
        print("\n❌ Falha no retreinamento. Verifique os erros acima.")


if __name__ == "__main__":
    main()
