"""
RETREINAMENTO CORRIGIDO - Modelo de Evasão Universitária
==========================================================
Com codificação explícita e consistente dos labels
"""

import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split, cross_val_score, StratifiedKFold
from sklearn.preprocessing import StandardScaler, LabelEncoder
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.impute import SimpleImputer
import pickle
import warnings

warnings.filterwarnings("ignore")


class ModeloEvasaoCorrigido:
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

        print("=" * 70)
        print("RETREINAMENTO CORRIGIDO - MODELO DE EVASÃO")
        print("=" * 70)

    def carregar_dados(self):
        """1. Carrega e examina os dados brutos"""
        print("\n1. CARREGANDO DADOS")
        print("-" * 70)

        self.df_original = pd.read_csv(
            self.arquivo_dados, na_values=["", " ", "Sem resposta"]
        )

        print(f"Dados carregados: {self.df_original.shape}")
        print(f"Colunas disponíveis: {len(self.df_original.columns)}")

        # Identificar coluna target
        target_candidates = [
            col
            for col in self.df_original.columns
            if "trancar" in col.lower() or "abandonar" in col.lower()
        ]

        if target_candidates:
            self.target_col = target_candidates[0]
            print(f"✅ Coluna target: {self.target_col}")
        else:
            print("❌ ERRO: Coluna target não encontrada!")
            return False

        return True

    def analisar_e_codificar_target(self):
        """2. Analisa e codifica o target de forma EXPLÍCITA"""
        print("\n2. ANÁLISE E CODIFICAÇÃO DO TARGET")
        print("-" * 70)

        # Remover linhas com target missing
        df_limpo = self.df_original.dropna(subset=[self.target_col]).copy()
        print(f"Após remover NaN no target: {df_limpo.shape}")

        # Analisar valores únicos do target
        print(f"\n📊 Valores únicos no target original:")
        valores_target = df_limpo[self.target_col].value_counts()
        print(valores_target)

        # VERIFICAR SE JÁ ESTÁ CODIFICADO NUMERICAMENTE
        valores_unicos = df_limpo[self.target_col].unique()
        ja_codificado = all(
            v in [0, 1, 0.0, 1.0] for v in valores_unicos if pd.notna(v)
        )

        if ja_codificado:
            print("\n✅ Target JÁ está codificado como 0/1")
            print("   IMPORTANTE: Invertendo labels para consistência...")
            print(
                "   (No CSV: 0=pensou, 1=não pensou → Invertendo para: 0=não pensou, 1=pensou)"
            )

            # NÃO inverter - usar labels originais do CSV
            df_limpo["target_codificado"] = 1 - df_limpo[self.target_col].astype(int)

        else:
            print("\n📝 Target contém texto. Aplicando função de codificação...")

            # CODIFICAÇÃO BASEADA EM TEXTO
            def codificar_resposta_evasao(resposta):
                """
                Codifica a resposta de forma consistente:
                - 0 = NÃO pensou em evasão (baixo risco)
                - 1 = SIM pensou em evasão (alto risco)
                """
                if pd.isna(resposta):
                    return np.nan

                # Se já for número, retornar direto
                if isinstance(resposta, (int, float, np.integer, np.floating)):
                    return int(resposta)

                resposta_lower = str(resposta).lower().strip()

                # Respostas curtas/vazias = não pensou
                if len(resposta_lower) < 3:
                    return 0

                # Analisa o início da resposta (primeiras palavras são mais indicativas)
                inicio = resposta_lower[:30]

                # Palavras que indicam que SIM pensou em evasão
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

                # Palavras que indicam que NÃO pensou em evasão
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

                # Verificar palavras de evasão forte em qualquer parte
                palavras_evasao_forte = [
                    "trancar",
                    "abandonar",
                    "desistir",
                    "parar",
                    "sair",
                    "largar",
                    "deixar o curso",
                ]

                # Prioridade 1: Início da resposta
                if any(palavra in inicio for palavra in palavras_sim):
                    return 1
                if any(palavra in inicio for palavra in palavras_nao):
                    return 0

                # Prioridade 2: Palavras de evasão forte em qualquer lugar
                if any(palavra in resposta_lower for palavra in palavras_evasao_forte):
                    return 1

                # Prioridade 3: Se não encontrou indicadores claros, assumir que não pensou
                return 0

            # Aplicar codificação
            df_limpo["target_codificado"] = df_limpo[self.target_col].apply(
                codificar_resposta_evasao
            )

        # Verificar resultado da codificação
        print(f"\n📊 Distribuição do target CODIFICADO:")
        distribuicao = df_limpo["target_codificado"].value_counts()
        print(f"Classe 0 (NÃO pensou em evasão): {distribuicao.get(0, 0)}")
        print(f"Classe 1 (SIM pensou em evasão): {distribuicao.get(1, 0)}")

        total = len(df_limpo)
        prop_0 = distribuicao.get(0, 0) / total
        prop_1 = distribuicao.get(1, 0) / total
        print(f"Proporção: {prop_0:.1%} / {prop_1:.1%}")

        # Validação
        if distribuicao.get(1, 0) == 0:
            print("\n❌ ATENÇÃO: Apenas uma classe detectada! Verifique a codificação.")
            print("   Isso impedirá o treinamento do modelo.")
            return False

        # Validar exemplos
        print(f"\n✅ VALIDAÇÃO - Exemplos de codificação:")
        exemplos = df_limpo.sample(min(5, len(df_limpo)))
        for idx, row in exemplos.iterrows():
            resposta_original = row[self.target_col]
            codigo = row["target_codificado"]
            label = "SIM pensou" if codigo == 1 else "NÃO pensou"

            if isinstance(resposta_original, str):
                resposta_str = resposta_original[:50]
            else:
                resposta_str = str(resposta_original)

            print(f"  • '{resposta_str}...' → {codigo} ({label})")

        self.df_limpo = df_limpo
        return True

    def limpeza_e_features(self):
        """3. Remove data leakage e seleciona features válidas"""
        print("\n3. LIMPEZA E SELEÇÃO DE FEATURES")
        print("-" * 70)

        # Definir todas as colunas a serem removidas
        colunas_para_remover = [
            # Data leakage
            "Carimbo de data/hora",
            "Curso",
            "Semestre de Ingresso",
            "Cite as principais dificuldades para sua permanência no Curso",
            self.target_col,
            "target_codificado",  # A coluna que criamos, não é uma feature
            # Feature redundante e confusa
            "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Não",
        ]

        # Features válidas são todas as colunas exceto as que vamos remover
        features_validas = [
            col for col in self.df_limpo.columns if col not in colunas_para_remover
        ]

        print(f"✅ Features selecionadas: {len(features_validas)}")
        for i, feat in enumerate(features_validas, 1):
            print(f"  {i:2d}. {feat}")

        # Criar dataset final
        self.df_final = self.df_limpo[features_validas + ["target_codificado"]].copy()
        self.feature_names = features_validas

        print(f"\n✅ Dataset final: {self.df_final.shape}")
        return True

    def preprocessing(self):
        """4. Preprocessing completo"""
        print("\n4. PREPROCESSING")
        print("-" * 70)

        # Separar X e y
        X = self.df_final.drop("target_codificado", axis=1)
        y = self.df_final["target_codificado"]

        # Verificar se há NaN no target
        if y.isna().any():
            print("⚠️ Removendo NaN remanescentes no target...")
            mask = ~y.isna()
            X = X[mask]
            y = y[mask]

        print(f"Tipos de dados: {X.dtypes.value_counts().to_dict()}")

        # Label Encoding para categóricas
        self.preprocessors["label_encoders"] = {}
        for col in X.columns:
            if X[col].dtype == "object":
                le = LabelEncoder()
                X[col] = le.fit_transform(X[col].astype(str))
                self.preprocessors["label_encoders"][col] = le

        # Imputação
        self.preprocessors["imputer"] = SimpleImputer(strategy="median")
        X_imputed = pd.DataFrame(
            self.preprocessors["imputer"].fit_transform(X), columns=X.columns
        )

        print(f"Valores ausentes após imputação: {X_imputed.isnull().sum().sum()}")

        # Split com estratificação
        self.X_train, self.X_test, self.y_train, self.y_test = train_test_split(
            X_imputed,
            y.astype(int),
            test_size=0.2,
            random_state=42,
            stratify=y.astype(int),
        )

        # Normalização
        self.preprocessors["scaler"] = StandardScaler()
        self.X_train_scaled = self.preprocessors["scaler"].fit_transform(self.X_train)
        self.X_test_scaled = self.preprocessors["scaler"].transform(self.X_test)

        print(f"Train: {self.X_train_scaled.shape}, Test: {self.X_test_scaled.shape}")
        print(f"Distribuição y_train: {np.bincount(self.y_train.astype(int)).tolist()}")
        print(f"Distribuição y_test: {np.bincount(self.y_test.astype(int)).tolist()}")

        return True

    def treinar_modelos(self):
        """5. Treina múltiplos modelos e seleciona o melhor"""
        print("\n5. TREINAMENTO DE MODELOS")
        print("-" * 70)

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
        }

        resultados = []
        print(f"{'Modelo':<30} {'CV_AUC':<10} {'Test_AUC':<10}")
        print("-" * 70)

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

            resultados.append(
                {
                    "nome": nome,
                    "modelo": modelo,
                    "cv_auc_mean": cv_scores.mean(),
                    "cv_auc_std": cv_scores.std(),
                    "test_auc": test_auc,
                }
            )

            print(f"{nome:<30} {cv_scores.mean():.4f}    {test_auc:.4f}")

        print("\n" + "!" * 70)
        print("SELEÇÃO DE MODELO MANUAL: Forçando o uso do RandomForest_balanced")
        print("!" * 70 + "\n")

        # Encontra o RandomForest nos resultados
        modelo_rf = next(
            item for item in resultados if item["nome"] == "RandomForest_balanced"
        )

        self.modelo_final = modelo_rf["modelo"]
        self.metricas = modelo_rf

        print(f"\n🏆 MODELO SELECIONADO (MANUAL): {modelo_rf['nome']}")
        print(
            f"   CV AUC: {modelo_rf['cv_auc_mean']:.4f} ± {modelo_rf['cv_auc_std']:.4f}"
        )
        print(f"   Test AUC: {modelo_rf['test_auc']:.4f}")

        return True

    def validar_sanidade(self):
        """6. Testes de sanidade do modelo"""
        print("\n6. TESTES DE SANIDADE")
        print("-" * 70)

        # Predições no test set
        y_pred = self.modelo_final.predict(self.X_test_scaled)
        y_pred_proba = self.modelo_final.predict_proba(self.X_test_scaled)[:, 1]

        print("Classification Report:")
        print(classification_report(self.y_test, y_pred, target_names=["Não", "Sim"]))

        # Teste: casos extremos
        print("\n🧪 TESTES DE CASOS EXTREMOS:")

        # Caso 1: Tudo favorável (baixo risco) → deve dar prob BAIXA classe 1
        caso_baixo = np.zeros(len(self.feature_names))
        caso_baixo_scaled = self.preprocessors["scaler"].transform([caso_baixo])
        prob_baixo = self.modelo_final.predict_proba(caso_baixo_scaled)[0][1]

        # Caso 2: Tudo desfavorável (alto risco) → deve dar prob ALTA classe 1
        caso_alto = np.ones(len(self.feature_names))
        caso_alto_scaled = self.preprocessors["scaler"].transform([caso_alto])
        prob_alto = self.modelo_final.predict_proba(caso_alto_scaled)[0][1]

        print(f"Caso extremo BAIXO risco  → Prob(pensou evasão) = {prob_baixo:.1%}")
        print(f"Caso extremo ALTO risco   → Prob(pensou evasão) = {prob_alto:.1%}")

        if prob_baixo < prob_alto:
            print("✅ LÓGICA CORRETA: Baixo risco < Alto risco")
        else:
            print("❌ LÓGICA INVERTIDA: Modelo pode estar com labels invertidos ainda!")

        # Feature importance
        if hasattr(self.modelo_final, "coef_"):
            print(f"\n📊 TOP 5 FEATURES MAIS IMPORTANTES:")
            importances = pd.DataFrame(
                {"feature": self.feature_names, "coef": self.modelo_final.coef_[0]}
            )
            importances["abs_coef"] = abs(importances["coef"])
            top5 = importances.nlargest(5, "abs_coef")

            for _, row in top5.iterrows():
                sinal = "⬆️" if row["coef"] > 0 else "⬇️"
                print(f"  {sinal} {row['feature']}: {row['coef']:.4f}")

        return True

    def salvar_modelo(self, nome="modelo_evasao_CORRIGIDO"):
        """7. Salva o modelo final"""
        print("\n7. SALVANDO MODELO")
        print("-" * 70)

        # Pacote completo
        modelo_completo = {
            "modelo": self.modelo_final,
            "preprocessors": self.preprocessors,
            "feature_names": self.feature_names,
            "metricas": self.metricas,
            "metadata": {
                "data_treinamento": pd.Timestamp.now(),
                "n_features": len(self.feature_names),
                "label_encoding": {
                    "0": "NÃO pensou em evasão (baixo risco)",
                    "1": "SIM pensou em evasão (alto risco)",
                },
                "classe_positiva": 1,
                "classe_negativa": 0,
            },
        }

        # Salvar
        arquivo = f"{nome}.pkl"
        with open(arquivo, "wb") as f:
            pickle.dump(modelo_completo, f)

        print(f"✅ Modelo salvo: {arquivo}")
        print(f"\n📋 RESUMO FINAL:")
        print(f"  Modelo: {self.metricas['nome']}")
        print(f"  AUC (CV): {self.metricas['cv_auc_mean']:.4f}")
        print(f"  AUC (Test): {self.metricas['test_auc']:.4f}")
        print(f"  Features: {len(self.feature_names)}")
        print(f"  Classe 0: NÃO pensou em evasão")
        print(f"  Classe 1: SIM pensou em evasão")

        return arquivo

    def executar_pipeline(self):
        """Executa todo o pipeline"""
        try:
            if not self.carregar_dados():
                return False

            if not self.analisar_e_codificar_target():
                return False

            if not self.limpeza_e_features():
                return False

            if not self.preprocessing():
                return False

            if not self.treinar_modelos():
                return False

            if not self.validar_sanidade():
                return False

            self.salvar_modelo()

            print("\n" + "=" * 70)
            print("✅ RETREINAMENTO CONCLUÍDO COM SUCESSO!")
            print("=" * 70)
            return True

        except Exception as e:
            print(f"\n❌ ERRO: {e}")
            import traceback

            traceback.print_exc()
            return False


def main():
    """Função principal"""
    modelo = ModeloEvasaoCorrigido()
    sucesso = modelo.executar_pipeline()

    if sucesso:
        print("\n🎉 Modelo retreinado e salvo com sucesso!")
        print(
            "📌 Use 'modelo_evasao_CORRIGIDO.pkl' no seu código Streamlit atualizado."
        )
    else:
        print("\n❌ Falha no retreinamento.")


if __name__ == "__main__":
    main()
