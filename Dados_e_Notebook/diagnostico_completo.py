import pickle
import numpy as np
import pandas as pd

print("=" * 80)
print("DIAGNÓSTICO COMPLETO DO MODELO")
print("=" * 80)

# Carregar modelo
with open("modelo_evasao_CORRIGIDO.pkl", "rb") as f:
    modelo_completo = pickle.load(f)

modelo = modelo_completo["modelo"]
scaler = modelo_completo["preprocessors"]["scaler"]
imputer = modelo_completo["preprocessors"]["imputer"]
feature_names = modelo_completo["feature_names"]

print(f"\n📋 Features do modelo ({len(feature_names)}):")
for i, name in enumerate(feature_names):
    print(f"  {i:2d}. {name}")

print("\n" + "=" * 80)
print("TESTE 1: Caso 'Estudante Feliz' (deveria dar BAIXO risco)")
print("=" * 80)

# Simular exatamente o que o Streamlit faz para "estudante feliz"
dados_feliz = {
    "Como é o seu deslocamento até a universidade?": 1,  # Carro
    "Com relação ao transporte do item anterior, ele é:": 3,  # Próprio
    "O transporte representa uma barreira/dificuldade para frequentar a universidade?": 1,  # Não, nunca foi
    "Você mora em Angicos?": 2,  # Sim
    "Você se identifica com o curso que está fazendo?": 2,  # Sim
    "Como você considera a acessibilidade do Campus?": 0,  # Adequada
    "Em relação ao tempo necessário como discente para dedicar no estudo?": 3,  # É suficiente
    "Você trabalha?": 1,  # Não trabalho
    "Se você trabalha, em quais horários?": 2,  # Não se aplica
    "É casado(a)/está em união estável?": 0,  # Não
    "Tem filhos?": 0,  # Não
    "Você contribui para o sustento financeiro da família?": 0,  # Não contribuo
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Aparência": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Condição financeira": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Cor de pele": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Curso": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Deficiência": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Dificuldade de aprendizado": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Gênero": 0,
    # "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Não": 1,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_idade": 0,
}

# Criar DataFrame
df_feliz = pd.DataFrame([dados_feliz])
df_feliz = df_feliz.reindex(columns=feature_names, fill_value=0)

print("\n📊 Valores das features (antes do preprocessing):")
for col in feature_names:
    print(f"  {col}: {df_feliz[col].values[0]}")

# Aplicar imputer
df_feliz_imputed = pd.DataFrame(imputer.transform(df_feliz), columns=feature_names)

print("\n📊 Após imputer (deve ser igual):")
print(f"  Min: {df_feliz_imputed.values.min()}, Max: {df_feliz_imputed.values.max()}")

# Aplicar scaler
df_feliz_scaled = scaler.transform(df_feliz_imputed)

print("\n📊 Após scaler (normalizado):")
print(f"  Min: {df_feliz_scaled.min():.3f}, Max: {df_feliz_scaled.max():.3f}")
print(f"  Shape: {df_feliz_scaled.shape}")

# Predição
prob_feliz = modelo.predict_proba(df_feliz_scaled)[0]
print(f"\n🎯 PREDIÇÃO:")
print(f"  Classe 0: {prob_feliz[0]:.3f} ({prob_feliz[0]*100:.1f}%)")
print(f"  Classe 1: {prob_feliz[1]:.3f} ({prob_feliz[1]*100:.1f}%)")

if prob_feliz[1] > 0.5:  # <-- CORRIGIDO: Agora olhando para a Classe 1 (Risco)
    print(f"  ⚠️ Previsto: ALTO RISCO (classe 1 > 50%)")
else:
    print(f"  ✅ Previsto: BAIXO RISCO (classe 1 < 50%)")

print("\n" + "=" * 80)
print("TESTE 2: Caso 'Estudante em Risco' (deveria dar ALTO risco)")
print("=" * 80)

# Simular estudante com muitos fatores de risco
dados_risco = {
    "Como é o seu deslocamento até a universidade?": 4,  # Ônibus
    "Com relação ao transporte do item anterior, ele é:": 4,  # Público
    "O transporte representa uma barreira/dificuldade para frequentar a universidade?": 2,  # Sim, sempre
    "Você mora em Angicos?": 1,  # Não
    "Você se identifica com o curso que está fazendo?": 1,  # Não, não sei se concluirei
    "Como você considera a acessibilidade do Campus?": 1,  # Inadequada
    "Em relação ao tempo necessário como discente para dedicar no estudo?": 0,  # Insuficiente, não consigo
    "Você trabalha?": 2,  # Emprego formal
    "Se você trabalha, em quais horários?": 4,  # Tempo integral
    "É casado(a)/está em união estável?": 1,  # Sim
    "Tem filhos?": 1,  # Sim
    "Você contribui para o sustento financeiro da família?": 3,  # Único com renda
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Aparência": 1,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Condição financeira": 1,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Cor de pele": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Curso": 1,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Deficiência": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Dificuldade de aprendizado": 1,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Gênero": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_Não": 0,
    "Você sofreu algum tipo de preconceito ou violência durante o curso relativo a:_idade": 0,
}

df_risco = pd.DataFrame([dados_risco])
df_risco = df_risco.reindex(columns=feature_names, fill_value=0)

df_risco_imputed = pd.DataFrame(imputer.transform(df_risco), columns=feature_names)
df_risco_scaled = scaler.transform(df_risco_imputed)

prob_risco = modelo.predict_proba(df_risco_scaled)[0]
print(f"\n🎯 PREDIÇÃO:")
print(f"  Classe 0: {prob_risco[0]:.3f} ({prob_risco[0]*100:.1f}%)")
print(f"  Classe 1: {prob_risco[1]:.3f} ({prob_risco[1]*100:.1f}%)")

if prob_risco[1] > 0.5:  # <-- CORRIGIDO: Agora olhando para a Classe 1 (Risco)
    print(f"  ⚠️ Previsto: ALTO RISCO (classe 1 > 50%)")
else:
    print(f"  ✅ Previsto: BAIXO RISCO (classe 1 < 50%)")

print("\n" + "=" * 80)
print("ANÁLISE FINAL")
print("=" * 80)

print(f"\nCaso FELIZ:")
print(f"  Classe 0: {prob_feliz[0]*100:.1f}%")
print(f"  Classe 1: {prob_feliz[1]*100:.1f}%")

print(f"\nCaso RISCO:")
print(f"  Classe 0: {prob_risco[0]*100:.1f}%")
print(f"  Classe 1: {prob_risco[1]*100:.1f}%")

if prob_feliz[0] < prob_risco[0]:
    print("\n✅ CONCLUSÃO: Classe 0 = ALTO RISCO")
    print("   Use no Streamlit: probabilidade_evasao = prob_classe_0")
else:
    print("\n✅ CONCLUSÃO: Classe 1 = ALTO RISCO")
    print("   Use no Streamlit: probabilidade_evasao = prob_classe_1")

print("\n" + "=" * 80)
