import streamlit as st
import joblib
import pandas as pd

# Configuração da página para ficar bonita e com título correto
st.set_page_config(page_title="Manutenção Preditiva - IFPB", page_icon="⚙️", layout="centered")

# 1. Função para carregar o modelo treinado (guarda na memória para ser rápido)
@st.cache_resource
def carregar_modelo():
    return joblib.load("artefatos/modelo_rf.joblib")

modelo = carregar_modelo()

# Cabeçalho da página
st.title("⚙️ Painel de Manutenção Preditiva")
st.markdown("### **Instituto Federal da Paraíba (IFPB)**")
st.write("Insira os dados atuais dos sensores do maquinário para prever o risco de falha em tempo real.")

st.divider()

# 2. Criar os controlos (sliders e inputs) com as variáveis exatas do teu projeto
st.subheader("📊 Dados dos Sensores")

col1, col2 = st.columns(2)

with col1:
    temperatura = st.slider("Temperatura (°C)", min_value=20.0, max_value=120.0, value=65.0, step=0.1)
    vibracao = st.slider("Vibração (RMS)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

with col2:
    corrente = st.slider("Corrente Elétrica (A)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    horas_uso = st.number_input("Horas de Uso Acumuladas", min_value=0, max_value=20000, value=150)

st.divider()

# 3. Botão para acionar a previsão
if st.button("🚀 Analisar Condição do Maquinário", use_container_width=True):
    
    # Criar um DataFrame com os nomes exatos das colunas que o teu pipeline do Colab espera
    dados_entrada = pd.DataFrame([{
        'temperatura_c': temperatura,
        'vibracao_rms': vibracao,
        'corrente_a': corrente,
        'horas_uso': horas_uso
    }])
    
    # O teu modelo calcula a probabilidade de falha (classe 1)
    probabilidade = modelo.predict_proba(dados_entrada)[0][1]
    
    st.subheader("🎯 Resultado da Análise:")
    
    # Vamos usar o limiar (threshold) otimizado que o teu projeto calculou
    # (Como o modelo da Random Forest obteve ~0.40 para maximizar o F1, usamos esse valor)
    threshold_projeto = 0.40
    
    if probabilidade >= threshold_projeto:
        st.error(f"🚨 **ALERTA DE FALHA IMINENTE!**\n\nRisco calculado: **{probabilidade*100:.1f}%**.\n\nRecomenda-se interromper a operação e acionar a equipa de manutenção preventiva imediatamente.")
    else:
        st.success(f"✅ **MAQUINÁRIO OPERANDO EM SEGURANÇA**\n\nRisco calculado: **{probabilidade*100:.1f}%**.\n\nNenhuma anomalia crítica detectada nos sensores.")