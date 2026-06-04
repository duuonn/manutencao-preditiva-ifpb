import streamlit as st
import joblib
import pandas as pd

# Configuração da página
st.set_page_config(page_title="Manutenção Preditiva - IFPB", page_icon="⚙️", layout="centered")

# Carregar o modelo treinado de forma segura
@st.cache_resource
def carregar_modelo():
    return joblib.load("artefatos/modelo_rf.joblib")

try:
    modelo = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo. Certifique-se de que a pasta 'artefatos' contém o arquivo 'modelo_rf.joblib'. Erro: {e}")
    st.stop()

# Cabeçalho
st.title("⚙️ Painel de Manutenção Preditiva")
st.markdown("### **Instituto Federal da Paraíba (IFPB)**")
st.write("Insira os dados atuais dos sensores para prever o risco de falha em tempo real.")

st.divider()

# Formulário de entrada de dados
st.subheader("📊 Dados dos Sensores")

col1, col2 = st.columns(2)

with col1:
    temperatura = st.slider("Temperatura (°C)", min_value=20.0, max_value=120.0, value=65.0, step=0.1)
    vibracao = st.slider("Vibração (RMS)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)

with col2:
    corrente = st.slider("Corrente Elétrica (A)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
    horas_uso = st.number_input("Horas de Uso Acumuladas", min_value=0, max_value=20000, value=150)

st.divider()

# Criamos um container fixo para o resultado. 
# Isso evita o erro de renderização dinâmica ("removeChild") no navegador.
resultado_placeholder = st.container()

if st.button("🚀 Analisar Condição do Maquinário", use_container_width=True):
    
    # Criar DataFrame com as colunas idênticas ao treino do notebook
    dados_entrada = pd.DataFrame([{
        'vibracao_rms': vibracao,
        'temperatura_c': temperatura,
        'pressao_bar': 5.5,     # Valor médio preenchido já que o slider não foi colocado
        'horas_uso': horas_uso,
        'corrente_a': corrente,
        'tensao_v': 220.0,      # Valor médio padrão
        'tipo_maquina': 'compressor', # Valor categórico padrão
        'turno': 'diurno',            # Valor categórico padrão
        'manut_prev': 'nao'           # Valor categórico padrão
    }])
    
    try:
        # Calcular probabilidade de falha
        probabilidade = modelo.predict_proba(dados_entrada)[0][1]
        threshold_projeto = 0.40
        
        # Exibir o resultado dentro do container fixo
        with resultado_placeholder:
            st.subheader("🎯 Resultado da Análise:")
            if probabilidade >= threshold_projeto:
                st.error(f"🚨 **ALERTA DE FALHA IMINENTE!**\n\nRisco calculado: **{probabilidade*100:.1f}%**.\n\nRecomenda-se interromper a operação imediatamente.")
            else:
                st.success(f"✅ **MAQUINÁRIO OPERANDO EM SEGURANÇA**\n\nRisco calculated: **{probabilidade*100:.1f}%**.\n\nNenhuma anomalia crítica detectada.")
    except Exception as e:
        with resultado_placeholder:
            st.error(f"Erro ao realizar a previsão. Verifique as colunas do modelo. Erro: {e}")
