import streamlit as st
import joblib
import pandas as pd

# 1. Configuração da página
st.set_page_config(
    page_title="Manutenção Preditiva Industrial", 
    page_icon="⚙️", 
    layout="centered"
)

# Carregar o modelo treinado de forma segura
@st.cache_resource
def carregar_modelo():
    return joblib.load("artefatos/modelo_rf.joblib")

try:
    modelo = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# 2. Barra Lateral (Sidebar) - Organizada e com link do GitHub
with st.sidebar:
    st.title("🏭 Inteligência Industrial")
    st.markdown("""
    Este sistema utiliza **Machine Learning** para prever falhas em equipamentos industriais, auxiliando equipes de engenharia na tomada de decisão e redução de custos com paradas não planejadas.
    """)
    st.divider()
    
    # Botão chamativo para o seu GitHub (Substitua o LINK abaixo pelo seu)
    st.markdown("**Código Fonte & Metodologia:**")
    st.link_button("📂 Ver Código no GitHub", "https://github.com/duuonn/manutencao-preditiva-ifpb", use_container_width=True)
    
    st.divider()
    st.markdown("""
    **Instituição:** Instituto Federal da Paraíba (IFPB)  
    **Desenvolvedor:** Eduardo R. Teixeira  
    """)

# 3. Área Principal - Banner e Título Profissional
st.subheader("🔧 PLATAFORMA DE MANUTENÇÃO PREDITIVA AI")
st.title("Monitoramento Operacional de Ativos")
st.caption("Insira os dados de telemetria dos sensores abaixo para análise de integridade em tempo real.")

# 4. CARD COM AS MÉTRICAS DO MODELO (O que o recrutador quer ver!)
st.write("")
st.markdown("##### 📊 Performance do Modelo Homologado (Random Forest)")
with st.container(border=True):
    m1, m2, m3 = st.columns(3)
    with m1:
        st.metric(label="Recall (Captação de Falhas)", value="94.8%")
    with m2:
        st.metric(label="F1-Score Geral", value="79.0%")
    with m3:
        st.metric(label="Threshold Otimizado", value="0.40")

st.divider()

# 5. Formulário de Sensores
st.markdown("##### 🔌 Telemetria Atual dos Sensores")
with st.container(border=True):
    col1, col2 = st.columns(2)
    
    with col1:
        st.markdown("**Variáveis Térmicas e Mecânicas**")
        temperatura = st.slider("Temperatura do Motor (°C)", min_value=20.0, max_value=120.0, value=65.0, step=0.1)
        vibracao = st.slider("Nível de Vibração (RMS)", min_value=0.0, max_value=10.0, value=2.5, step=0.1)
    
    with col2:
        st.markdown("**Variáveis Elétricas e Tempo**")
        corrente = st.slider("Corrente Elétrica (A)", min_value=0.0, max_value=50.0, value=18.0, step=0.1)
        horas_uso = st.number_input("Horas de Uso Acumuladas", min_value=0, max_value=20000, value=150, step=50)

st.write("") 

# Container de Resultados
resultado_placeholder = st.container()

# Botão de Execução
if st.button("🚀 Executar Análise de Risco", use_container_width=True, type="primary"):
    
    # DataFrame estruturado para o pipeline
    dados_entrada = pd.DataFrame([{
        'vibracao_rms': vibracao,
        'temperatura_c': temperatura,
        'pressao_bar': 5.5,
        'horas_uso': horas_uso,
        'corrente_a': corrente,
        'tensao_v': 220.0,
        'tipo_maquina': 'compressor',
        'turno': 'diurno',
        'manut_prev': 'nao'
    }])
    
    try:
        # Predição
        probabilidade = modelo.predict_proba(dados_entrada)[0][1]
        threshold_projeto = 0.40
        porcentagem_risco = probabilidade * 100
        
        with resultado_placeholder:
            st.write("")
            st.subheader("🎯 Diagnóstico do Sistema")
            
            c1, c2 = st.columns([1, 2])
            
            with c1:
                if probabilidade >= threshold_projeto:
                    st.metric(label="Risco de Falha", value=f"{porcentagem_risco:.1f}%", delta="CRÍTICO", delta_color="inverse")
                else:
                    st.metric(label="Risco de Falha", value=f"{porcentagem_risco:.1f}%", delta="ESTÁVEL", delta_color="normal")
            
            with c2:
                if probabilidade >= threshold_projeto:
                    st.error("🚨 **ALERTA: INTERRUPÇÃO RECOMENDADA**\n\nO risco calculado ultrapassou o limite de segurança de 40%. Acione a equipe de manutenção preventiva imediatamente para evitar quebras severas.")
                else:
                    st.success("✅ **CONDIÇÃO OPERACIONAL NORMAL**\n\nO ativo apresenta parâmetros de funcionamento estáveis. Nenhuma ação corretiva ou preventiva é necessária no momento.")
                    
    except Exception as e:
        with resultado_placeholder:
            st.error(f"Erro na previsão: {e}")
