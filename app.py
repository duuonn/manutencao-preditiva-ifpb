import streamlit as st
import joblib
import pandas as pd

# Configurações iniciais da página
st.set_page_config(
    page_title="Manutenção Preditiva Industrial", 
    page_icon="⚙️", 
    layout="centered"
)

# Carrega o modelo treinado apenas uma vez
@st.cache_resource
def carregar_modelo():
    return joblib.load("artefatos/modelo_rf.joblib")

try:
    modelo = carregar_modelo()
except Exception as e:
    st.error(f"Erro ao carregar o modelo: {e}")
    st.stop()

# Menu lateral com informações do projeto
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3095/3095221.png", width=80)
    st.title("Sobre o Projeto")
    st.markdown("""
    **Instituição:** Instituto Federal da Paraíba (IFPB)  
    
    **Desenvolvido por:** Eduardo R. Teixeira  
    
    **Orientação:** Prof. José Thiago Holanda  
    """)
    st.divider()
    st.info("Este sistema utiliza Inteligência Artificial (Random Forest) para prever quebras em maquinários antes que elas aconteçam.")
    st.divider()

    # Link para o repositório do projeto
    st.markdown("**Código Fonte & Metodologia:**")
    st.link_button("📂 Ver Código no GitHub", "https://github.com/duuonn/manutencao-preditiva-ifpb", use_container_width=True)

# Cabeçalho principal
st.subheader("🔧 PLATAFORMA DE MANUTENÇÃO PREDITIVA AI")
st.title("Monitoramento Operacional de Ativos")
st.caption("Insira os dados de telemetria dos sensores abaixo para análise de integridade em tempo real.")

# Indicadores de desempenho do modelo
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

# Entrada dos dados dos sensores
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

# Área onde o resultado será exibido
resultado_placeholder = st.container()

# Executa a análise quando o usuário clicar no botão
if st.button("🚀 Executar Análise de Risco", use_container_width=True, type="primary"):

    # Monta os dados no formato esperado pelo modelo
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
        # Calcula a probabilidade de falha
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
