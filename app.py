import streamlit as st
import joblib
import pandas as pd

# 1. Configuração da página (Tema Escuro e Amplo)
st.set_page_config(
    page_title="Predição de Falhas Industrial", 
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

# 2. Barra Lateral (Sidebar) - Organização de créditos e informações
with st.sidebar:
    st.image("https://cdn-icons-png.flaticon.com/512/3095/3095221.png", width=80) # Ícone de fábrica
    st.title("Sobre o Projeto")
    st.markdown("""
    **Instituição:** Instituto Federal da Paraíba (IFPB)  
    
    **Desenvolvido por:** Eduardo R. Teixeira  
    
    **Orientação:** Prof. José Thiago Holanda  
    """)
    st.divider()
    st.info("Este sistema utiliza Inteligência Artificial (Random Forest) para prever quebras em maquinários antes que elas aconteçam.")

# 3. Área Principal
st.title("⚙️ Monitoramento & Manutenção Preditiva")
st.caption("Painel de Controle e Análise de Risco de Ativos Industriais")
st.divider()

# Formulário organizado em caixas (Fieldset)
st.subheader("🔌 Telemetria dos Sensores")

# Criando os Sliders dentro de uma estrutura organizada
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

st.write("") # Espaçamento

# Container para exibir os resultados de forma limpa
resultado_placeholder = st.container()

# Botão de ação destacado
if st.button("🚀 Executar Análise de Risco", use_container_width=True, type="primary"):
    
    # Criar DataFrame com as colunas do modelo
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
        # Calcular probabilidade
        probabilidade = modelo.predict_proba(dados_entrada)[0][1]
        threshold_projeto = 0.40
        porcentagem_risco = probabilidade * 100
        
        with resultado_placeholder:
            st.write("")
            st.subheader("🎯 Diagnóstico do Sistema")
            
            # Layout de resposta com métrica grande
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
