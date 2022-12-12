import streamlit as st
import joblib
from PIL import Image
import pandas as pd

st.write("""
  # Churn Prediction

  - **Made by Eduardo Chaves - [GitHub](https://github.com/eduardochaves1)**

  Esta página calcula a porcentagem de um cliente X *(cadastrado pelo usuário na barra lateral)* de evadir de uma empresa de telecomunicações *(churn)*.
""")

st.image(Image.open('./assets/churnrate.jpg'))

# ====================
# SIDEBAR (user input_features)
# ====================

st.sidebar.header('**Cadastro dos Parâmetros**')

def input_features():
  gender = st.sidebar.radio('**- Sexo:**', ('Masculino', 'Feminino'), horizontal=True)
  seniorCitizen = st.sidebar.radio('**Idoso:**', ('Sim', 'Não'), horizontal=True)
  partner = st.sidebar.radio('**Casado/Namorando:**', ('Sim', 'Não'), horizontal=True)
  dependents = st.sidebar.radio('**Dependentes:**', ('Sim', 'Não'), horizontal=True)
  phoneService = st.sidebar.radio('**Serviço de Telefone:**', ('Sim', 'Não'), horizontal=True)
  multipleLines = st.sidebar.radio('**Multipla Linhas:**', ('Sim', 'Não', 'Sem Serviço de Telefone'))
  internetService = st.sidebar.radio('**Serviço de Internet:**', ('Fibra Optica', 'DSL', 'Não'))
  onlineSecurity = st.sidebar.radio('**Segurança Online:**', ('Sim', 'Não', 'Sem Serviço de Internet'))
  onlineBackup = st.sidebar.radio('**Backup Online:**', ('Sim', 'Não', 'Sem Serviço de Internet'))
  deviceProtection = st.sidebar.radio('**Proteção de Dispositivo:**', ('Sim', 'Não', 'Sem Serviço de Internet'))
  techSupport = st.sidebar.radio('**Suporte Tecnológico:**', ('Sim', 'Não', 'Sem Serviço de Internet'))
  streamingTV = st.sidebar.radio('**Streaming de TV:**', ('Sim', 'Não', 'Sem Serviço de Internet'))
  stramingMovies = st.sidebar.radio('**Streaming de Filmes:**', ('Sim', 'Não', 'Sem Serviço de Internet'))
  contract = st.sidebar.radio('**Tipo de Contrato:**', ('Mensal', 'Anual', 'Bianual'))
  paperlessBilling = st.sidebar.radio('**Cobrança Digital:**', ('Sim', 'Não'), horizontal=True)
  paymentMethod = st.sidebar.radio('**Método de Pagamento:**', ('Cartão de Crédito', 'Transferência Bancária', 'Cheque Eletrônico', 'Cheque Físico'))
  tenure = st.sidebar.slider('**Tempo de Contratação (Meses):**', 0, 120)
  monthlyCharges = st.sidebar.number_input('**Cobrança Mensal (R$):**')
  totalCharges = monthlyCharges * tenure

  data = {
    'gender': gender,
    'seniorCitizen': seniorCitizen,
    'partner': partner,
    'dependents': dependents,
    'phoneService': phoneService,
    'multipleLines': multipleLines,
    'internetService': internetService,
    'onlineSecurity': onlineSecurity,
    'onlineBackup': onlineBackup,
    'deviceProtection': deviceProtection,
    'techSupport': techSupport,
    'streamingTV': streamingTV,
    'stramingMovies': stramingMovies,
    'contract': contract,
    'paperlessBilling': paperlessBilling,
    'paymentMethod': paymentMethod,
    'monthlyCharges': monthlyCharges,
    'totalCharges': totalCharges,
    'tenure': tenure,
  }
  features = pd.DataFrame(data, index=[0])

  return features

# ====================
# MODEL PREDICTION (with user input_features)
# ====================

model = joblib.load('churn-prediction-model')

# ====================
# MAIN CONTENT
# ====================

df = input_features()

st.subheader('Cadastro dos Parâmetros')
st.write(df)

st.subheader('Resultado da Predição')
predictionMsg = 'continuará na' if predictionResult else 'sairá da'
st.write("""
  O modelo previu uma porcentagem de **{predictionPercentage:.2f}%** de que o cliente **{preditionMsg}** empresa!
""")
