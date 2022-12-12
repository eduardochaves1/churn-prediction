import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import numpy as np

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
  seniorCitizen = st.sidebar.radio('**Idoso:**', ('Não', 'Sim'), horizontal=True)
  partner = st.sidebar.radio('**Casado/Namorando:**', ('Não', 'Sim'), horizontal=True)
  dependents = st.sidebar.radio('**Dependentes:**', ('Não', 'Sim'), horizontal=True)
  phoneService = st.sidebar.radio('**Serviço de Telefone:**', ('Não', 'Sim'), horizontal=True)
  multipleLines = st.sidebar.radio('**Multipla Linhas:**', ('Não', 'Sim'), horizontal=True)
  internetService = st.sidebar.radio('**Serviço de Internet:**', ('Não', 'Fibra Optica', 'DSL'))
  onlineSecurity = st.sidebar.radio('**Segurança Online:**', ('Não', 'Sim'), horizontal=True)
  onlineBackup = st.sidebar.radio('**Backup Online:**', ('Não', 'Sim'), horizontal=True)
  deviceProtection = st.sidebar.radio('**Proteção de Dispositivo:**', ('Não', 'Sim'), horizontal=True)
  techSupport = st.sidebar.radio('**Suporte Tecnológico:**', ('Não', 'Sim'), horizontal=True)
  streamingTV = st.sidebar.radio('**Streaming de TV:**', ('Não', 'Sim'), horizontal=True)
  stramingMovies = st.sidebar.radio('**Streaming de Filmes:**', ('Não', 'Sim'), horizontal=True)
  contract = st.sidebar.radio('**Tipo de Contrato:**', ('Mensal', 'Anual', 'Bianual'))
  paperlessBilling = st.sidebar.radio('**Cobrança Digital:**', ('Não', 'Sim'), horizontal=True)
  paymentMethod = st.sidebar.radio('**Método de Pagamento:**', ('Cartão de Crédito', 'Transferência Bancária', 'Cheque Eletrônico', 'Cheque Físico'))
  tenure = st.sidebar.slider('**Tempo de Contratação (Meses):**', 0, 120, 12)
  monthlyCharges = st.sidebar.number_input('**Cobrança Mensal (R$):**', value=65)
  totalCharges = monthlyCharges * tenure

  data = {
    'gender': gender,
    'SeniorCitizen': seniorCitizen,
    'Partner': partner,
    'Dependents': dependents,
    'tenure': tenure,
    'PhoneService': phoneService,
    'MultipleLines': multipleLines,
    'InternetService': internetService,
    'OnlineSecurity': onlineSecurity,
    'OnlineBackup': onlineBackup,
    'DeviceProtection': deviceProtection,
    'TechSupport': techSupport,
    'StreamingTV': streamingTV,
    'StramingMovies': stramingMovies,
    'Contract': contract,
    'PaperlessBilling': paperlessBilling,
    'PaymentMethod': paymentMethod,
    'MonthlyCharges': monthlyCharges,
    'TotalCharges': totalCharges,
  }
  features = pd.DataFrame(data, index=[0])

  return features

df = input_features()

# ====================
# DF PREPROCESSING (with user input_features)
# ====================

categorical_features = df.drop(['TotalCharges', 'MonthlyCharges', 'tenure'], axis=1)
for feature in categorical_features.columns:
  df[feature] = df[feature].astype('category')

# IMPORTANT: As trocas por 0,1,2... seguem ordem alfabétida baseada nos textos do DF original
# treinado para o modelo salvo e não nos dados aqui escritos em português
df.replace(('Não', 'Sim'), (0, 1), inplace=True)
df.gender.replace(('Feminino', 'Masculino'), (0, 1), inplace=True)
df.InternetService.replace(('DSL', 'Não', 'Fibra Optica'), (0, 1, 2), inplace=True)
df.Contract.replace(('Mensal', 'Anual', 'Bianual'), (0, 1, 2), inplace=True)
df.PaymentMethod.replace(('Transferência Bancária', 'Cartão de Crédito', 'Cheque Eletrônico', 'Cheque Físico'), (0, 1, 2, 3), inplace=True)

# ====================
# MODEL PREDICTION (with user input_features)
# ====================

model = joblib.load('churn-prediction-model')
predictionPercentage = np.round((model.predict_proba(df) * 100), 2)

# ====================
# MAIN CONTENT
# ====================

st.subheader('Cadastro dos Parâmetros')
st.write('Os parâmetros estão todos em tipos númericos pois foram convertidos do texto cadastrado pelo usuário com o intuito de ter uma melhor predição pelo modelo previamente treinado nesta forma.')
st.write(df)

st.subheader('Resultado da Predição')

predictions = [predictionPercentage[0][0], predictionPercentage[0][1]]

predictionMsg = 'continuará na' if predictions[0] > 50 else 'sairá da'
predictionNumber = predictions[0] if predictions[0] > 50 else predictions[1]

st.write('O modelo previu uma porcentagem de', predictionNumber, '% de que o cliente', predictionMsg, 'empresa!')
st.write(predictionPercentage)
