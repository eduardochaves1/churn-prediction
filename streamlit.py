import streamlit as st
import joblib
from PIL import Image
import pandas as pd
import numpy as np

st.write("""
  # Churn Prediction

  - **Made by [Eduardo Chaves](https://linkedin.com/in/edu-chaves 'LinkedIn de Eduardo Chaves')**
  - **Repositório deste Projeto no [GitHub](https://github.com/eduardochaves1/churn-prediction 'Churn Prediction by Eduardo Chaves')**

  ## Qual o problema a ser resolvido?
  O intuito deste projeto é calcular a porcentagem de um cliente X *(cadastrado pelo usuário na barra lateral)* de cancelar ou continuar em um contrato (*plano*) de serviçoes de uma empresa de telecomunicações no próximo mês.

  Tal situação é medido pelo *Churn*, que é quando um cliente deixa de consumir o produto / serviço de uma empresa. E este projeto visa capacitar uma empresa a identificar possíveis clientes que estão mais propensos a tomar tal decisão.
  
  Assim como identificar as métricas – relacionadas aos serviços / produtos da empresa – que mais influenciam na escolha do cliente sair ou continuar na mesma. Visando tomar medidas a diminuir o índice de *Churn* na empresa nos meses seguintes.
""")

st.image(Image.open('./assets/churnrate.jpg'))

# ====================
# SIDEBAR (user input_features)
# ====================

st.sidebar.write('# **Cadastro dos Parâmetros**')

def input_features():
  gender = st.sidebar.radio('**Sexo:**', ('Masculino', 'Feminino'), horizontal=True)
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
  tenure = st.sidebar.slider('**Tempo de Contratação (Meses):**', 0, 70, 12)
  monthlyCharges = st.sidebar.number_input('**Cobrança Mensal (R$):**', value=65, min_value=10, max_value=120)
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
# MAIN CONTENT (first part)
# ====================

st.write("""
  ## Como usar esta ferramenta?
  Basta você cadastrar as variáveis na barra ao lado esquerdo e automaticamente ao manipular esses dados o modelo preditivo entrará em ação para trazer os resultados logo abaixo!

  ## Parâmetros Cadastrados
  - Abaixo segue os valores cadastrados pelo usuário na barra lateral:
""")
st.write(df)

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

model = joblib.load('./assets/churn-prediction-model')
predictionPercentage = np.round((model.predict_proba(df) * 100), 2)

# ====================
# MAIN CONTENT (last part)
# ====================

st.write('## Resultado da Predição')

predictions = [predictionPercentage[0][0], predictionPercentage[0][1]]

predictionMsg = '***continuará*** no' if predictions[0] > 50 else '***cancelará*** o'
predictionNumber = predictions[0] if predictions[0] > 50 else predictions[1]

st.write('O modelo previu uma porcentagem de', predictionNumber, '% de que o cliente', predictionMsg, 'contrato!')
st.write(predictionPercentage)
st.write('- 0 = Not Churn | 1 = Churn')

st.write("""
  ## Como o Modelo Chegou a este Resultado?
  Segue abaixo um gráfico exibindo as variáveis que mais importam na tomada de decisão de acordo com o aprendizado do modelo treinado previamente.
""")
st.image(Image.open('./assets/graphs/feature-importance.png'))

# ====================
# GRAPHS EXPOSURE
# ====================

st.write("""
  ---
  ## Como este Modelo foi Treinado?
  O modelo preditivo usado neste projeto foi treinado por um algorítimo de *Machine Learning* em cima de um *Data Frame*, o qual possuia as seguintes estatísticas...
""")
st.caption('**OBS.:** Para entender o lado mais técnico acesse o link para o repositório deste projeto no GitHub no ínicio desta página.')

st.write('- ### Pessoas que deram *Churn* nos 30 dias anteriores')
st.image(Image.open('./assets/graphs/churn.png'))

st.write("""
  - ### Tempo de Contrato, Gasto Mensal e Gasto Total em Serviços
  
  Os gráficos *boxplot* abaixo ([saiba como interpretar esse tipo de gráfico com esse tutotiral](https://www.youtube.com/watch?v=qU2lANG4hYQ&ab_channel=Statplace 'Vídeo no Youtube ensinando a interpretar um gráfico Boxplot')) demonstram respectivamente:
  
  - A quantidade de tempo, em meses, que os clientes manteram os contratos de serviços da empresa (*Tenure*);
  - O gasto mensal que os clientes mantinham nos serviços da empresa (*MonthlyCharges*);
  - O gasto total que os clientes tiveram nos serviços da empresa até o último mês (*TotalCharges*).
""")
st.image(Image.open('./assets/graphs/numeric-features.png'))

st.write("""
  - ### Informações sobre Contrato
  
  Os gráficos abaixo demonstram a quantidade de pessoas para as seguintes informações, respectivamente:
  
  - O tipo de período de contrato (Mensal, Anual ou Bianual);
  - Se o pagamento era realizado de forma digital;
  - O Método de Pagamento utilizado (Cheque Eletrônico, Cheque por Correios, Transferência Bancária e Cartão de Crédito, sendo esses dois últimos feitos de forma automática).
""")
st.image(Image.open('./assets/graphs/contract-features.png'))

st.write("""
  - ### Dados Demográficos
  
  Os gráficos demográficos abaixo (sobre os clientes da empresa) demonstram a quantidade de pessoas para as seguintes informações, respectivamente:
  
  - Sexo do cliente;
  - Se o cliente é idoso;
  - Se o cliente tem um parceiro (Casado ou Namorando);
  - Se o cliente tem dependentes (alguém que depende financeiramente do cliente).
""")
st.image(Image.open('./assets/graphs/demographic-features.png'))

st.write("""
  - ### Dados de Serviços
  
  Os gráficos abaixo demonstram a quantidade de pessoas que utilizam os seguintes serviços fornecidos pela empresa, respectivamente:
  
  - Serviço Telefônico;
  - Multiplas Linhas Telefônicas;
  - Serviço de Internet;
  - Serviço de Seguraça Online;
  - Serviço de Backup Online;
  - Proteção de Dispositivo;
  - Suporte Técnico;
  - Straming de TV;
  - Straming de Filmes.
""")
st.image(Image.open('./assets/graphs/services-features.png'))

st.write("""
  ---
  > ***"No futuro, o pensamento estatístico será tão necessário para a cidadania eficiente como saber ler e escrever."*** - H.G. Wells (escritor, autor de "A Guerra dos Mundos" e "A Máquina do Tempo")
""")
