import streamlit as st
import pandas as pd
import joblib
from PIL import Image


def get_prediction(data):
  for feature, fit in joblib.load('assets/labelEncoder_fit.jbl'):
    if feature != 'Churn':
      data[feature] = fit.transform(data[feature])

  for feature in data.drop(['MonthlyCharges', 'tenure'], axis=1).columns:
    data[feature] = data[feature].astype('category')

  for feature, scaler in joblib.load('assets/minMaxScaler_fit.jbl'):
    data[feature] = scaler.transform(data[feature].values.reshape(-1,1))

  model = joblib.load('./assets/churn-prediction-model.jbl')
  return pd.DataFrame({
    'Not Churn': f'{model.predict_proba(data)[0][0] * 100 :.1f}%',
    'Churn': f'{model.predict_proba(data)[0][1] * 100 :.1f}%'}, index=['Predictions'])


# ====================
# SIDEBAR (user input_features)
# ====================

st.sidebar.write('# **Cadastro dos Parâmetros**')

def input_features():
  def get_sidebar_radio(title, choices=('No', 'Yes'), horizontal=True):
    return st.sidebar.radio(f'**{title}:**', choices, horizontal=horizontal)

  gender = get_sidebar_radio('Sexo', ('Male', 'Female'))
  seniorCitizen = get_sidebar_radio('Idoso')
  partner = get_sidebar_radio('Casado/Namorando')
  dependents = get_sidebar_radio('Dependentes')
  phoneService = get_sidebar_radio('Serviço de Telefone')
  multipleLines = get_sidebar_radio('Multipla Linhas')
  internetService = get_sidebar_radio('Serviço de Internet', ('No', 'Fiber optic', 'DSL'), horizontal=False)
  onlineSecurity = get_sidebar_radio('Segurança Online')
  onlineBackup = get_sidebar_radio('Backup Online')
  deviceProtection = get_sidebar_radio('Proteção de Dispositivo')
  techSupport = get_sidebar_radio('Suporte Tecnológico')
  streamingTV = get_sidebar_radio('Streaming de TV')
  streamingMovies = get_sidebar_radio('Streaming de Filmes')
  contract = get_sidebar_radio('Tipo de Contrato', ('Month-to-month', 'One year', 'Bianual'), horizontal=False)
  paperlessBilling = get_sidebar_radio('Cobrança Digital')
  paymentMethod = get_sidebar_radio('Método de Pagamento', ('Credit card (automatic)', 'Bank transfer (automatic)', 'Electronic check', 'Mailed check'), horizontal=False)
  tenure = st.sidebar.slider('**Tempo de Contratação (Meses):**', min_value=0, max_value=70, value=12)
  monthlyCharges = st.sidebar.number_input('**Cobrança Mensal (R$):**', min_value=10.0, max_value=120.0, value=65.0)

  return pd.DataFrame({
    'gender': gender, 'SeniorCitizen': seniorCitizen, 'Partner': partner, 'Dependents': dependents,
    'tenure': tenure, 'PhoneService': phoneService, 'MultipleLines': multipleLines,
    'InternetService': internetService, 'OnlineSecurity': onlineSecurity, 'OnlineBackup': onlineBackup,
    'DeviceProtection': deviceProtection, 'TechSupport': techSupport, 'StreamingTV': streamingTV,
    'StreamingMovies': streamingMovies, 'Contract': contract, 'PaperlessBilling': paperlessBilling,
    'PaymentMethod': paymentMethod, 'MonthlyCharges': monthlyCharges}, index=['Input'])

data = input_features()


# ====================
# MAIN CONTENT (first part)
# ====================

st.write("""
  # Churn Prediction

  - **Made by [Eduardo Chaves](https://linkedin.com/in/edu-chaves 'LinkedIn de Eduardo Chaves')**
  - **Repositório deste Projeto no [GitHub](https://github.com/eduardochaves1/churn-prediction 'Churn Prediction by Eduardo Chaves')**
""")


st.image(Image.open('assets/banner_img.jpg'))


st.write("""
  ## Qual o problema a ser resolvido?

  O intuito deste projeto é calcular a porcentagem de um cliente X *(cadastrado pelo usuário na barra lateral)* de cancelar ou continuar em um contrato (*plano*) de serviçoes de uma empresa de telecomunicações no próximo mês.

  Tal situação é medido pelo *Churn*, que é quando um cliente deixa de consumir o produto / serviço de uma empresa. E este projeto visa capacitar uma empresa a identificar possíveis clientes que estão mais propensos a tomar tal decisão.
  
  Assim como identificar as métricas – relacionadas aos serviços / produtos da empresa – que mais influenciam na escolha do cliente sair ou continuar na mesma. Visando tomar medidas a diminuir o índice de *Churn* na empresa nos meses seguintes.
""")


st.write("""
  ## Como usar esta ferramenta?

  Basta você cadastrar as variáveis na barra ao lado esquerdo e automaticamente ao manipular esses dados o modelo preditivo entrará em ação para trazer os resultados logo abaixo!
""")


st.write("""
  ## Parâmetros Cadastrados

  - Abaixo segue os valores cadastrados pelo usuário na barra lateral:
""")
st.write(data)


st.write('## Resultado da Predição')

prediction = get_prediction(data)
predictionMsg = '***continuará*** no' if float(prediction['Churn'][0][:-1]) <= 50 else '***cancelará*** o'
predictionPercent = prediction['Not Churn'][0] if float(prediction['Churn'][0][:-1]) <= 50 else prediction['Churn'][0]

st.write(f'O modelo previu uma porcentagem de **{predictionPercent}** de que o cliente {predictionMsg} contrato!')
st.write(prediction)


st.write("""
  ## Como o Modelo Chegou a este Resultado?

  Segue abaixo um gráfico exibindo as variáveis que mais importam na tomada de decisão de acordo com o aprendizado do modelo treinado previamente.
""")
st.image(Image.open('./assets/graphs/feature-importance.png'))


st.write("""
  ---
  ## Como este Modelo foi Treinado?

  O modelo preditivo usado neste projeto foi treinado por um algorítimo de *Machine Learning* em cima de um *Data Frame*, o qual possuia as seguintes estatísticas...
""")
st.caption('**OBS.:** Para entender o lado mais técnico acesse o link para o repositório deste projeto no GitHub no ínicio desta página.')


st.write('### 1 - Pessoas que deram *Churn* nos 30 dias anteriores')
st.image(Image.open('./assets/graphs/churn.png'))


st.write("""
  ### 2 -  Tempo de Contrato, Gasto Mensal e Gasto Total em Serviços
  
  Os gráficos *boxplot* abaixo ([saiba como interpretar esse tipo de gráfico com esse tutotiral](https://www.youtube.com/watch?v=qU2lANG4hYQ&ab_channel=Statplace 'Vídeo no Youtube ensinando a interpretar um gráfico Boxplot')) demonstram respectivamente:
  
  - A quantidade de tempo, em meses, que os clientes manteram os contratos de serviços da empresa (*Tenure*);
  - O gasto mensal que os clientes mantinham nos serviços da empresa (*MonthlyCharges*);
  - O gasto total que os clientes tiveram nos serviços da empresa até o último mês (*TotalCharges*).
""")
st.image(Image.open('./assets/graphs/numeric-features.png'))


st.write("""
  ### 3 - Informações sobre Contrato
  
  Os gráficos abaixo demonstram a quantidade de pessoas para as seguintes informações, respectivamente:
  
  - O tipo de período de contrato (Mensal, Anual ou Bianual);
  - Se o pagamento era realizado de forma digital;
  - O Método de Pagamento utilizado (Cheque Eletrônico, Cheque por Correios, Transferência Bancária e Cartão de Crédito, sendo esses dois últimos feitos de forma automática).
""")
st.image(Image.open('./assets/graphs/contract-features.png'))


st.write("""
  ### 4 - Dados Demográficos
  
  Os gráficos demográficos abaixo (sobre os clientes da empresa) demonstram a quantidade de pessoas para as seguintes informações, respectivamente:
  
  - Sexo do cliente;
  - Se o cliente é idoso;
  - Se o cliente tem um parceiro (Casado ou Namorando);
  - Se o cliente tem dependentes (alguém que depende financeiramente do cliente).
""")
st.image(Image.open('./assets/graphs/demographic-features.png'))


st.write("""
  ### 5 - Dados de Serviços
  
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
