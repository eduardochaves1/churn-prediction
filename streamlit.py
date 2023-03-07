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

st.sidebar.write('# **Parameters Input**')

def input_features():
  def get_sidebar_radio(title, choices=('No', 'Yes'), horizontal=True):
    return st.sidebar.radio(f'**{title}:**', choices, horizontal=horizontal)

  gender = get_sidebar_radio('Gender', ('Male', 'Female'))
  seniorCitizen = get_sidebar_radio('Senior')
  partner = get_sidebar_radio('Partner')
  dependents = get_sidebar_radio('Dependents')
  phoneService = get_sidebar_radio('Phone Service')
  multipleLines = get_sidebar_radio('Multiple Lines')
  internetService = get_sidebar_radio('Internet Service', ('No', 'Fiber optic', 'DSL'), horizontal=False)
  onlineSecurity = get_sidebar_radio('Online Security')
  onlineBackup = get_sidebar_radio('Online Backup')
  deviceProtection = get_sidebar_radio('Device Protection')
  techSupport = get_sidebar_radio('Tech Support')
  streamingTV = get_sidebar_radio('TV Streaming')
  streamingMovies = get_sidebar_radio('Movie Streaming')
  contract = get_sidebar_radio('Contract', ('Month-to-month', 'One year', 'Two year'), horizontal=False)
  paperlessBilling = get_sidebar_radio('Paperless Billing')
  paymentMethod = get_sidebar_radio('Payment Method', ('Credit card (automatic)', 'Bank transfer (automatic)', 'Electronic check', 'Mailed check'), horizontal=False)
  tenure = st.sidebar.slider('**Tenure (Months):**', min_value=0, max_value=70, value=12)
  monthlyCharges = st.sidebar.number_input('**Monthly Charges ($):**', min_value=10.0, max_value=120.0, value=65.0)

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
  - **[GitHub Repository](https://github.com/eduardochaves1/churn-prediction 'Churn Prediction by Eduardo Chaves')**
""")


st.image(Image.open('assets/banner_img.jpg'))


st.write("""
  ## What Problem we're Tying to Solve?

  Imagine you own a telecommunication company and you want to have a Machine Learning model to predict the custumers that may probably churn in the next months. So I developed one and deployed in this prototype web application that would help you in this situation!

  For those who don't know, churn is when a custumer stops paying for a company's service. And as I've heard once, it is cheaper to keep the custumer you already have instead of spending more and more money and time to gather new ones.
""")


st.write("""
  ## How to use this tool?

  You only need to provide the parameters to the machine learning model at the sidebar on the left side of this page. And the predictions made by the model will be outputted right below.
""")


st.write("""
  ## Parameters Imputed:

  - Down below are the parameters setted up to the model by the inputs of the sidebar.
""")
st.write(data)


st.write('## Prediction Results:')

prediction = get_prediction(data)
predictionMsg = '***Not Churn***' if float(prediction['Churn'][0][:-1]) <= 50 else '***Churn***'
predictionPercent = prediction['Not Churn'][0] if float(prediction['Churn'][0][:-1]) <= 50 else prediction['Churn'][0]

st.write(f'The model predicted a percentage of **{predictionPercent}** that the custumer will {predictionMsg}!')
st.write(prediction)


st.write("""
  ## Interpreting the Machine Learning Model:

  Down below follows the features that most affects the model classification of Churn and Not Churn.
""")
st.image(Image.open('./assets/graphs/feature-importance.png'))


st.write("""
  ---
  ## How this model was trained?

  It was trained on a dataset with around seven thousand data registries from a telecommunication company.

  Down below I'll show some statistical facts about this dataset, but if you want to see the most technical side you can take a look at this project's [GitHub Repository](https://github.com/eduardochaves1/churn-prediction 'Churn Prediction by Eduardo Chaves').
""")

st.write('### 1 - Amount of Churn Cases on Previous Month')
st.image(Image.open('./assets/graphs/churn.png'))


st.write("""
  ### 2 -  Numerical Features of the Data Set

  With these boxplot graphs we can see some statistical values from these numerical features - like median, min, max and more.

  - **Tenure:** Amount of months paying for the service;
  - **Monthly Charges:** Amount of money monthly paid for the service;
  - **Total Charges:** Multiplication of Tenure and Monthly Charges (this feature wasn't used on the model final training as this didn't include any more information about a custumer);
""")
st.image(Image.open('./assets/graphs/numeric-features.png'))


st.write("""
  ### 3 - Features about the Contract

  In these graphs we can see the amount of custumers that had these types of contract for the services they use.
""")
st.image(Image.open('./assets/graphs/contract-features.png'))


st.write("""
  ### 4 - Demographic Features

  Here we can see the amount of people that have these particular characteristics in their personal lifes.
""")
st.image(Image.open('./assets/graphs/demographic-features.png'))


st.write("""
  ### 5 - Service's Features

  Most of the features included in the data set was about the services that a custumer is paying for, and these are the amouts of people that used each one.
""")
st.image(Image.open('./assets/graphs/services-features.png'))


st.write("""
  ---
  > ***"Statistical thinking will one day be as necessary for efficient citizenship as the ability to read and write."*** - H.G. Wells (writer and author of "The war of the worlds" and "The time machine")
""")
