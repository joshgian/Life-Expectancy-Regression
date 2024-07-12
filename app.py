import streamlit as st
import pandas as pd
from tensorflow.keras.models import load_model
import pickle

# Load model
model = load_model('model_ann.h5')

# Load encoders
with open('encoder.pkl', 'rb') as encoder_file:
    encoder_dict = pickle.load(encoder_file)

# Load scaler
with open('scaler.pkl', 'rb') as scaler_file:
    scaler = pickle.load(scaler_file)

# Input features
st.title('All the Features you need to input to start the Prediction')
image_url = 'https://raw.githubusercontent.com/joshgian/Life-Expectancy-Regression/main/All%20Features.png'
st.image(image_url, caption='All Features', use_column_width=True)
df = pd.read_csv('Life Expectancy Data.csv')
st.title('Example Value to input')
st.dataframe(df.head())

st.title('Life Expectancy Prediction Tools')
# Collect input data
country = st.selectbox('Country', df['Country'].unique())
year = st.number_input('Year', min_value=2000, max_value=2015, step=1)
status = st.selectbox('Status', ['Developing', 'Developed'])
adult_mortality = st.number_input('Adult Mortality', min_value=0.0, max_value=1000.0, step=1.0)
infant_deaths = st.number_input('infant deaths', min_value=0.0, max_value=1000.0, step=1.0)
alcohol = st.number_input('Alcohol', min_value=0.0, max_value=20.0, step=0.1)
percentage_expenditure = st.number_input('percentage expenditure', min_value=0.0, max_value=10000.0, step=1.0)
hep_b = st.number_input('Hepatitis B', min_value=0.0, max_value=100.0, step=1.0)
measles = st.number_input('Measles', min_value=0.0, max_value=100000.0, step=1.0)
bmi = st.number_input('BMI', min_value=0.0, max_value=100.0, step=0.1)
under_five_deaths = st.number_input('under-five deaths', min_value=0.0, max_value=1000.0, step=1.0)
polio = st.number_input('Polio', min_value=0.0, max_value=100.0, step=1.0)
total_expenditure = st.number_input('Total expenditure', min_value=0.0, max_value=100.0, step=1.0)
diphtheria = st.number_input('Diphtheria', min_value=0.0, max_value=100.0, step=1.0)
hiv_aids = st.number_input('HIV/AIDS', min_value=0.0, max_value=100.0, step=1.0)
gdp = st.number_input('GDP', min_value=0.0, max_value=100000.0, step=1.0)
population = st.number_input('Population', min_value=0.0, max_value=1000000000.0, step=1.0)
thinness_1_19_years = st.number_input('thinness  1-19 years', min_value=0.0, max_value=100.0, step=0.1)
thinness_5_9_years = st.number_input('thinness 5-9 years', min_value=0.0, max_value=100.0, step=0.1)
income_composition_of_resources = st.number_input('Income composition of resources', min_value=0.0, max_value=1.0, step=0.01)
schooling = st.number_input('Schooling', min_value=0.0, max_value=20.0, step=0.1)

# Prediction button
if st.button('Predict'):
    input_data = pd.DataFrame([[country, year, status, adult_mortality, infant_deaths, alcohol, 
                                percentage_expenditure, hep_b, measles, bmi, under_five_deaths, polio, 
                                total_expenditure, diphtheria, hiv_aids, gdp, population, thinness_1_19_years, 
                                thinness_5_9_years, income_composition_of_resources, schooling]],
                              columns=['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                                       'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 
                                       'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 
                                       'thinness  1-19 years', 'thinness 5-9 years', 'Income composition of resources', 
                                       'Schooling'])

    # Encode categorical features
    input_data['Country'] = encoder_dict['Country'].transform(input_data['Country'])
    input_data['Status'] = encoder_dict['Status'].transform(input_data['Status'])

    # Scale numerical features
    cols_to_scale = ['Year', 'Adult Mortality', 'infant deaths', 'Alcohol', 'percentage expenditure', 
                     'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 'Polio', 'Total expenditure', 
                     'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 'thinness  1-19 years', 'thinness 5-9 years', 
                     'Income composition of resources', 'Schooling']
    input_data[cols_to_scale] = scaler.transform(input_data[cols_to_scale])

    # Ensure column order matches
    input_data = input_data[['Country', 'Year', 'Status', 'Adult Mortality', 'infant deaths', 'Alcohol', 
                             'percentage expenditure', 'Hepatitis B', 'Measles', 'BMI', 'under-five deaths', 
                             'Polio', 'Total expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 
                             'thinness  1-19 years', 'thinness 5-9 years', 'Income composition of resources', 
                             'Schooling']]

    # Make prediction
    prediction = model.predict(input_data)
    st.write(f'Predicted Life Expectancy: {prediction[0][0]:.2f} years')
