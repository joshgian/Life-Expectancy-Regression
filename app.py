import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle


# Judul dengan gaya khusus menggunakan HTML
st.markdown('<p style="font-size:24px; color:blue;">All the Features you need to input to start the Prediction</p>', unsafe_allow_html=True)

image_url = 'https://raw.githubusercontent.com/joshgian/Life-Expectancy-Regression/main/All%20Features.png'

# Menampilkan gambar dari URL
st.image(image_url, caption='All Features', use_column_width=True)

# load dataset into pandas
st.title('Example Value to input')
df = pd.read_csv('Life Expectancy Data.csv')
df

# Judul aplikasi
st.title('Life Expectancy Prediction tools')

# Input dari pengguna
country = st.selectbox('Country', ['Country1', 'Country2', 'Country3'])  # Sesuaikan dengan daftar negara yang tersedia
year = st.slider('Year', min_value=2000, max_value=2015)
status = st.selectbox('Status', ['Developing', 'Developed'])
adult_mortality = st.number_input('Adult Mortality', min_value=0, max_value=1000)
infant_deaths = st.number_input('Infant Deaths', min_value=0, max_value=1000)
alcohol = st.number_input('Alcohol Consumption', min_value=0.0, max_value=20.0)
percentage_expenditure = st.number_input('Percentage Expenditure', min_value=0.0, max_value=1000.0)
hepatitis_b = st.number_input('Hepatitis B Immunization', min_value=0.0, max_value=100.0)
measles = st.number_input('Measles Cases', min_value=0, max_value=1000)
bmi = st.number_input('Average BMI', min_value=0.0, max_value=100.0)
under_five_deaths = st.number_input('Under-five Deaths', min_value=0, max_value=1000)
polio = st.number_input('Polio Immunization', min_value=0.0, max_value=100.0)
total_expenditure = st.number_input('Total Expenditure on Health', min_value=0.0, max_value=100.0)
diphtheria = st.number_input('Diphtheria Immunization', min_value=0.0, max_value=100.0)
hiv_aids = st.number_input('HIV/AIDS Deaths', min_value=0.0, max_value=100.0)
gdp = st.number_input('GDP per capita', min_value=0.0, max_value=100000.0)
population = st.number_input('Population', min_value=0, max_value=1000000000)
thinness_1_19_years = st.number_input('Thinness 10-19 Years', min_value=0.0, max_value=100.0)
thinness_5_9_years = st.number_input('Thinness 5-9 Years', min_value=0.0, max_value=100.0)
income_composition_of_resources = st.number_input('Income Composition of Resources', min_value=0.0, max_value=1.0)
schooling = st.number_input('Schooling (Years)', min_value=0.0, max_value=20.0)

# Prediksi ketika tombol diklik
if st.button('Predict'):
    # Muat model, encoder, dan scaler dari file
    model = load_model('model_ann.h5')

    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Transform input dari pengguna
    input_data = pd.DataFrame([[country, year, status, adult_mortality, infant_deaths, alcohol, percentage_expenditure,
                                hepatitis_b, measles, bmi, under_five_deaths, polio, total_expenditure, diphtheria,
                                hiv_aids, gdp, population, thinness_1_19_years, thinness_5_9_years,
                                income_composition_of_resources, schooling]],
                              columns=['Country', 'Year', 'Status', 'Adult Mortality', 'Infant Deaths', 'Alcohol',
                                       'Percentage Expenditure', 'Hepatitis B', 'Measles', 'BMI', 'Under-five Deaths',
                                       'Polio', 'Total Expenditure', 'Diphtheria', 'HIV/AIDS', 'GDP', 'Population',
                                       'Thinness 1-19 Years', 'Thinness 5-9 Years', 'Income Composition of Resources',
                                       'Schooling'])
    
    # Encode categorical features
    input_data['Country'] = label_encoder.transform(input_data['Country'])
    input_data['Status'] = label_encoder.transform(input_data['Status'])
    
    # Scale numerical features
    input_data[['Year', 'Adult Mortality', 'Infant Deaths', 'Alcohol', 'Percentage Expenditure', 'Hepatitis B', 
                'Measles', 'BMI', 'Under-five Deaths', 'Polio', 'Total Expenditure', 'Diphtheria', 'HIV/AIDS', 
                'GDP', 'Population', 'Thinness 1-19 Years', 'Thinness 5-9 Years', 'Income Composition of Resources', 
                'Schooling']] = scaler.transform(input_data[['Year', 'Adult Mortality', 'Infant Deaths', 'Alcohol', 
                                                             'Percentage Expenditure', 'Hepatitis B', 'Measles', 'BMI', 
                                                             'Under-five Deaths', 'Polio', 'Total Expenditure', 
                                                             'Diphtheria', 'HIV/AIDS', 'GDP', 'Population', 
                                                             'Thinness 1-19 Years', 'Thinness 5-9 Years', 
                                                             'Income Composition of Resources', 'Schooling']])
    
    # Prediksi
    prediction = model.predict(input_data)
    st.write(f'Predicted Life Expectancy: {prediction[0][0]:.2f} years')
