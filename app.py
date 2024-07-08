import streamlit as st
import pandas as pd
import numpy as np
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

# Judul dengan gaya khusus menggunakan HTML
st.markdown('<p style="font-size:24px; color:blue;">All the Features you need to input to start the Prediction</p>', unsafe_allow_html=True)

image_url = 'https://raw.githubusercontent.com/joshgian/Life-Expectancy-Regression/main/All%20Features.png'

# Menampilkan gambar dari URL
st.image(image_url, caption='All Features', use_column_width=True)

# load dataset into pandas
st.title('Example Value to input')
df = pd.read_csv('Life Expectancy Data.csv')
df



# Daftar negara
countries = np.array(['Afghanistan', 'Albania', 'Algeria', 'Angola',
       'Antigua and Barbuda', 'Argentina', 'Armenia', 'Australia',
       'Austria', 'Azerbaijan', 'Bahamas', 'Bahrain', 'Bangladesh',
       'Barbados', 'Belarus', 'Belgium', 'Belize', 'Benin', 'Bhutan',
       'Bolivia (Plurinational State of)', 'Bosnia and Herzegovina',
       'Botswana', 'Brazil', 'Brunei Darussalam', 'Bulgaria',
       'Burkina Faso', 'Burundi', "Côte d'Ivoire", 'Cabo Verde',
       'Cambodia', 'Cameroon', 'Canada', 'Central African Republic',
       'Chad', 'Chile', 'China', 'Colombia', 'Comoros', 'Congo',
       'Cook Islands', 'Costa Rica', 'Croatia', 'Cuba', 'Cyprus',
       'Czechia', "Democratic People's Republic of Korea",
       'Democratic Republic of the Congo', 'Denmark', 'Djibouti',
       'Dominica', 'Dominican Republic', 'Ecuador', 'Egypt',
       'El Salvador', 'Equatorial Guinea', 'Eritrea', 'Estonia',
       'Ethiopia', 'Fiji', 'Finland', 'France', 'Gabon', 'Gambia',
       'Georgia', 'Germany', 'Ghana', 'Greece', 'Grenada', 'Guatemala',
       'Guinea', 'Guinea-Bissau', 'Guyana', 'Haiti', 'Honduras',
       'Hungary', 'Iceland', 'India', 'Indonesia',
       'Iran (Islamic Republic of)', 'Iraq', 'Ireland', 'Israel', 'Italy',
       'Jamaica', 'Japan', 'Jordan', 'Kazakhstan', 'Kenya', 'Kiribati',
       'Kuwait', 'Kyrgyzstan', "Lao People's Democratic Republic",
       'Latvia', 'Lebanon', 'Lesotho', 'Liberia', 'Libya', 'Lithuania',
       'Luxembourg', 'Madagascar', 'Malawi', 'Malaysia', 'Maldives',
       'Mali', 'Malta', 'Marshall Islands', 'Mauritania', 'Mauritius',
       'Mexico', 'Micronesia (Federated States of)', 'Monaco', 'Mongolia',
       'Montenegro', 'Morocco', 'Mozambique', 'Myanmar', 'Namibia',
       'Nauru', 'Nepal', 'Netherlands', 'New Zealand', 'Nicaragua',
       'Niger', 'Nigeria', 'Niue', 'Norway', 'Oman', 'Pakistan', 'Palau',
       'Panama', 'Papua New Guinea', 'Paraguay', 'Peru', 'Philippines',
       'Poland', 'Portugal', 'Qatar', 'Republic of Korea',
       'Republic of Moldova', 'Romania', 'Russian Federation', 'Rwanda',
       'Saint Kitts and Nevis', 'Saint Lucia',
       'Saint Vincent and the Grenadines', 'Samoa', 'San Marino',
       'Sao Tome and Principe', 'Saudi Arabia', 'Senegal', 'Serbia',
       'Seychelles', 'Sierra Leone', 'Singapore', 'Slovakia', 'Slovenia',
       'Solomon Islands', 'Somalia', 'South Africa', 'South Sudan',
       'Spain', 'Sri Lanka', 'Sudan', 'Suriname', 'Swaziland', 'Sweden',
       'Switzerland', 'Syrian Arab Republic', 'Tajikistan', 'Thailand',
       'The former Yugoslav republic of Macedonia', 'Timor-Leste', 'Togo',
       'Tonga', 'Trinidad and Tobago', 'Tunisia', 'Turkey',
       'Turkmenistan', 'Tuvalu', 'Uganda', 'Ukraine',
       'United Arab Emirates',
       'United Kingdom of Great Britain and Northern Ireland',
       'United Republic of Tanzania', 'United States of America',
       'Uruguay', 'Uzbekistan', 'Vanuatu',
       'Venezuela (Bolivarian Republic of)', 'Viet Nam', 'Yemen',
       'Zambia', 'Zimbabwe'])

# Judul aplikasi
st.title('Life Expectancy Prediction tools')

# Input dari pengguna
country = st.selectbox('Country', countries)
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
    input_data['Country'] = encoder_dict['Country'].transform(input_data['Country'])
    input_data['Status'] = encoder_dict['Status'].transform(input_data['Status'])
    
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
