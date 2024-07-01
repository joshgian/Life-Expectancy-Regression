import streamlit as st
import pandas as pd
import numpy as np
from tensorflow.keras.models import load_model
import pickle


st.title('All the Features you need to input to start the Prediction')
image_url = 'https://raw.githubusercontent.com/joshgian/Life-Expectancy-Regression/main/All%20Features.png'

# Menampilkan gambar dari URL
st.image(image_url, caption='All Features', use_column_width=True)

# load dataset into pandas
st.title('Value to input')
df = pd.read_csv('Life Expectancy Data.csv')
df

# Judul aplikasi
st.title('Sistem Prediksi Umur Harapan Hidup')

# Input dari pengguna
feature1 = st.number_input('Feature1', min_value=0.0, max_value=100.0)
feature2 = st.number_input('Feature2', min_value=0.0, max_value=100.0)
category = st.selectbox('Category', ['A', 'B', 'C'])

# Prediksi ketika tombol diklik
if st.button('Predict'):
    # Muat model, encoder, dan scaler dari file
    model = load_model('model_ann.h5')

    with open('label_encoder.pkl', 'rb') as encoder_file:
        label_encoder = pickle.load(encoder_file)

    with open('scaler.pkl', 'rb') as scaler_file:
        scaler = pickle.load(scaler_file)

    # Transform input dari pengguna
    input_data = pd.DataFrame([[category, feature1, feature2]], columns=['Category', 'Feature1', 'Feature2'])
    input_data['Category'] = label_encoder.transform(input_data['Category'])
    input_data[['Feature1', 'Feature2']] = scaler.transform(input_data[['Feature1', 'Feature2']])

    # Prediksi
    prediction = model.predict(input_data)
    st.write('Predicted Life Expectancy:', prediction[0][0])
