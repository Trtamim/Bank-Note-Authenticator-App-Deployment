# -*- coding: utf-8 -*-
"""
Created on Thu Jul  2 20:16:17 2020

@author: Tamim
"""
import pandas as pd
import numpy as np
import pickle
import streamlit as st 

# Load the Random Forest CLassifier model
filename = 'Bank_note-Authentication-rfc-model.pkl'
classifier = pickle.load(open(filename, 'rb'))


def predict_note_authentication(variance,skewness,curtosis,entropy):
    prediction=classifier.predict([[variance,skewness,curtosis,entropy]])
    print(prediction)
    return prediction


def main():
    st.title("Welcome To Check The Bank-Note Authentication")
    html_temp = """
    <div style="background-color:#4794ff;padding:10px">
    <h2 style="color:Black;text-align:center;">Bank-Note Authenticator ML App </h2>
    </div>
    """
    st.markdown(html_temp,unsafe_allow_html=True)
    variance = st.number_input('variance', min_value=-7.0421, max_value=6.8248, value=.10)
    skewness = st.number_input('skewness', min_value=-13.7731, max_value=12.9516, value=3.331)
    curtosis = st.number_input('curtosis', min_value=-5.2861, max_value=17.9274, value=-2.2144)
    entropy = st.number_input('entropy', min_value=-8.5482, max_value=2.4495, value=-3.5144)
    
    result=""
    if st.button("Predict"):
        result=predict_note_authentication(variance,skewness,curtosis,entropy)
    st.success('The output is {}'.format(result))
    if st.button("Made by"):
        st.text("Tiabur Rahman Tamim")
       
       
if __name__=='__main__':
    main()