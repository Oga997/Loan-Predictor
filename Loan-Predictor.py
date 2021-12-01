#importing required libraries
import numpy as np
import pandas as pd
import streamlit as st
import pickle


model = pickle.load(open('./predictor.pkl', 'rb'))

st.header("Check if you are eligible to get a loan")
account_no = st.text_input('Enter Account number')

    ## Full Name
fn = st.text_input('Full Name')

    ## For edu
edu_display = ('Not Graduate','Graduate')
edu_options = list(range(len(edu_display)))
edu = st.selectbox("Education",edu_options, format_func=lambda x: edu_display[x])

    ## For emp status
emp_display = ('Job','Business')
emp_options = list(range(len(emp_display)))
emp = st.selectbox("Employment Status",emp_options, format_func=lambda x: emp_display[x])

    ## For Property status
prop_display = ('Rural','Semi-Urban','Urban')
prop_options = list(range(len(prop_display)))
prop = st.selectbox("Property Area",prop_options, format_func=lambda x: prop_display[x])

    ## For Credit Score
cred_display = ('Below 500','Above 750')
cred_options = list(range(len(cred_display)))
cred = st.selectbox("Credit Score",cred_options, format_func=lambda x: cred_display[x])

    ## Applicant Monthly Income
mon_income = st.number_input("Applicant's Monthly Income($)",value=0)

    ## Co-Applicant Monthly Income
co_mon_income = st.number_input("Co-Applicant's Monthly Income($)",value=0)

    ## Loan AMount
loan_amt = st.number_input("Loan Amount",value=0)

    ## loan duration
dur_display = ['2 Month','6 Month','8 Month','1 Year','16 Month']
dur_options = range(len(dur_display))
dur = st.selectbox("Loan Duration",dur_options, format_func=lambda x: dur_display[x])

if st.button("Predict"):
    duration = 0
    if dur == 0:
        duration = 60
    if dur == 1:
        duration = 180
    if dur == 2:
        duration = 240
    if dur == 3:
        duration = 360
    if dur == 4:
        duration = 480
    features = [[edu, emp, mon_income, co_mon_income, loan_amt, duration, cred, prop]]
    print(features)
    prediction = model.predict(features)
    lc = [str(i) for i in prediction]
    ans = int("".join(lc))
    if ans == 0:
        st.error(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'According to our Calculations, you will not get the loan from Bank')
    else:
        st.success(
                "Hello: " + fn +" || "
                "Account number: "+account_no +' || '
                'Congratulations!! you will get the loan from Bank')






