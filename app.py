import streamlit as st
import pandas as pd
import joblib
import numpy as np
import matplotlib.pyplot as plt
model = joblib.load("model112.pkl")
label_encoder = joblib.load("label_encoder_1.pkl")
scaler = joblib.load("scaler.pkl")


def main():
    # Set the app title
    st.title("Customer Default Prediction API")

    # Create input fields for user data
    limit_bal = st.number_input("Limit Balance", min_value=0)
    sex = st.selectbox("Gender", ["Male", "Female"])
    education = st.selectbox("Education", ["Graduate School", "University", "High School", "PG_Int"])
    marriage = st.selectbox("Marriage", ["Married", "Single", "Others"])
    age = st.number_input("Age", min_value=0)
    pay_0 = st.number_input("PAY_0", min_value=0)
    pay_2 = st.number_input("PAY_2", min_value=0)
    pay_3 = st.number_input("PAY_3", min_value=0)
    pay_6 = st.number_input("PAY_6", min_value=0)
    bill_amt6 = st.number_input("BILL_AMT6", min_value=0)
    pay_amt1 = st.number_input("PAY_AMT1", min_value=0)
    pay_amt2 = st.number_input("PAY_AMT2", min_value=0)
    pay_amt3 = st.number_input("PAY_AMT3", min_value=0)
    pay_amt4 = st.number_input("PAY_AMT4", min_value=0)
    pay_amt5 = st.number_input("PAY_AMT5", min_value=0)
    pay_amt6 = st.number_input("PAY_AMT6", min_value=0)

    # Create a button to trigger prediction
    if st.button("Predict"):
        # Preprocess the user input
        user_data = [[limit_bal, sex, education, marriage, age, pay_0, pay_2, pay_3, pay_6, bill_amt6, pay_amt1, pay_amt2, pay_amt3, pay_amt4, pay_amt5, pay_amt6]]
        df = pd.DataFrame(user_data, columns=['LIMIT_BAL', 'Gender', 'EDUCATION', 'MARRIAGE', 'AGE', 'PAY_0', 'PAY_2', 'PAY_3', 'PAY_6', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3', 'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'])

        # Apply preprocessing steps to the input data
    
       # Apply label encoding
        df['Gender_Encoded'] = label_encoder.fit_transform(df['Gender'])
        df['Education_Encoded'] = label_encoder.fit_transform(df['EDUCATION'])
        df['Marital_Status_Encoded'] = label_encoder.fit_transform(df['MARRIAGE'])

        columns_to_scale = ['LIMIT_BAL', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_6', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6',
       'Gender_Encoded', 'Education_Encoded', 'Marital_Status_Encoded']
        
        # Perform feature scaling
        df[columns_to_scale] = scaler.transform(df[columns_to_scale])
        new_df1 =df[['LIMIT_BAL','Gender_Encoded', 'Education_Encoded', 'Marital_Status_Encoded', 'AGE', 'PAY_0', 'PAY_2',
       'PAY_3', 'PAY_6', 'BILL_AMT6', 'PAY_AMT1', 'PAY_AMT2', 'PAY_AMT3',
       'PAY_AMT4', 'PAY_AMT5', 'PAY_AMT6'
       ]]

        # Make predictions
        predictions = model.predict(new_df1)

        # Display the prediction result
        if predictions == 1:
            st.write("Prediction: The customer will default")
        else:
            st.write("Prediction: The customer will not default")
    

if __name__ == '__main__':
    main()
