import streamlit as st
import pickle
import pandas as pd

#load the model
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

#make prediction using loaded-model
def predict_churn(features):
    prediction = model.predict(features)
    return prediction

def main():
    st.title('churn Prediction App')
    st.sidebar.title('Input Features')

    arpu_6 = st.sidebar.number_input('ARPU for month 6', min_value=0.0, max_value=10000.0, step=1.0)
    arpu_7 = st.sidebar.number_input('ARPU for month 7', min_value=0.0, max_value=10000.0, step=1.0)

    input_data = pd.DataFrame({'arpu_6': [arpu_6], 'arpu_7': [arpu_7]})

    if st.sidebar.button('Predict Churn'):
        prediction = predict_churn(input_data)
        st.write('Predicted Churn:', prediction)

if __name__ == '__main__':
    main()