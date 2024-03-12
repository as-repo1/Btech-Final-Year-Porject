import streamlit as st
import pickle

# Load the model
with open('lr_model.pkl', 'rb') as file:
    model = pickle.load(file)

# Define the function to make predictions
def predict_churn(features):
    prediction = model.predict(features)
    return prediction

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Customer Churn Prediction')

    # Add a description
    st.write("Enter the Information Below and find if a Customer will Churn out or not")

    # Add input fields for each feature
    for i in range(1, 25):
        st.write(f"Input {i}")
        input_value = st.number_input(label='', value=0.0, step=1.0, key=f'input_{i}')

    # Add a button to make predictions
    if st.button("Predict whether the Customer will leave the churn or not?"):
        # Get input values
        features = [st.session_state[f'input_{i}'] for i in range(1, 25)]
        
        # Perform prediction
        prediction = predict_churn([features])
        
        # Display prediction
        st.write(f"Predicted churn: {prediction}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
