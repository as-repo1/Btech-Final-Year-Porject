import streamlit as st

# Define the Streamlit app
def main():
    # Set the title of the app
    st.title('Customer Churn Prediction')

    # Add a description
    st.write("Enter the Information Below and find if a Customer will Churn out or not")

    # Add input fields for each feature
    st.write("Input 1")
    input_1 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 2")
    input_2 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 3")
    input_3 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 4")
    input_4 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 5")
    input_5 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 6")
    input_6 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 7")
    input_7 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 8")
    input_8 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 9")
    input_9 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 10")
    input_10 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 11")
    input_11 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 12")
    input_12 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 13")
    input_13 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 14")
    input_14 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 15")
    input_15 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 16")
    input_16 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 17")
    input_17 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 18")
    input_18 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 19")
    input_19 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 20")
    input_20 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 21")
    input_21 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 22")
    input_22 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 23")
    input_23 = st.number_input(label='', value=0.0, step=1.0)

    st.write("Input 24")
    input_24 = st.number_input(label='', value=0.0, step=1.0)

    # Add a button to make predictions
    if st.button("Predict whether the Customer will leave the churn or not?"):
        # Perform prediction logic here using the inputs
        # prediction = model.predict([input_1, input_2, ..., input_24])
        prediction = "Prediction goes here"
        st.write(f"Predicted churn: {prediction}")

# Run the Streamlit app
if __name__ == '__main__':
    main()
