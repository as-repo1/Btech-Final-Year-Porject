import streamlit as st

# Read HTML and CSS files
with open("index.html", "r") as f:
    index_html = f.read()

with open("styles.css", "r") as f:
    styles_css = f.read()

# Inject the CSS into the app
st.markdown(
    f"<style>{styles_css}</style>",
    unsafe_allow_html=True
)

# Render the HTML content
st.markdown(index_html, unsafe_allow_html=True)

# Add your Streamlit widgets here
# For example, you can include form inputs, buttons, etc.
col1, col2, col3 = st.columns(3)
loc_og_t2o_mou = col1.number_input("Local Outgoing Calls to Other Operator (MOU)")
std_og_t2o_mou = col2.number_input("STD Outgoing Calls to Other Operator (MOU)")
loc_ic_t2o_mou = col3.number_input("Local Incoming Calls from Other Operator (MOU)")

# Add more form inputs and widgets as needed

# Your code for handling the predict button and displaying the prediction
# For example:
X = [[loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou]]

if st.button("Predict Churn"):
    # Your model prediction code goes here
    prediction = your_model_predict_function(X)
    
    if prediction > 0.5:
        st.subheader("Prediction: 🏃 This customer is likely to churn.")
    else:
        st.subheader("Prediction: 💰 This customer is unlikely to churn.")
