import streamlit as st
import pickle

# Read HTML and CSS files
with open("index.html", "r") as f:
    index_html = f.read()

with open("styles.css", "r") as f:
    styles_css = f.read()

# Inject the CSS styles into the app
st.markdown(f"<style>{styles_css}</style>", unsafe_allow_html=True)

# Render the HTML content
st.markdown(index_html, unsafe_allow_html=True)

# Load pre-trained models
loaded_model1 = pickle.load(open('scaler.pk1', 'rb'))
loaded_model2 = pickle.load(open('pca1.pk1', 'rb'))
loaded_model3 = pickle.load(open('final_model1.pk1', 'rb'))

# Streamlit app widgets and inputs
col1, col2, col3 = st.columns(3)

# Define inputs using the columns
loc_og_t2o_mou = col1.number_input("Local Outgoing Calls to Other Operator (MOU)")
std_og_t2o_mou = col2.number_input("STD Outgoing Calls to Other Operator (MOU)")
loc_ic_t2o_mou = col3.number_input("Local Incoming Calls from Other Operator (MOU)")

# ... (additional inputs and widgets as per your code)

# Collecting inputs into a list
X = [[loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou, arpu_6, arpu_7, arpu_8, onnet_mou_6, onnet_mou_7, onnet_mou_8, offnet_mou_6, offnet_mou_7, offnet_mou_8, roam_ic_mou_6, roam_ic_mou_7, roam_ic_mou_8, roam_og_mou_6, roam_og_mou_7, roam_og_mou_8, loc_og_t2t_mou_6, loc_og_t2t_mou_7, loc_og_t2t_mou_8, loc_og_t2m_mou_6, loc_og_t2m_mou_7, loc_og_t2m_mou_8]]

# Predict button functionality
if st.button("Predict Churn"):
    # Apply transformations and make a prediction
    result1 = loaded_model1.transform(X)
    result2 = loaded_model2.transform(result1)
    prediction = loaded_model3.predict(result2)
    
    # Display prediction result
    if prediction > 0.5:
        st.subheader("Prediction: 🏃 This customer is likely to churn.")
    else:
        st.subheader("Prediction: 💰 This customer is unlikely to churn.")
