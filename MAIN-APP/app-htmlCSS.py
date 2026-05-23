import streamlit as st
import pickle
import os

# Load pre-trained models
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_model1 = pickle.load(open(os.path.join(script_dir, 'scaler.pk1'), 'rb'))
    loaded_model2 = pickle.load(open(os.path.join(script_dir, 'pca1.pk1'), 'rb'))
    loaded_model3 = pickle.load(open(os.path.join(script_dir, 'predict_model.pk1'), 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Please ensure scaler.pk1, pca1.pk1, and predict_model.pk1 are in the MAIN-APP directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()

# Render HTML content from index.html
html_path = os.path.join(script_dir, "index.html")
with open(html_path, "r") as f:
    index_html = f.read()
st.markdown(index_html, unsafe_allow_html=True)

# Render CSS content from styles.css
css_path = os.path.join(script_dir, "styles.css")
with open(css_path, "r") as f:
    styles_css = f.read()
st.markdown(f"<style>{styles_css}</style>", unsafe_allow_html=True)

# Use tabs
with st.container():
    col1, col2, col3 = st.columns(3)

    # Local Outgoing Call to Other Operator
    with col1: loc_og_t2o_mou = col1.number_input("Local Outgoing Calls to Other Operator (MOU)", min_value=0.0)
    
    # STD Outgoing Call to Other Operator
    with col2: std_og_t2o_mou = col2.number_input("STD Outgoing Calls to Other Operator (MOU)", min_value=0.0)
    
    # Local Incoming Call from Other Operator
    with col3: loc_ic_t2o_mou = col3.number_input("Local Incoming Calls from Other Operator (MOU)", min_value=0.0)
    
    # ARPU fields
    st.subheader("Average Revenue Per Unit (ARPU)")
    with col1: arpu_6 = col1.number_input("6th Month (ARPU)", min_value=0.0)
    with col2: arpu_7 = col2.number_input("7th Month (ARPU)", min_value=0.0)
    with col3: arpu_8 = col3.number_input("8th Month (ARPU)", min_value=0.0)
    
    # MOU fields
    st.subheader("Minutes of Usage (MOU)")
    with col1: onnet_mou_6 = col1.number_input("On-Net (6th Month)", min_value=0.0)
    with col2: onnet_mou_7 = col2.number_input("On-Net (7th Month)", min_value=0.0)
    with col3: onnet_mou_8 = col3.number_input("On-Net (8th Month)", min_value=0.0)

    with col1: offnet_mou_6 = col1.number_input("Off-Net (6th Month)", min_value=0.0)
    with col2: offnet_mou_7 = col2.number_input("Off-Net (7th Month)", min_value=0.0)
    with col3: offnet_mou_8 = col3.number_input("Off-Net (8th Month)", min_value=0.0)

    with col1: roam_ic_mou_6 = col1.number_input("Roaming Incoming Calls (6th Month)", min_value=0.0)
    with col2: roam_ic_mou_7 = col2.number_input("Roaming Incoming Calls (7th Month)", min_value=0.0)
    with col3: roam_ic_mou_8 = col3.number_input("Roaming Incoming Calls (8th Month)", min_value=0.0)

    with col1: roam_og_mou_6 = col1.number_input("Roaming Outgoing Calls (6th Month)", min_value=0.0)
    with col2: roam_og_mou_7 = col2.number_input("Roaming Outgoing Calls (7th Month)", min_value=0.0)
    with col3: roam_og_mou_8 = col3.number_input("Roaming Outgoing Calls (8th Month)", min_value=0.0)
    
    # Local outgoing calls within same operator
    with col1: loc_og_t2t_mou_6 = col1.number_input("Local Outgoing Calls to Same Operator (6th Month)", min_value=0.0)
    with col2: loc_og_t2t_mou_7 = col2.number_input("Local Outgoing Calls to Same Operator (7th Month)", min_value=0.0)
    with col3: loc_og_t2t_mou_8 = col3.number_input("Local Outgoing Calls to Same Operator (8th Month)", min_value=0.0)
    
    # Local outgoing calls to mobile
    with col1: loc_og_t2m_mou_6 = col1.number_input("Local Outgoing Calls to Mobile (6th Month)", min_value=0.0)
    with col2: loc_og_t2m_mou_7 = col2.number_input("Local Outgoing Calls to Mobile (7th Month)", min_value=0.0)
    with col3: loc_og_t2m_mou_8 = col3.number_input("Local Outgoing Calls to Mobile (8th Month)", min_value=0.0)

# Collecting inputs into a list
X = [[loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou, arpu_6, arpu_7, arpu_8, onnet_mou_6, onnet_mou_7, onnet_mou_8, offnet_mou_6, offnet_mou_7, offnet_mou_8, roam_ic_mou_6, roam_ic_mou_7, roam_ic_mou_8, roam_og_mou_6, roam_og_mou_7, roam_og_mou_8, loc_og_t2t_mou_6, loc_og_t2t_mou_7, loc_og_t2t_mou_8, loc_og_t2m_mou_6, loc_og_t2m_mou_7, loc_og_t2m_mou_8]]

# Predict button
if st.button("Predict Churn"):
    try:
        # Transform input data using loaded models
        result1 = loaded_model1.transform(X)
        result2 = loaded_model2.transform(result1)
        prediction = loaded_model3.predict(result2)
        
        # Display prediction result
        if prediction > 0.5:
            st.subheader("Prediction: 🏃 This customer is likely to churn.")
        else:
            st.subheader("Prediction: 💰 This customer is unlikely to churn.")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
