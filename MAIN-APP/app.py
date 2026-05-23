import streamlit as st
import pickle
import os

# title and subtitle
st.title("Telecom Churn Analysis")
st.write("Predict whether a customer will churn based on their call usage patterns.")

# pre-trained models
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

# Use tabs
with st.container():
    col1, col2, col3 = st.columns(3)

    # Local Outgoing Call to Other Operator
    loc_og_t2o_mou = col1.number_input("Local Outgoing Calls to Other Operator (MOU)", min_value=0.0)
    
    # STD Outgoing Call to Other Operator
    std_og_t2o_mou = col2.number_input("STD Outgoing Calls to Other Operator (MOU)", min_value=0.0)
    
    # Local Incoming Call from Other Operator
    loc_ic_t2o_mou = col3.number_input("Local Incoming Calls from Other Operator (MOU)", min_value=0.0)
    
    st.subheader("Average Revenue Per Unit (ARPU)")
    with col1: arpu_6 = col1.number_input("6th Month (ARPU)", min_value=0.0)
    with col2: arpu_7 = col2.number_input("7th Month (ARPU)", min_value=0.0)
    with col3: arpu_8 = col3.number_input("8th Month (ARPU)", min_value=0.0)
    
    # MOU fields
    st.subheader("Minutes of Usage (MOU)")
    onnet_mou_6 = col1.number_input("On-Net (6th Month)", min_value=0.0)
    onnet_mou_7 = col2.number_input("On-Net (7th Month)", min_value=0.0)
    onnet_mou_8 = col3.number_input("On-Net (8th Month)", min_value=0.0)

    offnet_mou_6 = col1.number_input("Off-Net (6th Month)", min_value=0.0)
    offnet_mou_7 = col2.number_input("Off-Net (7th Month)", min_value=0.0)
    offnet_mou_8 = col3.number_input("Off-Net (8th Month)", min_value=0.0)

    roam_ic_mou_6 = col1.number_input("Roaming Incoming Calls (6th Month)", min_value=0.0)
    roam_ic_mou_7 = col2.number_input("Roaming Incoming Calls (7th Month)", min_value=0.0)
    roam_ic_mou_8 = col3.number_input("Roaming Incoming Calls (8th Month)", min_value=0.0)
    
    roam_og_mou_6 = col1.number_input("Roaming Outgoing Calls (6th Month)", min_value=0.0)
    roam_og_mou_7 = col2.number_input("Roaming Outgoing Calls (7th Month)", min_value=0.0)
    roam_og_mou_8 = col3.number_input("Roaming Outgoing Calls (8th Month)", min_value=0.0)
    
    # Local outgoing calls within same operator
    loc_og_t2t_mou_6 = col1.number_input("Local Outgoing Calls to Same Operator (6th Month)", min_value=0.0)
    loc_og_t2t_mou_7 = col2.number_input("Local Outgoing Calls to Same Operator (7th Month)", min_value=0.0)
    loc_og_t2t_mou_8 = col3.number_input("Local Outgoing Calls to Same Operator (8th Month)", min_value=0.0)
    
    # Local outgoing calls to mobile
    loc_og_t2m_mou_6 = col1.number_input("Local Outgoing Calls to Mobile (6th Month)", min_value=0.0)
    loc_og_t2m_mou_7 = col2.number_input("Local Outgoing Calls to Mobile (7th Month)", min_value=0.0)
    loc_og_t2m_mou_8 = col3.number_input("Local Outgoing Calls to Mobile (8th Month)", min_value=0.0)

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