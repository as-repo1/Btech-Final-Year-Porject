import streamlit as st
import pickle
import os

st.title("TELECOM CHURN ANALYSIS")

# load pre-trained model
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

col1, col2, col3 = st.columns(3)
with col1:
    loc_og_t2o_mou = st.number_input("Minutes of Usage of Local Outgoing Calls to Other Operator", min_value=0.0)
with col2:
    std_og_t2o_mou = st.number_input("Minutes of Usage of STD Outgoing Calls to Other Operator", min_value=0.0)
with col3:
    loc_ic_t2o_mou = st.number_input("Minutes of Usage of Local Incoming Calls from Other Operator", min_value=0.0)

with col1:
    arpu_6 = st.number_input("Average revenue per unit in 6th month", min_value=0.0)
with col2:
    arpu_7 = st.number_input("Average revenue per unit in 7th month", min_value=0.0)
with col3:
    arpu_8 = st.number_input("Average revenue per unit in 8th month", min_value=0.0)

with col1:
    onnet_mou_6 = st.number_input("Minutes of usage of all kind of call in same network month of 6", min_value=0.0)
with col2:
    onnet_mou_7 = st.number_input("Minutes of usage of all kind of call in same network month of 7", min_value=0.0)
with col3:
    onnet_mou_8 = st.number_input("Minutes of usage of all kind of call in same network month of 8", min_value=0.0)
with col1:
    offnet_mou_6 = st.number_input("Minutes of usage of all kind of call in other network month of 6", min_value=0.0)
with col2:
    offnet_mou_7 = st.number_input("Minutes of usage of all kind of call in other network month of 7", min_value=0.0)
with col3:
    offnet_mou_8 = st.number_input("Minutes of usage of all kind of call in other network month of 8", min_value=0.0)
with col1:
    roam_ic_mou_6 = st.number_input("Minutes of usage of Roaming incoming call month of 6", min_value=0.0)
with col2:
    roam_ic_mou_7 = st.number_input("Minutes of usage of Roaming incoming call month of 7", min_value=0.0)
with col3:
    roam_ic_mou_8 = st.number_input("Minutes of usage of Roaming incoming call month of 8", min_value=0.0)
with col1:
    roam_og_mou_6 = st.number_input("Minutes of usage of Roaming outgoing call month of 6", min_value=0.0)
with col2:
    roam_og_mou_7 = st.number_input("Minutes of usage of Roaming outgoing call month of 7", min_value=0.0)
with col3:
    roam_og_mou_8 = st.number_input("Minutes of usage of Roaming outgoing call month of 8", min_value=0.0)
with col1:
    loc_og_t2t_mou_6 = st.number_input("Minutes of usage of Local outgoing calls within same operator month of 6", min_value=0.0)
with col2:
    loc_og_t2t_mou_7 = st.number_input("Minutes of usage of Local outgoing calls within same operator month of 7", min_value=0.0)
with col3:
    loc_og_t2t_mou_8 = st.number_input("Minutes of usage of Local outgoing calls within same operator month of 8", min_value=0.0)
with col1:
    loc_og_t2m_mou_6 = st.number_input("Minutes of usage of local outgoing calls to mobile in month of 6", min_value=0.0)
with col2:
    loc_og_t2m_mou_7 = st.number_input("Minutes of usage of local outgoing calls to mobile in month of 7", min_value=0.0)
with col3:
    loc_og_t2m_mou_8 = st.number_input("Minutes of usage of local outgoing calls to mobile in month of 8", min_value=0.0)

X = [[loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou, arpu_6, arpu_7, arpu_8, onnet_mou_6, onnet_mou_7, onnet_mou_8, offnet_mou_6, offnet_mou_7, offnet_mou_8, roam_ic_mou_6, roam_ic_mou_7, roam_ic_mou_8, roam_og_mou_6, roam_og_mou_7, roam_og_mou_8, loc_og_t2t_mou_6, loc_og_t2t_mou_7, loc_og_t2t_mou_8, loc_og_t2m_mou_6, loc_og_t2m_mou_7, loc_og_t2m_mou_8]]

if st.button("Click here to Predict"):
    try:
        result1 = loaded_model1.transform(X)
        result2 = loaded_model2.transform(result1)
        result = loaded_model3.predict(result2)
        if result > 0.5:
            st.subheader("This customer is likely to churn")
        else:
            st.subheader("This customer is unlikely to churn")
    except Exception as e:
        st.error(f"Prediction failed: {e}")
