import streamlit as st
import pickle
import os

# Resolve model paths relative to this script's directory
script_dir = os.path.dirname(os.path.abspath(__file__))
model_dir = os.path.join(script_dir, '..', 'MAIN-APP')

# Load the full pipeline: scaler -> PCA -> model
try:
    scaler = pickle.load(open(os.path.join(model_dir, 'scaler.pk1'), 'rb'))
    pca = pickle.load(open(os.path.join(model_dir, 'pca1.pk1'), 'rb'))
    model = pickle.load(open(os.path.join(model_dir, 'predict_model.pk1'), 'rb'))
except FileNotFoundError as e:
    st.error(f"Model file not found: {e}. Please ensure scaler.pk1, pca1.pk1, and predict_model.pk1 are in the MAIN-APP directory.")
    st.stop()
except Exception as e:
    st.error(f"Error loading models: {e}")
    st.stop()


def predict_churn(features):
    """Apply the full prediction pipeline: scale -> PCA -> predict."""
    scaled = scaler.transform([features])
    transformed = pca.transform(scaled)
    prediction = model.predict(transformed)
    return prediction


def main():
    st.title('Customer Churn Prediction')
    st.write("Enter the customer's telecom usage information below to predict if they will churn.")

    col1, col2, col3 = st.columns(3)

    # Cross-operator call features
    with col1:
        loc_og_t2o_mou = st.number_input("Local Outgoing Calls to Other Operator (MOU)", min_value=0.0, value=0.0, key='input_1')
    with col2:
        std_og_t2o_mou = st.number_input("STD Outgoing Calls to Other Operator (MOU)", min_value=0.0, value=0.0, key='input_2')
    with col3:
        loc_ic_t2o_mou = st.number_input("Local Incoming Calls from Other Operator (MOU)", min_value=0.0, value=0.0, key='input_3')

    # ARPU fields
    st.subheader("Average Revenue Per User (ARPU)")
    with col1:
        arpu_6 = st.number_input("ARPU - 6th Month", min_value=0.0, value=0.0, key='input_4')
    with col2:
        arpu_7 = st.number_input("ARPU - 7th Month", min_value=0.0, value=0.0, key='input_5')
    with col3:
        arpu_8 = st.number_input("ARPU - 8th Month", min_value=0.0, value=0.0, key='input_6')

    # On-Net MOU
    st.subheader("On-Net Minutes of Usage (MOU)")
    with col1:
        onnet_mou_6 = st.number_input("On-Net (6th Month)", min_value=0.0, value=0.0, key='input_7')
    with col2:
        onnet_mou_7 = st.number_input("On-Net (7th Month)", min_value=0.0, value=0.0, key='input_8')
    with col3:
        onnet_mou_8 = st.number_input("On-Net (8th Month)", min_value=0.0, value=0.0, key='input_9')

    # Off-Net MOU
    st.subheader("Off-Net Minutes of Usage (MOU)")
    with col1:
        offnet_mou_6 = st.number_input("Off-Net (6th Month)", min_value=0.0, value=0.0, key='input_10')
    with col2:
        offnet_mou_7 = st.number_input("Off-Net (7th Month)", min_value=0.0, value=0.0, key='input_11')
    with col3:
        offnet_mou_8 = st.number_input("Off-Net (8th Month)", min_value=0.0, value=0.0, key='input_12')

    # Roaming Incoming
    st.subheader("Roaming Incoming Calls (MOU)")
    with col1:
        roam_ic_mou_6 = st.number_input("Roaming Incoming (6th Month)", min_value=0.0, value=0.0, key='input_13')
    with col2:
        roam_ic_mou_7 = st.number_input("Roaming Incoming (7th Month)", min_value=0.0, value=0.0, key='input_14')
    with col3:
        roam_ic_mou_8 = st.number_input("Roaming Incoming (8th Month)", min_value=0.0, value=0.0, key='input_15')

    # Roaming Outgoing
    st.subheader("Roaming Outgoing Calls (MOU)")
    with col1:
        roam_og_mou_6 = st.number_input("Roaming Outgoing (6th Month)", min_value=0.0, value=0.0, key='input_16')
    with col2:
        roam_og_mou_7 = st.number_input("Roaming Outgoing (7th Month)", min_value=0.0, value=0.0, key='input_17')
    with col3:
        roam_og_mou_8 = st.number_input("Roaming Outgoing (8th Month)", min_value=0.0, value=0.0, key='input_18')

    # Local Outgoing to Same Operator
    st.subheader("Local Outgoing Calls to Same Operator (MOU)")
    with col1:
        loc_og_t2t_mou_6 = st.number_input("Same Operator (6th Month)", min_value=0.0, value=0.0, key='input_19')
    with col2:
        loc_og_t2t_mou_7 = st.number_input("Same Operator (7th Month)", min_value=0.0, value=0.0, key='input_20')
    with col3:
        loc_og_t2t_mou_8 = st.number_input("Same Operator (8th Month)", min_value=0.0, value=0.0, key='input_21')

    # Local Outgoing to Mobile
    st.subheader("Local Outgoing Calls to Mobile (MOU)")
    with col1:
        loc_og_t2m_mou_6 = st.number_input("To Mobile (6th Month)", min_value=0.0, value=0.0, key='input_22')
    with col2:
        loc_og_t2m_mou_7 = st.number_input("To Mobile (7th Month)", min_value=0.0, value=0.0, key='input_23')
    with col3:
        loc_og_t2m_mou_8 = st.number_input("To Mobile (8th Month)", min_value=0.0, value=0.0, key='input_24')

    # Predict button
    if st.button("Predict whether the Customer will churn or not"):
        features = [
            loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou,
            arpu_6, arpu_7, arpu_8,
            onnet_mou_6, onnet_mou_7, onnet_mou_8,
            offnet_mou_6, offnet_mou_7, offnet_mou_8,
            roam_ic_mou_6, roam_ic_mou_7, roam_ic_mou_8,
            roam_og_mou_6, roam_og_mou_7, roam_og_mou_8,
            loc_og_t2t_mou_6, loc_og_t2t_mou_7, loc_og_t2t_mou_8,
            loc_og_t2m_mou_6, loc_og_t2m_mou_7, loc_og_t2m_mou_8
        ]

        try:
            prediction = predict_churn(features)
            if prediction > 0.5:
                st.subheader("Prediction: 🏃 This customer is likely to churn.")
            else:
                st.subheader("Prediction: 💰 This customer is unlikely to churn.")
        except Exception as e:
            st.error(f"Prediction failed: {e}")


if __name__ == '__main__':
    main()
