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
    st.title('Churn Prediction App')
    st.sidebar.title('Input Features')

    # Cross-operator call features
    loc_og_t2o_mou = st.sidebar.number_input('Local Outgoing Calls to Other Operator (MOU)', min_value=0.0, max_value=10000.0, step=1.0)
    std_og_t2o_mou = st.sidebar.number_input('STD Outgoing Calls to Other Operator (MOU)', min_value=0.0, max_value=10000.0, step=1.0)
    loc_ic_t2o_mou = st.sidebar.number_input('Local Incoming Calls from Other Operator (MOU)', min_value=0.0, max_value=10000.0, step=1.0)

    # ARPU
    st.sidebar.subheader('ARPU')
    arpu_6 = st.sidebar.number_input('ARPU for month 6', min_value=0.0, max_value=10000.0, step=1.0)
    arpu_7 = st.sidebar.number_input('ARPU for month 7', min_value=0.0, max_value=10000.0, step=1.0)
    arpu_8 = st.sidebar.number_input('ARPU for month 8', min_value=0.0, max_value=10000.0, step=1.0)

    # On-Net MOU
    st.sidebar.subheader('On-Net MOU')
    onnet_mou_6 = st.sidebar.number_input('On-Net (month 6)', min_value=0.0, max_value=10000.0, step=1.0)
    onnet_mou_7 = st.sidebar.number_input('On-Net (month 7)', min_value=0.0, max_value=10000.0, step=1.0)
    onnet_mou_8 = st.sidebar.number_input('On-Net (month 8)', min_value=0.0, max_value=10000.0, step=1.0)

    # Off-Net MOU
    st.sidebar.subheader('Off-Net MOU')
    offnet_mou_6 = st.sidebar.number_input('Off-Net (month 6)', min_value=0.0, max_value=10000.0, step=1.0)
    offnet_mou_7 = st.sidebar.number_input('Off-Net (month 7)', min_value=0.0, max_value=10000.0, step=1.0)
    offnet_mou_8 = st.sidebar.number_input('Off-Net (month 8)', min_value=0.0, max_value=10000.0, step=1.0)

    # Roaming Incoming
    st.sidebar.subheader('Roaming Incoming MOU')
    roam_ic_mou_6 = st.sidebar.number_input('Roaming Incoming (month 6)', min_value=0.0, max_value=10000.0, step=1.0)
    roam_ic_mou_7 = st.sidebar.number_input('Roaming Incoming (month 7)', min_value=0.0, max_value=10000.0, step=1.0)
    roam_ic_mou_8 = st.sidebar.number_input('Roaming Incoming (month 8)', min_value=0.0, max_value=10000.0, step=1.0)

    # Roaming Outgoing
    st.sidebar.subheader('Roaming Outgoing MOU')
    roam_og_mou_6 = st.sidebar.number_input('Roaming Outgoing (month 6)', min_value=0.0, max_value=10000.0, step=1.0)
    roam_og_mou_7 = st.sidebar.number_input('Roaming Outgoing (month 7)', min_value=0.0, max_value=10000.0, step=1.0)
    roam_og_mou_8 = st.sidebar.number_input('Roaming Outgoing (month 8)', min_value=0.0, max_value=10000.0, step=1.0)

    # Local Outgoing to Same Operator
    st.sidebar.subheader('Local Outgoing to Same Operator MOU')
    loc_og_t2t_mou_6 = st.sidebar.number_input('Same Operator (month 6)', min_value=0.0, max_value=10000.0, step=1.0)
    loc_og_t2t_mou_7 = st.sidebar.number_input('Same Operator (month 7)', min_value=0.0, max_value=10000.0, step=1.0)
    loc_og_t2t_mou_8 = st.sidebar.number_input('Same Operator (month 8)', min_value=0.0, max_value=10000.0, step=1.0)

    # Local Outgoing to Mobile
    st.sidebar.subheader('Local Outgoing to Mobile MOU')
    loc_og_t2m_mou_6 = st.sidebar.number_input('To Mobile (month 6)', min_value=0.0, max_value=10000.0, step=1.0)
    loc_og_t2m_mou_7 = st.sidebar.number_input('To Mobile (month 7)', min_value=0.0, max_value=10000.0, step=1.0)
    loc_og_t2m_mou_8 = st.sidebar.number_input('To Mobile (month 8)', min_value=0.0, max_value=10000.0, step=1.0)

    if st.sidebar.button('Predict Churn'):
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