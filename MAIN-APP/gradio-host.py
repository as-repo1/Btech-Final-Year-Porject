import gradio as gr
import pickle
import os

# Load pre-trained models
try:
    script_dir = os.path.dirname(os.path.abspath(__file__))
    loaded_model1 = pickle.load(open(os.path.join(script_dir, 'scaler.pk1'), 'rb'))
    loaded_model2 = pickle.load(open(os.path.join(script_dir, 'pca1.pk1'), 'rb'))
    loaded_model3 = pickle.load(open(os.path.join(script_dir, 'predict_model.pk1'), 'rb'))
except FileNotFoundError as e:
    raise RuntimeError(f"Model file not found: {e}. Please ensure scaler.pk1, pca1.pk1, and predict_model.pk1 are in the MAIN-APP directory.")

# Define the prediction function
def predict_churn(loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou, arpu_6, arpu_7, arpu_8, 
                  onnet_mou_6, onnet_mou_7, onnet_mou_8, offnet_mou_6, offnet_mou_7, offnet_mou_8, 
                  roam_ic_mou_6, roam_ic_mou_7, roam_ic_mou_8, roam_og_mou_6, roam_og_mou_7, 
                  roam_og_mou_8, loc_og_t2t_mou_6, loc_og_t2t_mou_7, loc_og_t2t_mou_8, 
                  loc_og_t2m_mou_6, loc_og_t2m_mou_7, loc_og_t2m_mou_8):
    
    # Collecting inputs into a list
    X = [[loc_og_t2o_mou, std_og_t2o_mou, loc_ic_t2o_mou, arpu_6, arpu_7, arpu_8, onnet_mou_6,
         onnet_mou_7, onnet_mou_8, offnet_mou_6, offnet_mou_7, offnet_mou_8, roam_ic_mou_6, 
         roam_ic_mou_7, roam_ic_mou_8, roam_og_mou_6, roam_og_mou_7, roam_og_mou_8, loc_og_t2t_mou_6,
         loc_og_t2t_mou_7, loc_og_t2t_mou_8, loc_og_t2m_mou_6, loc_og_t2m_mou_7, loc_og_t2m_mou_8]]
    
    try:
        # Transform input data using loaded models
        result1 = loaded_model1.transform(X)
        result2 = loaded_model2.transform(result1)
        prediction = loaded_model3.predict(result2)
        
        # Determine the prediction result
        if prediction > 0.5:
            return "Prediction: 🏃 This customer is likely to churn."
        else:
            return "Prediction: 💰 This customer is unlikely to churn."
    except Exception as e:
        return f"Prediction failed: {e}"

# Create a Gradio interface (using modern Gradio API)
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.Number(label="Local Outgoing Calls to Other Operator (MOU)"),
        gr.Number(label="STD Outgoing Calls to Other Operator (MOU)"),
        gr.Number(label="Local Incoming Calls from Other Operator (MOU)"),
        gr.Number(label="Average Revenue Per Unit in 6th Month"),
        gr.Number(label="Average Revenue Per Unit in 7th Month"),
        gr.Number(label="Average Revenue Per Unit in 8th Month"),
        gr.Number(label="On-Net Calls in 6th Month (MOU)"),
        gr.Number(label="On-Net Calls in 7th Month (MOU)"),
        gr.Number(label="On-Net Calls in 8th Month (MOU)"),
        gr.Number(label="Off-Net Calls in 6th Month (MOU)"),
        gr.Number(label="Off-Net Calls in 7th Month (MOU)"),
        gr.Number(label="Off-Net Calls in 8th Month (MOU)"),
        gr.Number(label="Roaming Incoming Calls in 6th Month"),
        gr.Number(label="Roaming Incoming Calls in 7th Month"),
        gr.Number(label="Roaming Incoming Calls in 8th Month"),
        gr.Number(label="Roaming Outgoing Calls in 6th Month"),
        gr.Number(label="Roaming Outgoing Calls in 7th Month"),
        gr.Number(label="Roaming Outgoing Calls in 8th Month"),
        gr.Number(label="Local Outgoing Calls to Same Operator in 6th Month"),
        gr.Number(label="Local Outgoing Calls to Same Operator in 7th Month"),
        gr.Number(label="Local Outgoing Calls to Same Operator in 8th Month"),
        gr.Number(label="Local Outgoing Calls to Mobile in 6th Month"),
        gr.Number(label="Local Outgoing Calls to Mobile in 7th Month"),
        gr.Number(label="Local Outgoing Calls to Mobile in 8th Month"),
    ],
    outputs=gr.Textbox(label="Prediction Result")
)

# Launch the Gradio app
# iface.launch()
iface.launch(share=True)