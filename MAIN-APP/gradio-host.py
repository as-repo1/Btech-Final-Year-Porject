import gradio as gr
import pickle
import numpy as np

# Load pre-trained models
loaded_model1 = pickle.load(open('scaler.pk1', 'rb'))
loaded_model2 = pickle.load(open('pca1.pk1', 'rb'))
loaded_model3 = pickle.load(open('final_model1.pk1', 'rb'))

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
    
    # Transform input data using loaded models
    result1 = loaded_model1.transform(X)
    result2 = loaded_model2.transform(result1)
    prediction = loaded_model3.predict(result2)
    
    # Determine the prediction result
    if prediction > 0.5:
        return "Prediction: 🏃 This customer is likely to churn."
    else:
        return "Prediction: 💰 This customer is unlikely to churn."

# Create a Gradio interface
iface = gr.Interface(
    fn=predict_churn,
    inputs=[
        gr.inputs.Number(label="Local Outgoing Calls to Other Operator (MOU)"),
        gr.inputs.Number(label="STD Outgoing Calls to Other Operator (MOU)"),
        gr.inputs.Number(label="Local Incoming Calls from Other Operator (MOU)"),
        gr.inputs.Number(label="Average Revenue Per Unit in 6th Month"),
        gr.inputs.Number(label="Average Revenue Per Unit in 7th Month"),
        gr.inputs.Number(label="Average Revenue Per Unit in 8th Month"),
        gr.inputs.Number(label="On-Net Calls in 6th Month (MOU)"),
        gr.inputs.Number(label="On-Net Calls in 7th Month (MOU)"),
        gr.inputs.Number(label="On-Net Calls in 8th Month (MOU)"),
        gr.inputs.Number(label="Off-Net Calls in 6th Month (MOU)"),
        gr.inputs.Number(label="Off-Net Calls in 7th Month (MOU)"),
        gr.inputs.Number(label="Off-Net Calls in 8th Month (MOU)"),
        gr.inputs.Number(label="Roaming Incoming Calls in 6th Month"),
        gr.inputs.Number(label="Roaming Incoming Calls in 7th Month"),
        gr.inputs.Number(label="Roaming Incoming Calls in 8th Month"),
        gr.inputs.Number(label="Roaming Outgoing Calls in 6th Month"),
        gr.inputs.Number(label="Roaming Outgoing Calls in 7th Month"),
        gr.inputs.Number(label="Roaming Outgoing Calls in 8th Month"),
        gr.inputs.Number(label="Local Outgoing Calls to Same Operator in 6th Month"),
        gr.inputs.Number(label="Local Outgoing Calls to Same Operator in 7th Month"),
        gr.inputs.Number(label="Local Outgoing Calls to Same Operator in 8th Month"),
        gr.inputs.Number(label="Local Outgoing Calls to Mobile in 6th Month"),
        gr.inputs.Number(label="Local Outgoing Calls to Mobile in 7th Month"),
        gr.inputs.Number(label="Local Outgoing Calls to Mobile in 8th Month"),
    ],
    outputs=gr.outputs.Textbox(label="Prediction Result")
)

# Launch the Gradio app
# iface.launch()
iface.launch(share=True)