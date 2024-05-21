import streamlit as st
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import PCA
from sklearn.decomposition import IncrementalPCA
from sklearn.linear_model import LogisticRegression
from sklearn import metrics
from sklearn.metrics import confusion_matrix
from sklearn.model_selection import KFold
from sklearn.model_selection import cross_val_score
from sklearn.model_selection import GridSearchCV
import pickle
import numpy
import os
st.title("TELECOM CHURN ANALYSIS")
# load pre-trained model
loaded_model1 = pickle.load(open('scaler.pk1' , 'rb'))
loaded_model2 = pickle.load(open('pca1.pk1' , 'rb'))
loaded_model3 = pickle.load(open('final_model1.pk1','rb'))

col1,col2,col3=st.columns(3)
with col1:
  loc_og_t2o_mou=st.number_input("Minutes of Usage of Local Outgoing Calls to Other Operator")
with col2:
  std_og_t2o_mou=st.number_input("Minutes of Usage of STD Outgoing Calls to Other Operator")
with col3:
  loc_ic_t2o_mou=st.number_input("Minutes of Usage of Local Incoming Calls from Other Operator")

with col1:
  arpu_6=st.number_input("Average revenue per unit in 6th month")
with col2:
  arpu_7=st.number_input("Average revenue per unit in 7th month")
with col3:
  arpu_8=st.number_input("Average revenue per unit in 8th month")

with col1:
  onnet_mou_6=st.number_input("Minites of usage of all kind of call in same network month of 6_")
with col2:
  onnet_mou_7=st.number_input("Minites of usage of all kind of call in same network month of 7_")
with col3:
  onnet_mou_8=st.number_input("Minites of usage of all kind of call in same network month of 8_")
with col1:
  offnet_mou_6=st.number_input("Minites of usage of all kind of call in other network month of 6")
with col2:
  offnet_mou_7=st.number_input("Minites of usage of all kind of call in other network month of 7")
with col3:
  offnet_mou_8=st.number_input("Minites of usage of all kind of call in other network month of 8")
with col1:
  roam_ic_mou_6=st.number_input("Minites of usage of Roaming incoming call month of 6")
with col2:
  roam_ic_mou_7=st.number_input("Minites of usage of Roaming incoming call month of 7")
with col3:
  roam_ic_mou_8=st.number_input("Minites of usage of Roaming incoming call month of 8")
with col1:
  roam_og_mou_6=st.number_input("Minites of usage of Roaming outgoing call month of 6")
with col2:
  roam_og_mou_7=st.number_input("Minites of usage of Roaming outgoing call month of 7")
with col3:
  roam_og_mou_8=st.number_input("Minites of usage of Roaming outgoing call month of 8")
with col1:
  loc_og_t2t_mou_6=st.number_input("Minites of usage of Local outgoing calls within same operator month of 6")
with col2:
  loc_og_t2t_mou_7=st.number_input("Minites of usage of Local outgoing calls within same operator month of 7")
with col3:
  loc_og_t2t_mou_8=st.number_input("Minites of usage of Local outgoing calls within same operator month of 8")
with col1:
  loc_og_t2m_mou_6=st.number_input("Minites of usage of local outgoing calls to mobile in month of 6 ")
with col2: 
  loc_og_t2m_mou_7=st.number_input("Minites of usage of local outgoing calls to mobile in month of 7 ")
with col3: 
  loc_og_t2m_mou_8=st.number_input("Minites of usage of local outgoing calls to mobile in month of 8 ")
X=[[loc_og_t2o_mou,std_og_t2o_mou,loc_ic_t2o_mou,arpu_6,arpu_7,arpu_8,onnet_mou_6,onnet_mou_7,onnet_mou_8,offnet_mou_6,offnet_mou_7,offnet_mou_8,roam_ic_mou_6,roam_ic_mou_7,roam_ic_mou_8,roam_og_mou_6,roam_og_mou_7,roam_og_mou_8,loc_og_t2t_mou_6,loc_og_t2t_mou_7,loc_og_t2t_mou_8,loc_og_t2m_mou_6,loc_og_t2m_mou_7,loc_og_t2m_mou_8]]
if st.button("Click here to Predict"):
  result1=loaded_model1.transform(X)
  result2=loaded_model2.transform(result1)
  result=loaded_model3.predict(result2)
  if result>0.5:
    st.title("This person is churn")
  else:
    st.title("This person is not churn")
