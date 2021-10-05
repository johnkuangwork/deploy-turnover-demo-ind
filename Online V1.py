#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestRegressor,RandomForestClassifier
import shap
# shap.initjs()

import matplotlib as mpl
import matplotlib.pyplot as plt
import seaborn as sns
# get_ipython().run_line_magic('matplotlib', 'inline')

import pickle
import copy
# from joblib import dump, load

import gradio as gr
# import re


# In[2]:


# Load the Model and categorial levels back from file
Filename = "Model/Model_RF.pkl"
with open(Filename, 'rb') as file:  
    model = pickle.load(file)
    
Filename = "Model/cat_data_dict.pkl"
with open(Filename, 'rb') as file:  
    cat_data_dict = pickle.load(file)
    
default_input = pd.read_excel(r'Model/default_input.xlsx',sheet_name='Sheet1')
feature_input = pd.read_excel(r'Model/default_input.xlsx',sheet_name='Sheet2')
x_train = pd.read_excel(r'Model/x_train.xlsx',sheet_name='Sheet1')

feature_list = feature_input.columns.tolist()
explainer = shap.explainers.Tree(model, x_train)
# In[6]:

inputs_list = [
    gr.inputs.Dropdown(choices=cat_data_dict['FUNCTION'], type="value", default=None, label="Sub Job Function"),
    gr.inputs.Slider(minimum=20, maximum=70, step=5, default=30, label="Age"),
    gr.inputs.Slider(minimum=0, maximum=20, step=2, default=5, label="Tenure with company"),
    gr.inputs.Slider(minimum=0, maximum=5000, step=100, default=500, label="Spot Bonus"),
    gr.inputs.Number(default=200, label='Unvested Shares'),
    gr.inputs.Slider(minimum=0, maximum=0.2, step=0.01, default=0.05, label="SALARY_INC_CY_1"),
    
    gr.inputs.Radio(choices=[1,2,3,4,5], type="value", default=3, label="Engagement: Proud to work in company"),
    gr.inputs.Radio(choices=[1,2,3,4,5], type="value", default=3, label="Engagement: Recommend to work in company")
    ]


# In[7]:


def predict_turnover(FUNCTION, AGE, TENURE_COMPANY, SPOT_BONUSES, UNVESTED_LTI_SHARES, 
                     SALARY_INC_CY_1, ENG_PROUD_WRK_CY_1,ENG_RECOM_WRK_CY_1):
    pred_input = copy.deepcopy(default_input)
    pred_input['AGE'] = AGE
    pred_input['TENURE_COMPANY'] = TENURE_COMPANY
    pred_input['SPOT_BONUSES'] =  SPOT_BONUSES
    pred_input['UNVESTED_LTI_SHARES'] =  UNVESTED_LTI_SHARES
    pred_input['SALARY_INC_CY_1'] =  SALARY_INC_CY_1
    pred_input['ENG_PROUD_WRK_CY_1'] = int(ENG_PROUD_WRK_CY_1)
    pred_input['ENG_RECOM_WRK_CY_1'] = int(ENG_RECOM_WRK_CY_1)
    
    FUNCTION_COL = "FUNCTION_"+FUNCTION
    pred_input[FUNCTION_COL] = 1
    
    # Make prediction
    y_score_stay = model.predict_proba(pred_input)[:,0][0]
    y_score_turnover = model.predict_proba(pred_input)[:,1][0]
    pred_dict = {'Stay':y_score_stay, 'Leave': y_score_turnover}
#     pred_dict = {'Leave': y_score_turnover}
#     , pred_input
    
    # Explain with SHAP
    plt.clf()
    choosen_instance = pred_input.iloc[0]
    shap_values_instance = explainer.shap_values(choosen_instance)
    shap.waterfall_plot(shap.Explanation(values=shap_values_instance[1], 
                                             base_values=explainer.expected_value[1], 
                                             data=pred_input.iloc[0],feature_names = feature_list),max_display=6,show=False)
    plt.tight_layout()
    
    # Download of an excel file for the prediction
    pred_input.to_excel(r'Model/pred_input.xlsx')
#     with ZipFile('Model/tmp.zip', 'wb') as zipObj:
#         zipObj.write('Model/pred_input.xlsx', "pred_input.xlsx")

    return pred_dict, plt, 'Model/pred_input.xlsx'


# In[8]:


output_result = gr.outputs.Label(num_top_classes=2, label = 'Probability of Leaving in the next 12 months')
output_img = gr.outputs.Image(type="auto", labeled_segments=False, label="Top 5 reasons drive turnover")
output_file = gr.outputs.File(label="Download input")

outputs_list = [output_result, output_img, output_file]
# outputs_list = [output_result]


# In[9]:


iface = gr.Interface(
    fn = predict_turnover, 
    inputs = inputs_list,
    outputs = outputs_list,
    live = True,
    theme = "default",
    interpretation=None,
    title="Predict Employee Turnover",
    description="Enter employee information",
    flagging_options=["Correct", "Wrong", "Not sure"]
)


# In[10]:

iface.launch()


