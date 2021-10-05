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
    
# Filename = "Model/Explainer.pkl"  
# with open(Filename, 'rb') as file:  
#     explainer = pickle.load(file)  
# explainer = load("Model/Explainer.joblib")
    
default_input = pd.read_excel(r'Model/default_input.xlsx',sheet_name='Sheet1')
x_train = pd.read_excel(r'Model/x_train.xlsx',sheet_name='Sheet1')
pred_input = copy.deepcopy(default_input)

explainer = shap.explainers.Tree(model, x_train)

# In[6]:


inputs_list = [
    gr.inputs.Slider(minimum=18, maximum=100, step=1, default=40, label="Age"),
    gr.inputs.Slider(minimum=0, maximum=30, step=1, default=5, label="Tenure with company"),
    gr.inputs.Number(default=100000, label='Base Salary'),
    gr.inputs.Slider(minimum=0, maximum=350, step=10, default=150, label="PTO Taken (in Hours)"),
    gr.inputs.Dropdown(choices=cat_data_dict['FUNCTION'], type="value", default=None, label="Sub Job Function")
    ]


# In[7]:


def predict_turnover(AGE, TENURE_COMPANY, SALARY, PTO, FUNCTION):
    pred_input['AGE'] = AGE
    pred_input['TENURE_COMPANY'] = TENURE_COMPANY
    pred_input['SALARY'] =  SALARY
    pred_input['PTO'] =  PTO
    
    FUNCTION_COL = "FUNCTION_"+FUNCTION
    pred_input[FUNCTION_COL] = 1
    
    # Make prediction
    y_score_stay = model.predict_proba(pred_input)[:,0][0]
    y_score_turnover = model.predict_proba(pred_input)[:,1][0]
    pred_dict = {'Stay':y_score_stay, 'Leave': y_score_turnover}
#     , pred_input
    
    # Explain with SHAP
    plt.clf()
    choosen_instance = pred_input.iloc[0]
    shap_values_instance = explainer.shap_values(choosen_instance)
    shap.waterfall_plot(shap.Explanation(values=shap_values_instance[1], 
                                             base_values=explainer.expected_value[1], 
                                             data=pred_input.iloc[0]),max_display=5,show=False)

#     plt.savefig('online_shap.jpg', bbox_inches="tight")
    plt.tight_layout()
    plt.spring()
    
    return pred_dict, plt


# In[8]:


output_result = gr.outputs.Label(num_top_classes=2, label = 'Probability of Leaving in the next 12 months')
# output_input = gr.outputs.Dataframe(headers=None, max_rows=3, max_cols=10, overflow_row_behaviour="paginate", type="auto", label="Inputs")
output_img = gr.outputs.Image(type="auto", labeled_segments=False, label="SHAP")

outputs_list = [output_result, output_img]
# outputs_list = [output_result]


# In[9]:


iface = gr.Interface(
    fn = predict_turnover, 
    inputs = inputs_list,
    outputs = outputs_list,
    live = True,
    theme = "default",
    interpretation=None,
    title="Predict Company Turnover",
    description="Enter employee information",
    flagging_options=["Correct", "Wrong", "Not sure"]
)


# In[10]:

iface.launch()


