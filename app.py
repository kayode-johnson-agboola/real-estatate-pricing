import numpy as np 
import pandas as pd 
import joblib
import matplotlib.pyplot as plt
from matplotlib.ticker import StrMethodFormatter
import streamlit as st
from PIL import Image


# Load  model a 
model = joblib.load(open("model-v1.joblib","rb"))

def data_preprocessor(df_US_States_clean):
    """this function preprocess the user input
        return type: pandas dataframe
    """
    df_US_States_clean["state"] = df_US_States_clean["state"].map({'Massachusetts':0, 'New Jersey':1, 'Connecticut':2, 'New York':3, 'Rhode Island':4, 'New Hampshire':5, 'Maine':6, 'Vermont':7, 'Pennsylvania':8, 'Delaware':9,'Virgin Islands':10,'Wyoming':11, 'West Virginia':12})
    return df_US_States_clean

def visualize_confidence_level(prediction_proba):
    """
    this function uses matplotlib to create inference bar chart rendered with streamlit in real-time 
    return type : matplotlib bar chart  
    """
    data = (prediction_proba[0]*100).round(2)
    grad_percentage = pd.DataFrame(data = data,columns = ['Percentage'],index = ['Low','Ave','High'])
    ax = grad_percentage.plot(kind='barh', figsize=(7, 4), color='#722f37', zorder=10, width=0.5)
    ax.legend().set_visible(False)
    ax.set_xlim(xmin=0, xmax=100)
    
    ax.spines['right'].set_visible(False)
    ax.spines['top'].set_visible(False)
    ax.spines['left'].set_visible(True)
    ax.spines['bottom'].set_visible(True)

    ax.tick_params(axis="both", which="both", bottom="off", top="off", labelbottom="on", left="off", right="off", labelleft="on")
    
    vals = ax.get_xticks()
    for tick in vals:
        ax.axvline(x=tick, linestyle='dashed', alpha=0.4, color='#eeeeee', zorder=1)

    ax.set_xlabel(" Percentage(%) Confidence Level", labelpad=2, weight='bold', size=12)
    ax.set_ylabel("US State_clean", labelpad=10, weight='bold', size=12)
    ax.set_title('Prediction Confidence Level ', fontdict=None, loc='center', pad=None, weight='bold')

    st.pyplot()
    return

st.write("""
# House Price Prediction ML Web-App 
This app predicts the ** Price of Housing **  using **house features** input via the **side panel** 
""")

#read in wine image and render with streamlit
image = Image.open('house.jpg')
st.image(image, caption='House',use_column_width=True)

st.sidebar.header('User Input Parameters') #user input parameter collection with streamlit side bar


def get_user_input():
    """
    this function is used to get user input using sidebar slider and selectbox 
    return type : pandas dataframe

    """
    state = st.sidebar.selectbox("Select US State",("Massachusetts", "New Jersey", "Connecticut", "New York", "Rhode Island", "New Hampshire", "Vermont", "Maine", "Pennsylvania", "Delaware", "Virgin Islands", "Wyoming"))
    bed = st.sidebar.slider('bed', 3.8, 15.9, 7.0)
    bath = st.sidebar.slider('bath', 0.08, 1.58, 0.4)
    acre_lot  = st.sidebar.slider('acre lot', 0.0, 1.66, 0.3)
    house_size  = st.sidebar.slider('house size', 0.009, 0.611, 0.211)
    
    features = {'state': state,
            'bed': bed,
            'bath': bath,
            'acre_lot': acre_lot,
            'state': state,
            'house_size': house_size,
            }
    data = pd.DataFrame(features,index=[0])

    return data

user_input_df = get_user_input()
processed_user_input = data_preprocessor(user_input_df)

st.subheader('User Input parameters')
st.write(user_input_df)

prediction = model.predict(processed_user_input)
prediction_proba = model.predict_proba(processed_user_input)

visualize_confidence_level(prediction_proba)