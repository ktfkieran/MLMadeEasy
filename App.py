import streamlit as st 
import pandas as pd
import ProjectCopy
import time
import streamlit as st
from PIL import Image

if "button_clicked" not in st.session_state:
    st.session_state.button_clicked = False

def callback():
    st.session_state.button_clicked = True


st.set_page_config(page_title="ML for Idiots",
page_icon="bar_chart:",
layout="wide"
)

st.title("Machine Learning for Idiots")
st.subheader('*made by idiots')


with st.container():
    st.write("Want to find what models are best for your machine learning project? Simply upload your CSV file below to get started!​")
    st.write("(Please note: For the best outcome, have all column headers on row one and the data formatted as a table)​")
    user_file = st.file_uploader("Upload CSV file.",type=['csv'])
    if user_file is not None:
        dataframe = pd.read_csv(user_file)
        if st.button("Click to preview uploaded file."):
            st.dataframe(dataframe,1000,400)
        column1, column2, column3 = st.columns(3)
        with column1:
            y_var = st.selectbox("Select which column you would like to predict:", options = dataframe.columns)
        with column2:    
            x_var = st.multiselect("Select which columns you would like to use to predict the above:", options = dataframe.columns[ ~dataframe.columns.isin([y_var])])
        with column3:
            modelType =  st.radio("Select which model type you would like:",["Regression", "Classification"],)   
        if (y_var is not None) and (x_var is not None):
            if st.button("Make the models", on_click=callback or st.session_state.button_clicked):
                with st.spinner('Wait for it...'):
                    Model1Name,Model1,Model1Score, Model2Name, Model2, Model2Score, Model3Name, Model3, Model3Score=ProjectCopy.Backend(dataframe, modelType.lower(), x_var, y_var)
                    time.sleep(5)
                st.success('Done!')                
                column4, column5, column6 = st.columns(3)
                descrip = {'Regression':'Negative mean squared error measures the average squared difference between the observed and predicted value.','Classification':'F1 score is the mean of the precision and recall scores.','Random Forest Classifier':'Random Forest classification is a method that predicts off multiple decision trees and then takes the mode of these predictions.','Logistic Regression':'Logistic regression is a classification model that uses observations within a dataset to predict a binary output.','KNN Classifier':'KNN Classification is a method that takes the closest n neighbours and classifies it as the majority of its neighbours','Random Forest regressor':'Random Forest regression is a method that predicts off multiple decision trees and then takes the mean of these predictions.','Linear regression':'Linear Regression is a method that finds the relationship between two variables via a line of best fit, which can then be used to predict unknown data.','KNN Regression':'KNN regression is a method that takes the closest n neighbours and creates a new data point based on the mean value of the collective n neighbours.'}
                with column4:
                    st.write(Model2Name)
                    st.write(Model2Score)
                    st.image(Image.open('second.png'))
                    st.write(descrip[Model2Name])
        
                    
                with column5:
                    st.write(Model1Name)
                    st.write(Model1Score)    
                    st.image(Image.open('first.png'))
                    st.write(descrip[Model1Name])
                st.write(descrip[modelType])
                    
                with column6:
                    st.write(Model3Name)
                    st.write(Model3Score)
                    st.image(Image.open('third.png'))
                    st.write(descrip[Model2Name])
                    
