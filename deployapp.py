import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from make_inf import model_inference, grad_cam
import os
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()


st.title("Welcome to Chest X-Ray Abnormalities Classification using Machine learning")
st.sidebar.header("Classify between the following types of abnormalities:")

abnorms = ['Cardiomegaly', 
          'Emphysema', 
          'Effusion', 
          'Hernia', 
          'Infiltration', 
          'Mass', 
          'Nodule', 
          'Atelectasis',
          'Pneumothorax',
          'Pleural_Thickening', 
          'Pneumonia', 
          'Fibrosis', 
          'Edema', 
          'Consolidation']

st.sidebar.dataframe(abnorms)
image_path = '00022803_000.png'

st.text("Please upload your Chest X-Ray images in png format only")

st.write("**Upload your Image**")
img = st.file_uploader("Click here and upload an X-Ray image.")

if img:
    st.markdown("Your image has been successfully uploaded!", unsafe_allow_html = True)
    cximg = Image.open(img)
    st.image(cximg, caption = "Uploaded Chest X-ray", use_column_width = True)
    st.write("")    

    with st.spinner('Working on your image...'):
        time.sleep(10)
    

    output = model_inference(cximg,'pretrained_model.h5')
    output1 = list(output)
    maxvalue = max(output1)
    index = output1.index(maxvalue)

    gradcambutton = st.sidebar.button("Compute CAMs")
    
    df = pd.DataFrame(output,columns = abnorms)
    st.success('Done!')
    st.table(df.T)

    if gradcambutton:
        st.write("")
        
        st.header("Computing prediction probabilites for all abnormalities")        

        camimage = grad_cam(cximg,index)

        fig,ax = plt.subplots()
        plt.axis('off')
        plt.title("Viewing CNN Class Activation mappings")
        plt.imshow(cximg,cmap='gray')
        plt.imshow(camimage, cmap='jet')

        st.pyplot(fig)
        

