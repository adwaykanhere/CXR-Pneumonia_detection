import streamlit as st
import matplotlib.pyplot as plt
import pandas as pd
from PIL import Image
from make_inf import model_inference, grad_cam
import os
import gc
import time
import tensorflow as tf
tf.compat.v1.disable_eager_execution()

st.set_option('deprecation.showfileUploaderEncoding', False)
@st.cache(allow_output_mutation=True)
def loading_model():
  model = "pretrained_model.h5"
  return model


gc.enable()

st.sidebar.title("Deep learning based medical image diagnosis")

st.title("Welcome! This page is about Chest X-Ray Abnormalities Classification using Machine learning")
st.sidebar.subheader("Classify between the following types of abnormalities:")

st.write("This application can be used to predict the occurence of one or more medical abnormalities from radiographic chest X-ray images as listed for the conditions described on the left.")

st.write("The machine learning model being used here is based on a pretrained model developed on [this Chest X-Ray dataset](https://nihcc.app.box.com/v/ChestXray-NIHCC); the model is pretrained and is based from [here](https://www.coursera.org/learn/ai-for-medical-diagnosis?)")

st.write("**For a step-by-step approach to training and testing this framework, check out this** [Google Colab notebook](https://colab.research.google.com/drive/1o1tYdm80PqJNyT3F9aiueqb0bc0SU9ek?usp=sharing)")

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
st.sidebar.write("***")
st.sidebar.subheader("About the model performance:")
st.sidebar.write("*The deep learning model that is deployed is based on the **[DENSENET-121](https://www.kaggle.com/pytorch/densenet121)** which is modified later. The AU-ROC curve of the model is shown:*")
st.sidebar.image('roc_cxr.png')
st.sidebar.write("*Refer to the Colab notebook for more details*")

image_path_1 = '00022803_000.png'
# image_path_2 = ''
# image_path_3 = ''


st.markdown('***')

st.write('**Select a Demo image**')
menu = ['Select an Image','Image 1']
choice = st.selectbox('Select an image', menu)

if choice == 'Image 1':
    img = Image.open(image_path_1)

# elif choice == 'Image 2':  ## Work out correct paths
#     img = image_path_2
# elif choice == 'Image 3':
#     img = image_path_3


st.subheader("**Upload your Image:**")
st.markdown("**Please upload your images in png format only**")
img = st.file_uploader("")

if img:
    st.markdown("Your image has been successfully uploaded!", unsafe_allow_html = True)
    cximg = Image.open(img)
    st.image(cximg, caption = "Uploaded Chest X-ray", use_column_width = True)
    st.write("")    

    with st.spinner('Working on your image...'):
        time.sleep(10)
    

    output, image_out = model_inference(cximg,'pretrained_model.h5')
    output1 = output.ravel().tolist()
    maxvalue = max(output1)
    index = output1.index(maxvalue)
    perc = int(maxvalue * 100)
    
    df = pd.DataFrame(output,columns = abnorms)
    
    st.success("*Prediction:* **{}** *with an approx probability of* **{}%**".format(abnorms[index],perc))
    st.markdown("***")
    st.subheader("Computing prediction probabilites for all abnormalities")        
    st.table(df.T.style.highlight_max(axis=0))

    with st.spinner('Computing class activation mapping.......'):
        time.sleep(15)

    
    st.write("")

    camimage = grad_cam(cximg,index)
    plt.title("Viewing CNN Class Activation mappings")
    fig,ax = plt.subplots()
    plt.axis('off')
    ax.imshow(image_out,cmap='gray')
    ax.imshow(camimage, cmap='jet',alpha=min(0.45,maxvalue))

    st.pyplot(fig, use_column_width=True)
    with st.expander("See explanation"):
        st.write("The above representation is a Grad-CAM heatmap juxtaposed on the input image. CAM or Class activation mapping, is an ML interpretability tool that shows the regions where the model is *looking at*. This is done by extracting the gradients from the prefinal convolutional layer of the predicted output class. For more details check out the [GradCAM paper](https://arxiv.org/abs/1610.02391)")   
    
    del img, output, image_out,camimage,output1,maxvalue,index,perc,df
    gc.collect()


