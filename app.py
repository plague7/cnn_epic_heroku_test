import os
from PIL import Image
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.utils import *
import tensorflow as tf
import streamlit as st
from streamlit_drawable_canvas import st_canvas

### Import model file ###
MODEL_DIR = os.path.join(os.path.dirname('__file__'), 'first_model.h5')
model = tf.keras.models.load_model(MODEL_DIR)

### Functions ###

# @st.cache
def load_data():
    data_test = pd.read_csv(uploaded_file)
    st.header('csv uploaded!')
    return data_test

def viz_num(img):
    #Reshape the 768 values to a 28x28 image
    image = img.reshape([28,28])
    fig = plt.figure(figsize=(1, 1))
    plt.imshow(image, cmap=plt.get_cmap('gray'))
    plt.axis("off")
    #fig.show()
    return fig

def predict_random_picture(csv):
    rand_img = data.sample()
    img_test = rand_img.values.reshape(rand_img.shape[0], 28, 28, 1)
    pred = model.predict(img_test)
    pred = np.argmax(pred, axis=1)
    st.write('Prediction: ', int(pred))
    st.pyplot(viz_num(img_test))

def predict_drawn_image(number):
    image = Image.fromarray((number_drawn[:, :, 0]).astype(np.uint8))
    image = image.resize((28, 28))
    image = image.convert('L')
    image = (tf.keras.utils.img_to_array(image)/255)
    image = image.reshape(1,28,28,1)
    x_2 = tf.convert_to_tensor(image)
    pred = model.predict(x_2)
    pred = np.argmax(pred, axis=1)
    st.write('Predicted number : ' + str(pred))    

### Changing style ###
st.markdown(""" 
    <style> 
    img {
        width:150px !important; height:150px;
    } 
    </style> 
""",
unsafe_allow_html=True)

### Header ###
st.title('Digit Prediction App')
st.header('Which prediction tool to use?')

### Randomizing Tool View ###
if st.selectbox('Tools', ['Randomizing Tool', 'Drawing Tool']) == 'Randomizing Tool':
    ### Upload test dataset ###
    uploaded_file = st.file_uploader("Choose a file")
    if uploaded_file is not None:
        data = load_data()

        if st.button("Predict one picture"):
            predict_random_picture(data)

### Drawing Tool View ###               
else:
    st.write('Draw a number and let the app makes a prediction')

    # Specify canvas parameters in application
    stroke_width = st.sidebar.slider("Stroke width: ", 1, 25, 8)
    stroke_color = st.sidebar.color_picker("Stroke color hex: ", "#fff")
    bg_color = st.sidebar.color_picker("Background color hex: ", "#000")

    # Create a canvas component
    canvas_result = st_canvas(
        fill_color="rgba(255, 165, 0, 0.3)",  # Fixed fill color with some opacity
        stroke_width=stroke_width,
        stroke_color=stroke_color,
        background_color=bg_color,
        height=150,
        width=150,
        key="canvas",
    )

    # Resize drawn image and predict it
    if canvas_result.image_data is not None:
        number_drawn = canvas_result.image_data
        if st.button('Predict your drawn number'):
            predict_drawn_image(number_drawn)