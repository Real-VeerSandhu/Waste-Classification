from numpy.lib.npyio import load
import streamlit as st
import tensorflow as tf
from PIL import Image
import numpy as np

@st.cache
def fetch_model():
    return tf.keras.models.load_model('E:/Development/Waste-Classification/models/resnet50_gar0.h5')

loaded_model = fetch_model()

def pred_img(x):
    labels = {0: 'cardboard', 1: 'glass', 2: 'metal', 3: 'paper', 4: 'plastic', 5: 'trash'}
    img1 = Image.open(x).convert(mode="RGB")
    img1 = img1.resize((256,256))
    array1 = np.array(img1.getdata())
    img_np_array = np.reshape(array1, (256,256,3))

    a = np.expand_dims(img_np_array, axis=0)
    return labels[np.argmax(loaded_model.predict(a))]

st.title('Waste Classification')
st.write('Use a neural network to classify waste as glass, paper, plastic, metal, or trash')
st.markdown('----')

col1, col2 = st.beta_columns([2,1])

with col1:
    raw_image = st.file_uploader('Upload an Image')

    if raw_image:
        st.image(raw_image)

with col2:
    st.write('Make a Prediction')
    if st.button('Run Model') and raw_image:
        st.write(f'`Prediction: ` {pred_img(raw_image)}')