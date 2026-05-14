import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.title("Handwriten Digit Classifier")
st.write('Upload an image of a handwriten digit')

model = tf.keras.models.load_model('model/mnist_cnn.h5')
uploaded = st.file_uploader("Choose an image", type=['jpg','jpeg', 'png'])

if uploaded:
    image = Image.open(uploaded).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))

    st.image(image, caption='Your image', width= 150)

    image_array = np.array(image)/255.0
    image_array = image_array.reshape(1,28, 28, 1)

    prediction = model.predict(image_array)
    digit = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 1)

    st.success(f"predicted_digit: {digit}")
    st.write(f"confidence: {confidence}%")