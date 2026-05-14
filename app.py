import streamlit as st
import tensorflow as tf
import numpy as np
from PIL import Image, ImageOps

st.title("Handwritten Digit Classifier")
st.write("Upload an image of a handwritten digit (0-9)")

@st.cache_resource
def load_model():
    (x_train, y_train), (x_test, y_test) = tf.keras.datasets.mnist.load_data()
    x_train = x_train.reshape(-1, 28, 28, 1) / 255.0
    x_test  = x_test.reshape(-1, 28, 28, 1) / 255.0
    
    model = tf.keras.models.Sequential([
        tf.keras.layers.Conv2D(32, (3,3), activation='relu', input_shape=(28,28,1)),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Conv2D(64, (3,3), activation='relu'),
        tf.keras.layers.MaxPooling2D(2,2),
        tf.keras.layers.Flatten(),
        tf.keras.layers.Dense(128, activation='relu'),
        tf.keras.layers.Dropout(0.3),
        tf.keras.layers.Dense(10, activation='softmax')
    ])
    
    model.compile(optimizer='adam',
                  loss='sparse_categorical_crossentropy',
                  metrics=['accuracy'])
    
    model.fit(x_train, y_train, epochs=3, verbose=0)
    return model

with st.spinner("Model loading... please wait"):
    model = load_model()

st.success("Model ready!")

uploaded = st.file_uploader("Choose an image", type=["png","jpg","jpeg"])

if uploaded:
    image = Image.open(uploaded).convert('L')
    image = ImageOps.invert(image)
    image = image.resize((28, 28))
    
    st.image(image, caption="Your image", width=150)
    
    img_array = np.array(image) / 255.0
    img_array = img_array.reshape(1, 28, 28, 1)
    
    prediction = model.predict(img_array)
    digit = np.argmax(prediction)
    confidence = round(float(np.max(prediction)) * 100, 1)
    
    st.success(f"Predicted digit: {digit}")
    st.write(f"Confidence: {confidence}%")
