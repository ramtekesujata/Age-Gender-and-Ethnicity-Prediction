import streamlit as st
import pickle
from PIL import Image
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import numpy as np 
import pandas as pd
import tensorflow as tf
from tensorflow import keras
import cv2

# model = pickle.load(open('dfdbfj.pkl','rb'))

def predict_age(file):
    model = keras.models.load_model("age.h5")
    img = cv2.imread(file)
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    new_array = cv2.resize(gray_image,(48,48))
    X=np.array(new_array)
    X = X.reshape(1,48,48,1)
    X=X/255.0
    prediction = model.predict(X)
    return prediction[0][0]

def main():
    st.title("Gender Prediction")
    html_temp = """
    <body style="background-color:red;">
    <div style="background-color:blue ;padding:10px">
    <h2 style="color:white;text-align;center;">Gender Recongnition WebApp</h2>
    </div>
    </body>
    """
    html_women = """
    <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black ;text-align:center;"> WOMEN </h2>
        </div>
    """
    html_men = """
    <div style="background-color:#F08080;padding:10px >
        <h2 style="color:black ;text-align:center;"> MEN </h2>
        </div>
    """

    st.markdown(html_temp, unsafe_allow_html=True)

    image_file = st.file_uploader("Upload Image", type=['jpg', 'png', 'jpeg'])
    if image_file is not None:
        our_image = Image.open(image_file)
        st.text("Original Image")
        st.image(our_image)
        print()

    if st.button("Recognise"):
        result_img=predict_age(image_file.name)
        
        if result_img > 0.5:
            st.markdown(html_women, unsafe_allow_html=True)
        else:
            st.markdown(html_men, unsafe_allow_html=True)

if __name__ == '__main__':
        main()



