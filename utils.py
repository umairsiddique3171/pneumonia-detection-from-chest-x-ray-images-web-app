import base64
import cv2
import json
import pickle
import numpy as np
import streamlit as st


def detect(img,model,class_names):

    img_resized = cv2.resize(img, (224,224))
    img_gray = cv2.cvtColor(img_resized,cv2.COLOR_BGR2GRAY)
    gaus = cv2.GaussianBlur(img_gray,(5,5),0)
    clahe = cv2.createCLAHE(clipLimit = 3)
    final_img = clahe.apply(gaus)
    flattened_img = np.reshape(final_img, (1,50176))
    flat_data = np.array(flattened_img)
    prediction = model.predict(flat_data)[0]
    confidence_score_array = model.predict_proba(flat_data)[0]
    class_name = class_names[str(prediction)]
    confidence_score = f"{confidence_score_array[prediction] * 100} %"
    results = {}
    results['detection'] = class_name
    results['score'] = confidence_score
    
    return results


def load_class_names(file_path):
    with open(file_path, 'r') as json_file:
        class_names = json.load(json_file)
    return class_names


def load_model(model_path): 
    model = pickle.load(open(model_path, "rb"))
    return model


def set_background(img_file):

    with open(img_file, "rb") as f:
        img_data = f.read()
    b64_encoded = base64.b64encode(img_data).decode()
    style = f"""
        <style>
        .stApp {{
            background-image: url(data:image/png;base64,{b64_encoded});
            background-size: cover;
        }}
        </style>
    """
    st.markdown(style, unsafe_allow_html=True)