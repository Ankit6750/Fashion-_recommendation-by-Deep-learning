import os
import streamlit as st
from PIL import Image
import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors

"https://www.kaggle.com/paramaggarwal/fashion-product-images-dataset"

st.title('Fashion Recommendation')

filename = pickle.load(open('filename.pkl','rb'))
featurelist = np.array(pickle.load(open('featurelist.pkl','rb')))

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False
model = Sequential([model,GlobalMaxPooling2D()])

def save_upload(upload_file):
    try:
        with open(os.path.join('uploads',upload_file.name),'wb') as f:
            f.write(upload_file.getbuffer())
        return 1
    except:
        return 0

def feature_extrc(img_path,model):
    img = image.load_img(img_path, target_size=(224, 224))
    img_arr = image.array_to_img(img)
    expand_arr = np.expand_dims(img_arr, axis=0)
    process_img = preprocess_input(expand_arr)
    result = model.predict(process_img).flatten()
    norm_result = result / norm(result)
    return  norm_result

def recommend(features,featurelist):
    neighbors = NearestNeighbors(n_neighbors=5, algorithm='brute', metric='euclidean')
    neighbors.fit(featurelist)

    distance, indices = neighbors.kneighbors([features])

    return indices

# File upload
upload_file = st.file_uploader("Choose image")
if upload_file is not None:
    if save_upload(upload_file): #file uploaded
        display_img = Image.open((upload_file))
        # didplay file
        st.image(display_img)
        # feature extract
        features = feature_extrc(os.path.join('uploads',upload_file.name),model)
        #st.text(features)
        #recommend
        indices = recommend(features,featurelist)
        # show
        col1,col2,col3,col4,col5 = st.columns(5)
        with col1:
            st.image(filename[indices[0][0]])
        with col2:
            st.image(filename[indices[0][1]])
        with col3:
            st.image(filename[indices[0][2]])
        with col4:
            st.image(filename[indices[0][3]])
        with col5:
            st.image(filename[indices[0][4]])

    else:
        st.header('Some upload error')