import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import pickle
from sklearn.neighbors import NearestNeighbors
import cv2


filename = pickle.load(open('filename.pkl','rb'))
featurelist = np.array(pickle.load(open('featurelist.pkl','rb')))

#print(len(filename))
#print(featurelist.shape)

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = Sequential([model,GlobalMaxPooling2D()])

img = image.load_img("sample/shoe.jpg", target_size=(224, 224))
img_arr = image.array_to_img(img)
expand_arr = np.expand_dims(img_arr, axis=0)
process_img = preprocess_input(expand_arr)
result = model.predict(process_img).flatten()
norm_result = result / norm(result)

print(norm_result)

neighbors = NearestNeighbors(n_neighbors=5,algorithm='brute',metric='euclidean')
neighbors.fit(featurelist)

distance,indices = neighbors.kneighbors([norm_result])

for file in indices[0][1:6]:
    temp_img = cv2.imread(filename[file])
    cv2.imshow('output',cv2.resize(temp_img,(512,512)))
    cv2.waitKey(0)