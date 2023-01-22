import tensorflow
from tensorflow.keras.preprocessing import image
from tensorflow.keras.layers import GlobalMaxPooling2D
from tensorflow.keras.models import Sequential
from tensorflow.keras.applications.resnet import ResNet50,preprocess_input
import numpy as np
from numpy.linalg import norm
import os
from tqdm import tqdm
import pickle

model = ResNet50(weights='imagenet',include_top=False,input_shape=(224,224,3))
model.trainable=False

model = Sequential([model,GlobalMaxPooling2D()])

def extract_feature(img_path,model):
    img = image.load_img(img_path,target_size=(224,224))
    img_arr = image.array_to_img(img)
    expand_arr = np.expand_dims(img_arr,axis=0)
    process_img = preprocess_input(expand_arr)
    result = model.predict(process_img).flatten()
    norm_result = result/norm(result)
    return norm_result

filename=[]

for file in os.listdir('images'):
    filename.append(os.path.join('images',file))
print(len(filename))

feature_list=[]
for file in tqdm(filename):
    feature_list.append(extract_feature(file,model))

print(np.array(feature_list).shape)

pickle.dump(feature_list,open('featurelist.pkl','wb'))
pickle.dump(filename,open('filename.pkl','wb'))