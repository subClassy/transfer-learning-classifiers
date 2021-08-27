import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.python.distribute.distribute_utils import value_container

my_path = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(my_path, "./monkey_breed_mobileNet.h5")
validation_path = os.path.join(my_path, "./monkey_breed/validation")

model = load_model(model_dir)

def getRandomImage(path):
    folders = list(filter(lambda x: os.path.isdir(os.path.join(path, x)), os.listdir(path)))
    random_dir = np.random.randint(0, len(folders))
    path_class = folders[random_dir]
    file_path = path + "/" + path_class
    file_names = [f for f in os.listdir(file_path) if os.path.isfile(os.path.join(file_path, f))]
    random_file_index = np.random.randint(0,len(file_names))
    image_name = file_names[random_file_index]
    return cv2.imread(file_path + "/" + image_name) , path_class   

for i in range(0, 10):
    input_im, input_class = getRandomImage(validation_path)
    input_original = input_im.copy()
    input_original = cv2.resize(input_original, None, fx=0.5, fy=0.5, interpolation = cv2.INTER_LINEAR)
    
    input_im = cv2.resize(input_im, (224, 224), interpolation = cv2.INTER_LINEAR)
    input_im = input_im / 255.
    input_im = input_im.reshape(1,224,224,3) 
    
    res = np.argmax(model.predict(input_im, 1, verbose = 0), axis=1)
    print(res, input_class)
    
