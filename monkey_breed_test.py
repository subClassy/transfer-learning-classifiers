import os
import cv2
import numpy as np
from tensorflow.keras.models import load_model
import re

my_path = os.path.abspath(os.path.dirname(__file__))
model_dir = os.path.join(my_path, "./monkey_breed_mobileNet.h5")
validation_path = os.path.join(my_path, "./monkey_breed/validation")
monkey_breeds_dict = {"0": "mantled_howler ", 
                      "1": "patas_monkey",
                      "2": "bald_uakari",
                      "3": "japanese_macaque",
                      "4": "pygmy_marmoset ",
                      "5": "white_headed_capuchin",
                      "6": "silvery_marmoset",
                      "7": "common_squirrel_monkey",
                      "8": "black_headed_night_monkey",
                      "9": "nilgiri_langur"}

model = load_model(model_dir)

def draw_test(input_class, pred_class, im):
    input_class = monkey_breeds_dict[str(re.findall(r'\d+', input_class)[0])]
    pred_class = monkey_breeds_dict[str(re.findall(r'\d+', pred_class)[0])]
    
    BLACK = [0,0,0]
    expanded_image = cv2.copyMakeBorder(im, 80, 0, 0, 100 ,cv2.BORDER_CONSTANT, value=BLACK)
    cv2.putText(expanded_image, input_class, (20, 60) , cv2.FONT_HERSHEY_PLAIN, 1, (0,0,255), 2)
    cv2.putText(expanded_image, pred_class, (20, 80) , cv2.FONT_HERSHEY_PLAIN, 1, (0,255,255), 2)
    cv2.imshow("img", expanded_image)

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
    draw_test(input_class, str(res), input_original)
    cv2.waitKey(0)

cv2.destroyAllWindows()
    
