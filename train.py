import cv2
import os
import numpy as np
import pickle
import os.path as path
from PIL import Image
from os import remove


fileTrain = 'train.yml'
if path.exists(fileTrain):
    remove(fileTrain)


path = "D:/Recognition_Face_OpenCV/Cascades/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(path)

recognizer = cv2.face.LBPHFaceRecognizer_create()
recognition = cv2.face.LBPHFaceRecognizer_create()

pathDataSet = os.path.dirname(os.path.abspath(__file__))
path_image = os.path.join(pathDataSet,"images")

current_id = 0
label_id = {}
label_y = []
train_x = []

for root, dirs, files in os.walk(path_image):
    for file_image in files:
        if file_image.endswith("png") or file_image.endswith("jpg"):
            pathImagen = os.path.join(root,file_image)
            label = os.path.basename(root).replace(" ", "-")
            
            
            if not label in label_id:                
                label_id[label] = current_id
                current_id += 1            
            id_ = label_id[label]
            
            pil_image = Image.open(pathImagen).convert("L")
            size = (550,550)
            end_image = pil_image.resize(size, Image.ANTIALIAS)
            image_array = np.array(pil_image,"uint8")
            
            faces = faceCascade.detectMultiScale(image_array, 1.2, 5)
            
            for (x,y,w,h) in faces:
                recog = image_array[y:y+h, x:x+w]
                train_x.append(recog)
                label_y.append(id_)

with open("labels.pickle",'wb') as f:
    pickle.dump(label_id, f)

recognition.train(train_x, np.array(label_y))
recognition.save(fileTrain)