import cv2
import pickle

path = "D:/Recognition_Face_OpenCV/Cascades/haarcascade_frontalface_alt2.xml"
faceCascade = cv2.CascadeClassifier(path)

eyeCascade = cv2.CascadeClassifier("D:/Recognition_Face_OpenCV/Cascades/haarcascade_eye.xml")
smileCascade = cv2.CascadeClassifier("D:/Recognition_Face_OpenCV/Cascades/haarcascade_smile.xml")

recognition = cv2.face.LBPHFaceRecognizer_create()
recognition.read("train.yml")

labels = {"person_name" : 1 }
with open("labels.pickle",'rb') as f:
    pre_labels = pickle.load(f)
    labels = { v:k for k,v in pre_labels.items()}

camera = cv2.VideoCapture(0)

while True:
    por, border = camera.read()
    imageGray = cv2.cvtColor(border, cv2.COLOR_BGR2GRAY)
    imageEqualize = cv2.equalizeHist(imageGray)    
    faces = faceCascade.detectMultiScale(imageEqualize, 1.2, 5)

    for (x, y, w, h) in faces:
        roi_gray = imageGray[y:y+h, x:x+w]
        roi_color = border[y:y+h, x:x+w]

        id_, conf = recognition.predict(roi_gray)
        if conf >= 4  and conf < 85:
            font = cv2.FONT_HERSHEY_SIMPLEX            
            name = labels[id_]

            if conf > 50:
                name = "undefined"

            color = (255,255,255)
            borderWidth = 2
            cv2.putText(border, name, (x,y), font, 1, color, borderWidth, cv2.LINE_AA)

        imageItem = "my-image.png"
        cv2.imwrite(imageItem, roi_gray)
        
        cv2.rectangle(border, (x, y), (x+w, y+h), (0, 255, 0), 2)

        facialFeatures = smileCascade.detectMultiScale(roi_gray)
        for(ex,ey,ew,eh) in facialFeatures:
            cv2.rectangle(roi_color, (ex, ey), (ex+ew, ey+eh), (0, 255, 0), 2)
    
    
    border_display = cv2.resize(border, (1200, 650), interpolation = cv2.INTER_CUBIC)
    cv2.imshow('Recognition face', border_display)

    if cv2.waitKey(1) & 0xFF == ord('q'):
        break


camera.release()
cv2.destroyAllWindows()