from imutils.video import VideoStream
import time
import cv2
import os


def captureInfo(LabelName):
    
    rootPath = os.getcwd()
    path = rootPath + "/Cascades/haarcascade_frontalface_default.xml"
    faceCascade = cv2.CascadeClassifier(path)

    os.makedirs("images/"+LabelName, exist_ok=True)
    rootimage = rootPath + "/images/"+ LabelName

    camera = VideoStream(src=0).start()           

    time.sleep(2.0)

    count = 0
    while(True):
        imagen = camera.read()
        grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
        imageEqualize = cv2.equalizeHist(grises)
        face = faceCascade.detectMultiScale(imagen, 1.5, 5)

        for(x,y,w,h) in face:
            cv2.rectangle(imagen, (x,y), (x+w, y+h), (255,0,0), 4)
            pathImage = rootimage + "/" + LabelName + "_" + str(count)+".jpg"
            cv2.imwrite(pathImage, imageEqualize[y:y+h, x:x+w])
            cv2.imshow("Create Dataset", imagen)
            count += 1
        if cv2.waitKey(1) & 0xFF == ord('e'):
            break

        elif count >= 300:
            break

    camera.stop()
    cv2.destroyAllWindows()


