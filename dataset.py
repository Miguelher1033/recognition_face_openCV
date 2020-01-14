import cv2

camera = cv2.VideoCapture(0)

Path = "D:/Apps/Recognition_Face_OpenCV/Cascades/haarcascade_frontalface_default.xml"
faceCascade = cv2.CascadeClassifier(Path)

count = 0

while(True):
    _, imagen = camera.read()

    grises = cv2.cvtColor(imagen, cv2.COLOR_BGR2GRAY)
    imageEqualize = cv2.equalizeHist(grises)
    face = faceCascade.detectMultiScale(imagen, 1.5, 5)
    
    for(x,y,w,h) in face:
        cv2.rectangle(imagen, (x,y), (x+w, y+h), (255,0,0), 4)
        pathImage ="D:/Apps/Recognition_Face_OpenCV/images/Miguel/Miguel_"+str(count)+".jpg"
        cv2.imwrite(pathImage, imageEqualize[y:y+h, x:x+w])
        cv2.imshow("Create Dataset", imagen)
        count += 1


    if cv2.waitKey(1) & 0xFF == ord('e'):
        break
    
    elif count >= 300:
        break

camera.release()
cv2.destroyAllWindows()