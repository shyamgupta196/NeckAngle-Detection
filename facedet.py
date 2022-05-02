import cv2

ear_cascade = cv2.CascadeClassifier('cascade.xml')
cascPath = 'facedet.xml'

faceCascade = cv2.CascadeClassifier(cascPath)


cap = cv2.VideoCapture(0)

while 1:
    ret, img = cap.read()


    gray = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
    
    faces = faceCascade.detectMultiScale(
            gray,
            scaleFactor=2.5,
            minNeighbors=5,
            minSize=(30, 30),
            flags=cv2.CASCADE_SCALE_IMAGE)
    ear = ear_cascade.detectMultiScale(gray, 1.3, 5)


    for (x,y,w,h) in ear:
      cv2.rectangle(gray, (x,y), (x+w,y+h), (255,0,0), 3)
      cv2.rectangle(gray, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('img',gray)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
      break

#average 
#but which frame to consider 
cap.release()
cv2.destroyAllWindows()



