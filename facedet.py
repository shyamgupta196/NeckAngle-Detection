import cv2
from facenet_pytorch import MTCNN
import torch
import numpy as np
import mmcv
from PIL import Image, ImageDraw
from IPython import display

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

ear_cascade = cv2.CascadeClassifier('cascade.xml')
cascPath = 'facedet.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

mtcnn = MTCNN(keep_all=True, device=device)

cap = cv2.VideoCapture(0)

img = cv2.imread('friends.jpg')

while 1:
    # ret, img = cap.read()

    # frames = [Image.fromarray(cv2.cvtColor(frame, cv2.COLOR_BGR2GRAY)) for frame in img]

    # gray = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    
    # faces = faceCascade.detectMultiScale(
    #         gray,
    #         scaleFactor=2.5,
    #         minNeighbors=5,
    #         minSize=(30, 30),
    #         flags=cv2.CASCADE_SCALE_IMAGE)
    ear = ear_cascade.detectMultiScale(img, 1.3, 5)
    cv2.imshow('img',img)
    cv2.waitKey(300)
    frames_tracked = []
    for i, frame in enumerate(img):
        print('\rTracking frame: {}'.format(i + 1), end='')
        # print(type(boxes))
        # print(boxes[0])
        
        # Detect faces
        boxes, _ = mtcnn(frame)
        # Draw faces
        frame_draw = frame.copy()
        draw = ImageDraw.Draw(frame_draw)
        for box in boxes:
            cv2.rectangle(img,box.tolist(), color=(255, 0, 0), width=6)
        
        # Add to frame list
        frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    
    
    for (x,y,w,h) in ear:
      cv2.rectangle(img, (x,y), (x+w,y+h), (255,0,0), 3)
      # cv2.rectangle(img, (x, y), (x+w, y+h), (0, 255, 0), 2)

    cv2.imshow('img',img)
    k = cv2.waitKey(30) & 0xff
    if k == ord('q'):
      break

#average 
#but which frame to consider 
cap.release()
cv2.destroyAllWindows()



