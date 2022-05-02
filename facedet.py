import cv2
from facenet_pytorch import MTCNN,InceptionResnetV1
import torch
import numpy as np
import mmcv
from PIL import Image, ImageDraw
from IPython import display
import sys
import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

ear_cascade = cv2.CascadeClassifier('cascade.xml')
cascPath = 'facedet.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

mtcnn = MTCNN(image_size=160,keep_all=True, device=device)

resnet = InceptionResnetV1(pretrained='vggface2').eval()

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
    print(img.shape)
    ear = ear_cascade.detectMultiScale(img, 1.3, 5)
    face = mtcnn(img)  
  
    # for box in boxes:
    #     cv2.rectangle(img,box.tolist(), color=(255, 0, 0), width=6)
    print(face.shape)
    
    cv2.imshow(face.permute(1, 2, 0).int().numpy())
    
        # Add to frame list
        
        # frames_tracked.append(frame_draw.resize((640, 360), Image.BILINEAR))
    
    
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



