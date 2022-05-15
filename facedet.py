from tkinter import font
import cv2
from facenet_pytorch import MTCNN
import math
import torch
import numpy as np
from PIL import Image, ImageDraw, ImageFont

import matplotlib.pyplot as plt

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
print('Running on device: {}'.format(device))

ear_cascade = cv2.CascadeClassifier('cascade.xml')
cascPath = 'facedet.xml'

faceCascade = cv2.CascadeClassifier(cascPath)

mtcnn = MTCNN(device=device, keep_all=True)

img = Image.open('images/im1.jpg')


print(img.size)


# def detect(img):
boxes, _ = mtcnn.detect(img)
imgc = img.copy()
draw = ImageDraw.Draw(imgc)

a = boxes[0][0]
b = boxes[0][1]
c = boxes[0][2]
d = boxes[0][3]

draw.rectangle((a, b, c, d), width=2)
print(boxes)
plt.imshow(imgc)
plt.show()

img1 = Image.open('images/im2.jpg')

boxes, _ = mtcnn.detect(img1)
imgc1 = img1.copy()
draw = ImageDraw.Draw(imgc1)

e = boxes[0][1]
f = boxes[0][2]
g = boxes[0][3]
h = boxes[0][0]

print(boxes)

draw.line((c, d, a, d), fill=(255, 0, 0), width=2)
draw.line((c, d, f, e), fill=(255, 0, 0), width=4)
draw.line((c, d, c, b), fill=(255, 0, 0), width=2)


def slope(p1, p2):
    return (p2[1]-p1[1])/(p2[0]-p1[0])


x1 = [c, d]
x2 = [a, d]
x3 = [f, e]

font = ImageFont.truetype("arial.ttf", 100)
m1 = slope(x1, x2)
m2 = slope(x1, x3)
print(m1, m2)
print('8'*10)

angle = math.atan((m2-m1)/1+m1*m2)

print('angle', (m2-m1)/1+m1*m2)
print('radian', angle)
angle = round(math.degrees((angle)))
angle = 90-angle

print('round angle :', angle)

draw.text((c, d), text=f'{angle}', color=(0, 0, 255), font=font)

fig, ax = plt.subplots(1, 2)


ax[0].imshow(imgc)
ax[1].imshow(imgc1)
# plt.imshow(imgc1)
ax[0].set_xticklabels([])
ax[1].set_xticklabels([])
plt.tight_layout()

plt.show()
