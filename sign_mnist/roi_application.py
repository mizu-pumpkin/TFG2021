#!/usr/bin/env python

#   █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
#   █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█

from umucv.stream import autoStream
from umucv.util import putText

import torch
import cv2 as cv
import numpy as np

from collections import deque
from statistics import mode

#   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░
#   █░▀░█ █▄█ █▄▀ ██▄ █▄▄

# Import my models
import my_models as my
# Choose model type
model = my.ResNet9(1,26)
# Load the model weights
if torch.cuda.is_available():
    model.load_state_dict(torch.load('models/classifier_v3_100.pth'))
else:
    model.load_state_dict(torch.load('models/classifier_v3_100.pth', map_location='cpu'))

#   ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
#   █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█

def predict_sign(sign):
    #sign = cv.flip(sign, 1) # No need to flip, when using model with data augmentation
    sign = cv.resize(sign, (28,28))
    sign = cv.cvtColor(sign, cv.COLOR_BGR2GRAY)
    sign = np.array(sign)
    sign = torch.from_numpy(sign).type(torch.FloatTensor)
    sign = torch.reshape(sign, (1, 28, 28))

    xb = sign.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return chr(preds[0].item() + 65)

def nothing(x):
    pass

mainWindowName = 'ASL classifier'
cv.namedWindow(mainWindowName)
cv.moveWindow(mainWindowName, 0, 0)
cv.createTrackbar('umbral', mainWindowName, 100, 255, nothing)
  
x1, y1, wh = 300, 100, 28*10
x2, y2 = x1+wh, y1+wh
history = deque(maxlen=5)
first = False

for key, frame in autoStream():
    # Flip frame horizontally so the app is easier to use
    frame = cv.flip(frame, 1)
    # Extract information from ROI and background
    roi = frame[y1:y2, x1:x2]
    if first == False:
        first = True
        bg = roi
    cv.imshow('ROI', roi)
    # Generate mask
    mask = np.sum(cv.absdiff(bg,roi), axis=2) > cv.getTrackbarPos('umbral', mainWindowName)
    cv.imshow('Mask', np.uint8(mask)*255)
    # Extract hand with mask
    hand = np.zeros([280,280,3],dtype=np.uint8)
    hand.fill(255)
    np.copyto( hand, roi, where = np.expand_dims(mask,axis=2) )
    cv.imshow('Hand', cv.cvtColor(np.uint8(hand), cv.COLOR_BGR2GRAY))
    # Show prediction based on a k-frame history
    history.append(predict_sign(hand))
    prediction = mode(history)
    putText(frame, 'Prediction: '+prediction, orig=(x1,y1-8))
    cv.rectangle(frame, (x1,y1-1), (x2,y2-1), color=(0,255,255), thickness=2)
    cv.imshow(mainWindowName, frame)

cv.destroyAllWindows()