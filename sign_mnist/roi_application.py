#!/usr/bin/env python

#   █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
#   █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█

from umucv.stream import autoStream
from umucv.util import putText

import torch
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

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

cv.namedWindow("ASL classifier")
cv.moveWindow('ASL classifier', 0, 0)
    
x1, y1, wh = 300, 100, 28*10
x2, y2 = x1+wh, y1+wh

for key, frame in autoStream():
    # Flip frame horizontally so the app is easier to use
    frame = cv.flip(frame, 1)
    # Predict what's in the ROI
    roi = frame[y1:y2, x1:x2]
    prediction = predict_sign(roi)
    cv.imshow('ROI', cv.cvtColor(roi, cv.COLOR_BGR2GRAY))
    # Show prediction and frame
    putText(frame, 'Prediction: '+prediction, orig=(x1,y1-8))
    cv.rectangle(frame, (x1,y1-1), (x2,y2-1), color=(0,255,255), thickness=2)
    cv.imshow('ASL classifier', frame)

cv.destroyAllWindows()