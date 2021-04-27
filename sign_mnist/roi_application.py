#!/usr/bin/env python


#    █ █▀▄▀█ █▀█ █▀█ █▀█ ▀█▀ █▀
#    █ █░▀░█ █▀▀ █▄█ █▀▄ ░█░ ▄█

from umucv.stream import autoStream
from umucv.util import putText

import torch
import torch.nn as nn
import torch.nn.functional as F
import cv2 as cv
import numpy as np
import matplotlib
import matplotlib.pyplot as plt

#   █▀▄▀█ █▀█ █▀▄ █▀▀ █░░
#   █░▀░█ █▄█ █▄▀ ██▄ █▄▄

def accuracy(outputs, labels):
    _, preds = torch.max(outputs, dim=1)
    return torch.tensor(torch.sum(preds == labels).item() / len(preds))

class ImageClassificationBase(nn.Module):
    def training_step(self, batch):
        images, labels = batch 
        out = self(images)                  # Generate predictions
        loss = F.cross_entropy(out, labels) # Calculate loss
        return loss
    
    def validation_step(self, batch):
        images, labels = batch 
        out = self(images)                    # Generate predictions
        loss = F.cross_entropy(out, labels)   # Calculate loss
        acc = accuracy(out, labels)           # Calculate accuracy
        return {'val_loss': loss.detach(), 'val_acc': acc}
        
    def validation_epoch_end(self, outputs):
        batch_losses = [x['val_loss'] for x in outputs]
        epoch_loss = torch.stack(batch_losses).mean()   # Combine losses
        batch_accs = [x['val_acc'] for x in outputs]
        epoch_acc = torch.stack(batch_accs).mean()      # Combine accuracies
        return {'val_loss': epoch_loss.item(), 'val_acc': epoch_acc.item()}
    
    def epoch_end(self, epoch, result):
        print("Epoch [{}], val_loss: {:.4f}, val_acc: {:.4f}".format(epoch, result['val_loss'], result['val_acc']))

class SignMnistCNNModel(ImageClassificationBase):
    def __init__(self):
        super().__init__()
        self.network = nn.Sequential(
            # input: 1 x 28 x 28
            nn.Conv2d(1, 32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 64, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 64 x 14 x 14

            nn.Conv2d(64, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.Conv2d(128, 128, kernel_size=3, stride=1, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2, 2), # output: 128 x 7 x 7

            nn.Flatten(), 
            nn.Linear(128*7*7, 1024),
            nn.ReLU(),
            nn.Linear(1024, 512),
            nn.ReLU(),
            nn.Linear(512, 26))
        
    def forward(self, xb):
        return self.network(xb)

# Load the model weights
model = SignMnistCNNModel()
model.load_state_dict(torch.load('models/classifier_v2.pth', map_location='cpu'))

#   ▄▀█ █▀█ █▀█ █░░ █ █▀▀ ▄▀█ ▀█▀ █ █▀█ █▄░█
#   █▀█ █▀▀ █▀▀ █▄▄ █ █▄▄ █▀█ ░█░ █ █▄█ █░▀█

def predict_sign(sign):
    sign = cv.resize(sign, (28,28))
    sign = cv.cvtColor(sign, cv.COLOR_BGR2GRAY)
    sign = np.array(sign)
    sign = torch.from_numpy(sign).type(torch.FloatTensor)
    sign = torch.reshape(sign, (1, 28, 28))

    xb = sign.unsqueeze(0)
    yb = model(xb)
    _, preds = torch.max(yb, dim=1)
    return chr(preds[0].item() + 65)

cv.namedWindow("sign-MNIST")
cv.moveWindow('sign-MNIST', 0, 0)
    
x1, y1, wh = 300, 100, 28*10
x2, y2 = x1+wh, y1+wh

for key, frame in autoStream():
    # Flip frame horizontally
    frame = cv.flip(frame, 1)
    # Predict what's in the ROI
    sign = cv.flip(frame[y1:y2, x1:x2], 1)
    prediction = predict_sign(sign)
    # Show prediction and frame
    putText(frame, 'Prediction: '+prediction, orig=(x1,y1-8))
    cv.rectangle(frame, (x1,y1-1), (x2,y2-1), color=(0,255,255), thickness=2)
    cv.imshow('sign-MNIST', frame)

cv.destroyAllWindows()

