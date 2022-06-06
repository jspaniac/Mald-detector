#---------#
# IMPORTS #
#---------#

import os
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision import transforms, utils

import torchvision.transforms as transforms
import torchvision.io as io
import numpy as np
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

from cv2 import cv2
from facenet_pytorch import MTCNN
from PIL import Image

#-----------#
# CONSTANTS #
#-----------#

CLASS = ["positive", "negative"]
KEY = ["angry", "disgusted", "fearful", "happy", "neutral", "sad", "surprised"]
KEY_TO_CLASS = [1, 1, 1, 0, 0, 1, 1]

ROOT = "./data"

#---------#
# CLASSES #
#---------#

class FaceEmotionDataset(Dataset):

    def __init__(self, root, train, transform=None):
        self.root = os.path.join(root, 
                    os.path.join("emotions", 
                                 "train" if train else "test"))
        self.transform = transform

        # Calculate overall number of files for each category
        self.total = 0
        self.files = []
        for i in range(len(KEY)):
          size = len(os.listdir(os.path.join(self.root, KEY[i])))
          self.files.append(size)
          self.total += size

    def __len__(self):
        return self.total

    def __getitem__(self, idx):
        for i in range(len(KEY)):
            if idx < self.files[i]:
                img_name = os.path.join(os.path.join(self.root, KEY[i]),
                                        "im" + str(idx) + ".png")
                image = io.read_image(img_name).float()
                if self.transform:
                    image = self.transform(image)
                
                ret = [image, KEY_TO_CLASS[i]]
                return ret
            else:
                idx -= self.files[i]
        return None

#------------#
# NUERAL NET #
#------------#

class EmotionConvBNN(nn.Module):
    def __init__(self):
        super(EmotionConvBNN, self).__init__()

        self.conv1 = nn.Conv2d(1, 24, 5, stride=2, padding=1)
        self.bn1 = nn.BatchNorm2d(24)
        
        self.conv2 = nn.Conv2d(24, 48, 3, stride=2, padding=1)
        self.bn2 = nn.BatchNorm2d(48)
        
        self.conv3 = nn.Conv2d(48, 96, 3, stride=2, padding=1)
        self.bn3 = nn.BatchNorm2d(96)

        self.conv4 = nn.Conv2d(96, 192, 3, stride=2, padding=1)
        self.bn4 = nn.BatchNorm2d(192)
        
        self.fc2 = nn.Linear(1728, len(CLASS))

    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.relu(x)

        x = self.conv2(x)
        x = self.bn2(x)
        x = F.relu(x)

        x = self.conv3(x)
        x = self.bn3(x)
        x = F.relu(x)

        x = self.conv4(x)
        x = self.bn4(x)
        x = F.relu(x)

        x = torch.flatten(x, 1)
        x = self.fc2(x)
        return x

#-----------------#
# DATA COLLECTION #
#-----------------#

def get_emotion_data(augmentation=0):
  # Data augmentation transformations. Not for Testing!
  transform_test = None
  if augmentation:
    transform_train = transforms.Compose([
      transforms.RandomHorizontalFlip(),    # 50% of time flip image along y-axis
      transforms.RandomRotation(15),
    ])
  else :
    transform_train = transform_test
  

  trainset = FaceEmotionDataset(root=ROOT, train=True, transform=transform_train)
  trainloader = torch.utils.data.DataLoader(trainset, batch_size=16, shuffle=True,
                                            num_workers=0)

  testset = FaceEmotionDataset(root=ROOT, train=False, transform=transform_test)
  testloader = torch.utils.data.DataLoader(testset, batch_size=16, shuffle=False,
                                           num_workers=0)
  return {'train': trainloader, 'test': testloader, 'classes': CLASS}

def train(net, dataloader, epochs=1, lr=0.01, momentum=0.9, decay=0.0, verbose=1):
  net.to(device)
  losses = []
  criterion = nn.CrossEntropyLoss()
  optimizer = optim.SGD(net.parameters(), lr=lr, momentum=momentum, weight_decay=decay)
  for epoch in range(epochs):
    sum_loss = 0.0
    for i, batch in enumerate(dataloader, 0):
        # get the inputs; data is a list of [inputs, labels]
        inputs, labels = batch[0].to(device), batch[1].to(device)

        # zero the parameter gradients
        optimizer.zero_grad()

        # forward + backward + optimize 
        outputs = net(inputs)
        loss = criterion(outputs, labels)
        loss.backward()  # autograd magic, computes all the partial derivatives
        optimizer.step() # takes a step in gradient direction

        # print statistics
        losses.append(loss.item())
        sum_loss += loss.item()
        if i % 100 == 99:    # print every 100 mini-batches
            if verbose:
              print('[%d, %5d] loss: %.3f' %
                  (epoch + 1, i + 1, sum_loss / 100))
            sum_loss = 0.0
  return losses

def accuracy(net, dataloader):
  correct = 0
  total = 0
  with torch.no_grad():
      for batch in dataloader:
          images, labels = batch[0].to(device), batch[1].to(device)
          outputs = net(images)
          _, predicted = torch.max(outputs.data, 1)
          total += labels.size(0)
          correct += (predicted == labels).sum().item()
  return correct/total

def smooth(x, size):
  return np.convolve(x, np.ones(size)/size, mode='valid')

##############

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
print(device)
print()

emotion_data = get_emotion_data(augmentation=1)

conv_net = EmotionConvBNN()
conv_losses = train(conv_net, emotion_data['train'], epochs=10, lr=0.01)

print("Training accuracy: %f" % accuracy(conv_net, emotion_data['train']))
print("Testing  accuracy: %f" % accuracy(conv_net, emotion_data['test']))
print()

################

PADDING = 60
mtcnn = MTCNN(select_largest=False)
cap = cv2.VideoCapture(0)

percent_pos = [0.0, 0.0]
while True:
    ret, frame = cap.read()
    boxes, probs = mtcnn.detect(Image.fromarray(frame), landmarks=False)
    try:
      if not isinstance(boxes, type(None)):
        for box in boxes:
            p1 = (int(box[0]) - PADDING, int(box[1]) - PADDING)
            p2 = (int(box[2]) + PADDING, int(box[3]) + PADDING)
            # crop out this image
            cropped = frame[p1[1]:p2[1], p1[0]:p2[0]]
            
            # scale it to 48x48 / grayscale
            cropped = cv2.resize(cropped, (48, 48))
            cropped = cv2.cvtColor(cropped, cv2.COLOR_RGB2GRAY)

            # run it through model
            outputs = conv_net(torch.from_numpy(cropped).view(1, 1, 48, 48).float().to(device))
            _, predicted = torch.max(outputs.data, 1)
            
            
            # increment stats
            percent_pos[1] += 1
            percent_pos[0] += 1 if predicted == 0 else 0

            # post result
            cv2.rectangle(frame, p1, p2, (0, 255, 0), 2)
            cv2.putText(frame, CLASS[predicted], (p1[0], p2[1]), 
                        cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2, cv2.LINE_AA)
            
            # print percentage
            print("percent positive: " + str(percent_pos[0] / percent_pos[1]))
            
            # debug show cropped
            cv2.imshow('cropped', cropped)
            # save one image - for the report writeup
            # print(percent_pos[0])
            # if (percent_pos[0] < 10):
            #   os.chdir('/Users/jspaniac/Desktop/project')
            #   cv2.imwrite('before.bmp', frame)
            #   cv2.imwrite('after.bmp', cropped)
    except:
      pass

    cv2.imshow('frame', frame)
    if cv2.waitKey(1) & 0xFF == ord('q'):
        break

cap.release()
