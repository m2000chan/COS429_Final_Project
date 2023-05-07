import numpy as np
import torch
import torch.nn as nn

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import tqdm

from data_utils import get_CIFAR10_data

from PIL import Image

class SimpleConv2d(nn.Module):
    def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True):
        super(SimpleConv2d, self).__init__()

        self.conv = nn.Conv2d(in_channels, out_channels, kernel_size, padding=padding, bias=bias)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.conv(x)

class CNN(torch.nn.Module):
    def __init__(self, n_classes=10):

        super(CNN, self).__init__()

        channels = [8, 16, 32, 64, 128]

        conv0 = SimpleConv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1, bias=True)
        conv1 = SimpleConv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, bias=True)
        conv2 = SimpleConv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1, bias=True)
        conv3 = SimpleConv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1, bias=True)
        conv4 = SimpleConv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1, bias=True)

        maxpool_conv = torch.nn.MaxPool2d(kernel_size=(3,3), stride=(2,2), padding=(1,1))
        relu = torch.nn.ReLU()
        gap = torch.nn.AdaptiveAvgPool2d(1)  # Global Average Pooling layer

        self.output_layer = torch.nn.Linear(channels[4], 10)

        self.model = torch.nn.Sequential(conv0, maxpool_conv, relu,
                                        conv1, maxpool_conv, relu,
                                        conv2, maxpool_conv, relu,
                                        conv3, maxpool_conv, relu,
                                        conv4, gap, relu)
    def forward(self, input: torch.Tensor):
        out = self.model(input)
        out = self.output_layer(torch.flatten(out, start_dim=1))
        return out

#------------------------------------------------------------------------------#
# CIFAR_Dataset()
#------------------------------------------------------------------------------#

class CIFAR_Dataset(Dataset):

    def __init__(self, mode, transform=None):
        assert mode in ['train', 'test']

        self.transform = transform

        X_train, y_train, X_test, y_test = get_CIFAR10_data()

        if mode == "train":
            self.images = X_train
            self.labels = y_train
        else:
            self.images = X_test
            self.labels = y_test

        self.num_samples = len(self.labels)
        self.images = np.transpose(self.images, (3, 0, 1, 2))  # Change to (N, H, W, C) format
        self.images = self.images.astype(np.float32)

        # Pad the images to have shape 33 x 33
        self.images = np.pad(self.images, pad_width=((0, 0), (1, 0), (1, 0), (0, 0)), mode='edge')

        assert self.images.shape == (self.labels.shape[0], 33, 33, 3)

    def __getitem__(self, index):
        image, label = self.images[index], self.labels[index]
        image = Image.fromarray(np.uint8(image * 255))
        if self.transform is not None:
            image = self.transform(image)
        return image, label

    def __len__(self):
        return len(self.labels)

device = 'cpu'
train_set = CIFAR_Dataset('train', ToTensor())
test_set = CIFAR_Dataset('test', ToTensor())

def train_model(model: torch.nn.Module):

  train_loader = torch.utils.data.DataLoader(train_set, batch_size=64)
  loss_function = torch.nn.CrossEntropyLoss()
  optimizer = torch.optim.Adam(model.parameters(), lr=7e-5, weight_decay=5e-5)

  model.to(device)
  model.train()

  for epoch in tqdm.tqdm(range(2)):
    
    for i, (x, t) in enumerate(train_loader):

        x = x.to(device)
        t = t.to(device)

        y = model(x)

        loss = loss_function(y, t)
        # print(loss.item())

        loss.backward()

        optimizer.step()
        optimizer.zero_grad()
  return model

def test_model(model: torch.nn.Module):
  test_loader = torch.utils.data.DataLoader(test_set, batch_size=64)
  total = 0
  correct = 0
  with torch.no_grad():
      model.eval()
      for i, (x, t) in tqdm.tqdm(enumerate(test_loader)):

          x = x.to(device)
          t = t.to(device)
          
          y = model(x)

          _, prediction = torch.max(y.data, 1)
          total += t.shape[0]
          correct += (prediction == t).sum().item()
  accuracy = correct/total*100.

  return accuracy

def train_CNN():

  model = CNN()
  
  model = train_model(model)
  acc = test_model(model)
  print(f'Test Accuracy of CNN: {acc :.3f}')