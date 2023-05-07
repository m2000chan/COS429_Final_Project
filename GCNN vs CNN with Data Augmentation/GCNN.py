#------------------------------------------------------------------------------#
# Authors: Max Chan and Sophie Chen                                            #
# This implementation is largely based on a notebook from Gabriele Cesa        #
#------------------------------------------------------------------------------#

import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

from torch.utils.data import Dataset
from torchvision.transforms import ToTensor
import tqdm

from data_utils import get_CIFAR10_data

from PIL import Image

#------------------------------------------------------------------------------#
# C4                                                                           #
# Implements the *group law* of the group C_4                                  #
#------------------------------------------------------------------------------#

class C4:

  @staticmethod
  def product(r: int, s: int) -> int:
    # The input `r` and `s` must be integers in {0, 1, 2, 3} and 
    # represent two elements of the group.
    # The method should return the integer representing the 
    # product of the two input elements.
    
    allowable_values = [0, 1, 2, 3]
    r_is_valid = r in allowable_values
    s_is_valid = s in allowable_values

    if not r_is_valid:
      print("{} is not a valid input for r".format(r))

    if not s_is_valid:
      print("{} is not a valid input for s".format(s))

    is_valid = s_is_valid and r_is_valid

    if is_valid:
      return (r + s) % 4
    
  @staticmethod
  def inverse(r: int) -> int:
    # Implements the *inverse* operation of the group C_4.
    # The input `r` must be an integer in {0, 1, 2, 3} and represents an 
    # element of the group.
    # The method should return the integer representing the inverse of 
    # input element.

    allowable_values = [0, 1, 2, 3]
    r_is_valid = r in allowable_values

    if not r_is_valid:
      print("{} is not a valid input for r".format(r))

    if r_is_valid:
      r_inv = np.abs(r - 4) % 4
      return r_inv
    
#------------------------------------------------------------------------------#
# D4                                                                           #
# Implements the *group law* of the group D_4                                  #
#------------------------------------------------------------------------------#

class D4:

  @staticmethod
  def product(a: tuple, b: tuple) -> tuple:
    # The input `a` and `b` must be tuples containing two integers, 
    # e.g. `a = (f, r)`.
    # The two integers indicate whether the group element includes a reflection 
    # and the number of rotations.
    # The method should return the tuple representing the product of the two 
    # input elements.

    # r1 f1 r2 f2 = r1 r2_inv f1 f2
    af, ar = a
    bf, br = b

    r_allowable_values = [0, 1, 2, 3]
    ar_is_valid = ar in r_allowable_values
    br_is_valid = br in r_allowable_values

    f_allowable_values = [0, 1]
    af_is_valid = af in f_allowable_values
    bf_is_valid = bf in f_allowable_values

    if not (ar_is_valid and br_is_valid):
      print("{} or {} is not a valid input for r".format(ar, br))

    if not (af_is_valid and bf_is_valid):
      print("{} or {} is not a valid input for f".format(af, bf))

    r_is_valid = ar_is_valid and br_is_valid
    f_is_valid = af_is_valid and bf_is_valid
    is_valid = r_is_valid and f_is_valid

    if is_valid:
      br_inv = np.abs(br - 4) % 4
      r = (ar + br_inv) % 4 if af else (ar + br) % 4
      f = (af + bf) % 2 if bf else af
      return (f, r)
  
  @staticmethod
  def inverse(g: int) -> int:
    # Implements the *inverse* operation of the group D_4.
    # The input `g` must be a tuple containing two integers, e.g. `g = (f, r)`.
    # The two integers indicate whether the group element includes a reflection 
    # and the number of rotations.
    # The method should return the tuple representing the inverse of the 
    # input element.

    # (r f)_inv = f r_inv = r f
    # proof (r f)(r f) = r (f r) f = r r_inv f f
    gf, gr = g

    r_allowable_values = [0, 1, 2, 3]
    gr_is_valid = gr in r_allowable_values

    f_allowable_values = [0, 1]
    gf_is_valid = gf in f_allowable_values

    if not gr_is_valid:
      print("{} is not a valid input for r".format(gr))

    if not gf_is_valid:
      print("{} is not a valid input for f".format(gf))

    is_valid = gr_is_valid and gf_is_valid

    if is_valid:
      gr_inv = np.abs(gr - 4) % 4
      g_inv = g if gf else (gf, gr_inv)
      return g_inv
    
#------------------------------------------------------------------------------#
# rotate(x: torch.Tensor, r: int)                                              #
# Method which implements the action of the group element `g` indexed by `r`   #
# on the input image `x`.                                                      #
#------------------------------------------------------------------------------#

def rotate(x: torch.Tensor, r: int) -> torch.Tensor:
  # The method returns the image `g.x`
  # note that we rotate the last 2 dimensions of the input, since we want to 
  # later use this method to rotate minibatches containing multiple images
  return x.rot90(r, dims=(-2, -1))

#------------------------------------------------------------------------------#
# class IsotropicConv2d(torch.nn.Module)                                       #
#------------------------------------------------------------------------------#

class IsotropicConv2d(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, bias: bool = True):
    
    super(IsotropicConv2d, self).__init__()

    self.kernel_size = 3
    self.stride = 1
    self.dilation = 1
    self.padding = 1
    self.out_channels = out_channels
    self.in_channels = in_channels
    
    # In this block you need to create a tensor which stores the learnable 
    # weights
    # Recall that each 3x3 filter has only `2` learnable parameters, one for 
    # the center and one for the ring around it.
    # In total, there are `in_channels * out_channels` different filters.
    # Remember to wrap the weights tensor into a `torch.nn.Parameter` and set 
    # `requires_grad = True`

    # initialize the weights with some random values from a normal distribution 
    # with std = 1 / sqrt(out_channels * in_channels)
    self.weight = None
    self.weight = torch.nn.Parameter(torch.randn(out_channels * in_channels * 2)
                                     / np.sqrt(out_channels * in_channels), 
                                     requires_grad=True)

    if bias:
      self.bias = torch.nn.Parameter(torch.zeros(out_channels), 
                                     requires_grad=True)
    else:
      self.bias = None
  
  def build_filter(self) ->torch.Tensor:
    # using the tensor of learnable parameters, build the `out_channels x 
    # in_channels x 3 x 3` filter
    
    # Make sure that the tensor `filter3x3` is on the same device of 
    # `self.weight`
    
    filter3x3 = None

    device = self.weight.device
    filter3x3 = torch.ones(self.out_channels, self.in_channels, 3, 3, 
                           device=device)
    filter3x3[:, :, 1, 1] = self.weight[::2].reshape(self.out_channels, 
                                                     self.in_channels)
    filter3x3[:, :, 0, :] = self.weight[1::2].reshape(self.out_channels, 
                                                      self.in_channels, 1)
    filter3x3[:, :, 2, :] = self.weight[1::2].reshape(self.out_channels, 
                                                      self.in_channels, 1)
    filter3x3[:, :, :, 0] = self.weight[1::2].reshape(self.out_channels, 
                                                      self.in_channels, 1)
    filter3x3[:, :, :, 2] = self.weight[1::2].reshape(self.out_channels, 
                                                      self.in_channels, 1)

    return filter3x3

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    _filter = self.build_filter()

    return  torch.conv2d(x, _filter,
                            stride=self.stride,
                            padding=self.padding,
                            dilation=self.dilation,
                            bias=self.bias)

#------------------------------------------------------------------------------#
# rotate_p4()                                                                  #
#------------------------------------------------------------------------------#

def rotate_p4(y: torch.Tensor, r: int) -> torch.Tensor:
  # `y` is a function over p4, i.e. over the pixel positions and over the
  # elements of the group C_4.
  # This method implements the action of a rotation `r` on `y`.
  # To be able to reuse this function later with a minibatch of inputs, 
  # assume that the last two dimensions (`dim=-2` and `dim=-1`) of `y` are the 
  # spatial dimensions
  # while `dim=-3` has size `4` and is the C_4 dimension.
  # All other dimensions are considered batch dimensions
  assert len(y.shape) >= 3
  assert y.shape[-3] == 4

  y_out = rotate(y, r)
  y_out = torch.roll(y_out, r, dims=-3)

  return y_out

#------------------------------------------------------------------------------#
# LiftingConv2d                                                                #
#------------------------------------------------------------------------------#

class LiftingConv2d(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, padding: int = 0, bias: bool = True):
    
    super(LiftingConv2d, self).__init__()

    self.kernel_size = kernel_size
    self.stride = 1
    self.dilation = 1
    self.padding = padding
    self.out_channels = out_channels
    self.in_channels = in_channels
    
    # In this block you need to create a tensor which stores the learnable filters
    # Recall that this layer should have `out_channels x in_channels` different learnable filters, each of shape `kernel_size x kernel_size`
    # During the forward pass, you will build the bigger filter of shape `out_channels x 4 x in_channels x kernel_size x kernel_size` by rotating 4 times 
    # the learnable filters in `self.weight`
    
    # initialize the weights with some random values from a normal distribution with std = 1 / sqrt(out_channels * in_channels)

    self.weight = None

    ### BEGIN SOLUTION
    self.weight = torch.nn.Parameter(torch.randn(out_channels * in_channels * kernel_size**2) / np.sqrt(out_channels * in_channels), requires_grad=True)
    
    ### END SOLUTION

    # This time, you also need to build the bias
    # The bias is shared over the 4 rotations
    # In total, the bias has `out_channels` learnable parameters, one for each independent output
    # In the forward pass, you need to convert this bias into an "expanded" bias by repeating each entry `4` times
    
    self.bias = None
    if bias:
    ### BEGIN SOLUTION
      self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)

    ### END SOLUTION  
  
  def build_filter(self) ->torch.Tensor:
    # using the tensors of learnable parameters, build 
    # - the `out_channels x 4 x in_channels x kernel_size x kernel_size` filter
    # - the `out_channels x 4` bias
    
    _filter = None
    _bias = None

    # Make sure that the filter and the bias tensors are on the same device of `self.weight` and `self.bias`

    # First build the filter
    # Recall that `_filter[:, i, :, :, :]` should contain the learnable filter rotated `i` times

    ### BEGIN SOLUTION
    device = self.weight.device
    _filter = torch.ones(self.out_channels, 4, self.in_channels, self.kernel_size, self.kernel_size, device=device)
    _filter[:, 0, :, :, :] = self.weight.reshape(self.out_channels, self.in_channels, self.kernel_size, self.kernel_size)
    _filter[:, 1, :, :, :] = rotate(_filter[:, 0, :, :], 1)
    _filter[:, 2, :, :, :] = rotate(_filter[:, 0, :, :], 2)
    _filter[:, 3, :, :, :] = rotate(_filter[:, 0, :, :], 3)

    ### END SOLUTION

    # Now build the bias
    # Recall that `_bias[:, i]` should contain a copy of the learnable bias for each `i=0,1,2,3`

    if self.bias is not None:
    ### BEGIN SOLUTION
      device = self.bias.device
      _bias = torch.ones(self.out_channels, 4, device=device)
      # _bias = torch.tile(self.bias.reshape(self.out_channels, 1), (1,4)).to(device)


    ### END SOLUTION
    else:
      _bias = None

    return _filter, _bias

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    _filter, _bias = self.build_filter()
    
    assert _bias.shape == (self.out_channels, 4)
    assert _filter.shape == (self.out_channels, 4, self.in_channels, self.kernel_size, self.kernel_size)

    # to be able to use torch.conv2d, we need to reshape the filter and bias to stack together all filters
    _filter = _filter.reshape(self.out_channels * 4, self.in_channels, self.kernel_size, self.kernel_size)
    _bias = _bias.reshape(self.out_channels * 4)

    out = torch.conv2d(x, _filter,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       bias=_bias)
    
    # `out` has now shape `batch_size x out_channels*4 x W x H`
    # we need to reshape it to `batch_size x out_channels x 4 x W x H` to have the shape we expect

    return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])

#------------------------------------------------------------------------------#
# GroupConv2d                                                                  #
#------------------------------------------------------------------------------#

class GroupConv2d(torch.nn.Module):

  def __init__(self, in_channels: int, out_channels: int, kernel_size: int, 
               padding: int = 0, bias: bool = True):
    
    super(GroupConv2d, self).__init__()

    self.kernel_size = kernel_size
    self.stride = 1
    self.dilation = 1
    self.padding = padding
    self.out_channels = out_channels
    self.in_channels = in_channels
    
    # In this block you need to create a tensor which stores the learnable 
    # filters
    # Recall that this layer should have `out_channels x in_channels` 
    # different learnable filters, each of shape `4 x kernel_size x kernel_size`
    # During the forward pass, you will build the bigger filter of shape 
    # `out_channels x 4 x in_channels x 4 x kernel_size x kernel_size` by 
    # rotating 4 times 
    # the learnable filters in `self.weight`
    
    # initialize the weights with some random values from a normal distribution 
    # with std = 1 / np.sqrt(out_channels * in_channels)

    self.weight = None
    self.weight = torch.nn.Parameter(torch.randn(out_channels * in_channels * 4 * kernel_size**2) / np.sqrt(out_channels * in_channels), requires_grad=True)
    

    # The bias is shared over the 4 rotations
    # In total, the bias has `out_channels` learnable parameters, one for each 
    # independent output
    # In the forward pass, you need to convert this bias into an "expanded" bias
    # by repeating each entry `4` times
    
    self.bias = None
    if bias:
      self.bias = torch.nn.Parameter(torch.zeros(out_channels), requires_grad=True)
  
  def build_filter(self) ->torch.Tensor:
    # using the tensors of learnable parameters, build 
    # - the `out_channels x 4 x in_channels x 4 x kernel_size x kernel_size` 
    # filter
    # - the `out_channels x 4` bias
    
    _filter = None
    _bias = None

    # Make sure that the filter and the bias tensors are on the same device of 
    # `self.weight` and `self.bias`

    # First build the filter
    # Recall that `_filter[:, r, :, :, :, :]` should contain the learnable 
    # filter rotated `r` times
    # Also, recall that a rotation includes both a rotation of the pixels and 
    # a cyclic permutation of the 4 rotational input channels

    device = self.weight.device
    _filter = torch.ones(self.out_channels, 4, self.in_channels, 4, 
                         self.kernel_size, self.kernel_size, device=device)
    _filter[:, 0, :, :, :, :] = self.weight.reshape(self.out_channels, 
                                                    self.in_channels, 4, 
                                                    self.kernel_size, 
                                                    self.kernel_size)
    _filter[:, 1, :, :, :, :] = rotate_p4(_filter[:, 0, :, :], 1)
    _filter[:, 2, :, :, :, :] = rotate_p4(_filter[:, 0, :, :], 2)
    _filter[:, 3, :, :, :, :] = rotate_p4(_filter[:, 0, :, :], 3)

    # Now build the bias
    # Recall that `_bias[:, i]` should contain a copy of the learnable bias 
    # for each `i`

    if self.bias is not None:
      device = self.bias.device
      _bias = torch.tile(self.bias.reshape(self.out_channels, 
                                           1), (1,4)).to(device)

    else:
      _bias = None

    return _filter, _bias

  def forward(self, x: torch.Tensor) -> torch.Tensor:

    _filter, _bias = self.build_filter()

    assert _bias.shape == (self.out_channels, 4)
    assert _filter.shape == (self.out_channels, 4, self.in_channels, 4, 
                             self.kernel_size, self.kernel_size)

    # to be able to use torch.conv2d, we need to reshape the filter and bias 
    # to stack together all filters
    _filter = _filter.reshape(self.out_channels * 4, self.in_channels * 4, 
                              self.kernel_size, self.kernel_size)
    _bias = _bias.reshape(self.out_channels * 4)

    # this time, also the input has shape `batch_size x in_channels x 4 x W x H`
    # so we need to reshape it to `batch_size x in_channels*4 x W x H` to be 
    # able to use torch.conv2d
    x = x.view(x.shape[0], self.in_channels*4, x.shape[-2], x.shape[-1])

    out = torch.conv2d(x, _filter,
                       stride=self.stride,
                       padding=self.padding,
                       dilation=self.dilation,
                       bias=_bias)
    
    # `out` has now shape `batch_size x out_channels*4 x W x H`
    # we need to reshape it to `batch_size x out_channels x 4 x W x H` to 
    # have the shape we expect

    return out.view(-1, self.out_channels, 4, out.shape[-2], out.shape[-1])

#------------------------------------------------------------------------------#
# C4CNN                                                                        #
#------------------------------------------------------------------------------#

# The network performs a first lifting layer with  8  output channels and is 
# followed by  4  group convolution with, respectively,  16 ,  32 ,  64  and  
# 128  output channels. All convolutions have kernel size  3 , padding  1  and 
# stride  1  and should use the bias. All convolutions are followed by 
# torch.nn.MaxPool3d and torch.nn.ReLU. Note that we use MaxPool3d rather than 
# MaxPool2d since our feature tensors have  5  dimensions (there is an 
# additional dimension of size  4 ). In all pooling layers, we will use a kernel
#  of size  (1,3,3) , a stride of  (1,2,2)  and a padding of  (0,1,1). This 
# ensures pooling is done only on the spatial dimensions, while the rotational 
# dimension is preserved. The last pooling layer, however, will also pool over 
# the rotational dimension so it will use a kernel of size  (4,3,3) , stride  
# (1,1,1)  and padding  (0,0,0) .

class C4CNN(torch.nn.Module):
  def __init__(self, n_classes=10):

    super(C4CNN, self).__init__()

    channels = [8, 16, 32, 64, 128]

    lifting_layer = LiftingConv2d(in_channels=3, out_channels=channels[0], kernel_size=3, padding=1, bias=True)
    conv1 = GroupConv2d(in_channels=channels[0], out_channels=channels[1], kernel_size=3, padding=1, bias=True)
    conv2 = GroupConv2d(in_channels=channels[1], out_channels=channels[2], kernel_size=3, padding=1, bias=True)
    conv3 = GroupConv2d(in_channels=channels[2], out_channels=channels[3], kernel_size=3, padding=1, bias=True)
    conv4 = GroupConv2d(in_channels=channels[3], out_channels=channels[4], kernel_size=3, padding=1, bias=True)

    maxpool_conv = torch.nn.MaxPool3d(kernel_size=(1,3,3), stride=(1,2,2), padding=(0,1,1))
    relu = torch.nn.ReLU()
    maxpool_out = torch.nn.MaxPool3d(kernel_size=(4,3,3), stride=(1,1,1), padding=(0,0,0))

    self.output_layer = torch.nn.Linear(channels[4], 10)

    self.model = torch.nn.Sequential(lifting_layer, maxpool_conv, relu,
                                      conv1, maxpool_conv, relu,
                                      conv2, maxpool_conv, relu,
                                      conv3, maxpool_conv, relu,
                                      conv4, maxpool_out, relu)

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
  optimizer = torch.optim.Adam(model.parameters(), lr=4.1e-4, weight_decay=8e-5)

  model.to(device)
  model.train()

  for epoch in tqdm.tqdm(range(500)):
    
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

def train_GCNN(model_path=None):
  model = C4CNN()

  if model_path is not None:
        model.load_state_dict(torch.load(model_path))
  
  model = train_model(model)
  torch.save(model.state_dict(), 'GCNN_model.pth')
  acc = test_model(model)
  print(f'Test Accuracy of G-CNN: {acc :.3f}')