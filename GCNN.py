import numpy as np
import torch

import matplotlib
import matplotlib.pyplot as plt

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
# GroupConv2d                                                                  #
# Testing all the testing code from the entire notebook in one go              #
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
# testing()                                                                    #
# Testing all the testing code from the entire notebook in one go              #
#------------------------------------------------------------------------------#
    
def testing():

    # C4 Testing
    assert C4.product(1, 3) == 0
    assert C4.product(0, 0) == 0
    assert C4.product(2, 3) == 1
    assert C4.inverse(0) == 0
    assert C4.inverse(1) == 3

    # D4 Testing
    e = (0, 0) # the identity element
    f = (1, 0) # the horizontal reflection
    r = (0, 1) # the rotation by 90 degrees

    # Let's verify that the implementation is consistent with the instructions given
    assert D4.product(e, e) == e
    assert D4.product(f, f) == e
    assert D4.product(f, r) == D4.product(D4.inverse(r), f)

    # Let's verify that the implementation satisfies the group axioms
    a = (1, 2)
    b = (0, 3)
    c = (1, 1)

    assert D4.product(a, e) == a
    assert D4.product(e, a) == a
    assert D4.product(b, D4.inverse(b)) == e
    assert D4.product(D4.inverse(b), b) == e

    assert D4.product(D4.product(a, b), c) == D4.product(a, D4.product(b, c))

    # Let's check if the layer is really equivariant
    in_channels = 5
    out_channels = 10
    batchsize = 6
    S = 33

    layer = IsotropicConv2d(in_channels=in_channels, out_channels=out_channels, bias=True)
    layer.eval()

    x = torch.randn(batchsize, in_channels, S, S)
    gx = rotate(x, 1)


    psi_x = layer(x)
    psi_gx = layer(gx)

    g_psi_x = rotate(psi_x, 1)

    assert psi_x.shape == g_psi_x.shape
    assert psi_x.shape == (batchsize, out_channels, S, S)

    # check the model is giving meaningful outputs
    assert not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4)

    # check equivariance
    assert torch.allclose(psi_gx, g_psi_x, atol=1e-6, rtol=1e-6)

    # check the model has the right number of parameters
    assert layer.weight.numel() == in_channels * out_channels * 2

    # Let's test a rotation by r=1
    y = torch.randn(1, 1, 4, 33, 33)**2

    ry = rotate_p4(y, 1)

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(y[0, 0, i].numpy())
    fig.suptitle('Original y')
    # plt.show()

    fig, axes = plt.subplots(1, 4, sharex=True, sharey=True, squeeze=True, figsize=(16, 4))
    for i in range(4):
        axes[i].imshow(ry[0, 0, i].numpy())
    fig.suptitle('Rotated y')
    # plt.show()


    # check that the images are actually rotated:
    for _ in range(10):
        p = np.random.randint(0, 33, size=2)
        s = np.random.randint(0, 4)

        # compute r^-1 s
        _rs = C4.product(C4.inverse(1), s)
        
        # compute r^-1 p
        # note that the rotation is around the central pixel (16, 16)
        # A rotation by r^-1 = -90 degrees maps (X, Y) -> (Y, -X)
        center = np.array([16, 16])
        # center the point
        centered_p = p - center
        # rotate round the center
        rotated_p = np.array([centered_p[1], -centered_p[0]])
        # shift the point back
        _rp = rotated_p + center

        # Finally check that [r.y](p, s) = y(r^-1 p, r^-1 s)

        # However, in a machine, an image is stored with the coordinates (H-1-Y, X) rather than the usual (X, Y), where H is the height of the image;
        # we need to take this into account
        assert torch.isclose(
            ry[..., s, 32-p[1], p[0]],
            y[..., _rs, 32-_rp[1], _rp[0]],
            atol=1e-5, rtol=1e-5
        )

    # Let's check if the layer is really equivariant

    in_channels = 5
    out_channels = 10
    kernel_size = 3
    batchsize = 4
    S = 33

    layer = GroupConv2d(in_channels=in_channels, out_channels=out_channels, kernel_size = kernel_size, padding=1, bias=True)
    layer.eval()

    x = torch.randn(batchsize, in_channels, 4, S, S)**2
    # the input image belongs to the space Y, so this time we use the new action to rotate it
    gx = rotate_p4(x, 1)

    # compute the output
    psi_x = layer(x)
    psi_gx = layer(gx)

    # the output is a function in the space Y, so we need to use the new action to rotate it
    g_psi_x = rotate_p4(psi_x, 1)

    assert psi_x.shape == g_psi_x.shape
    assert psi_x.shape == (batchsize, out_channels, 4, S, S)

    # check the model is giving meaningful outputs
    assert not torch.allclose(psi_x, torch.zeros_like(psi_x), atol=1e-4, rtol=1e-4)

    # check equivariance
    assert torch.allclose(psi_gx, g_psi_x, atol=1e-5, rtol=1e-5)

    # check the model has the right number of parameters
    assert layer.weight.numel() == in_channels * out_channels * 4* kernel_size**2
    assert layer.bias.numel() == out_channels

    print("Passed all tests!")

if __name__ == '__main__':
    testing()