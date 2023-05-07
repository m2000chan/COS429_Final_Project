#------------------------------------------------------------------------------#
# Authors: Max Chan and Sophie Chen                                            #
#------------------------------------------------------------------------------#

from GCNN import train_GCNN
from CNN import train_CNN

if __name__ == '__main__':
  print('Training CNN')
  CNN = train_CNN(model_path='CNN_model.pth')
  print('Training G-CNN')
  GCNN = train_GCNN(model_path='GCNN_model.pth')