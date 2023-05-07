from GCNN import train_GCNN
from CNN import train_CNN

if __name__ == '__main__':
  print('Training CNN')
  CNN = train_CNN()
  print('Training G-CNN')
  GCNN = train_GCNN()