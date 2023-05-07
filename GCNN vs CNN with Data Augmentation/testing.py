from GCNN import train_GCNN
from CNN import train_CNN

if __name__ == '__main__':
  print('Training CNN with Data Augmentation')
  CNN = train_CNN()
  print('Training G-CNN with Data Augmentation')
  GCNN = train_GCNN()