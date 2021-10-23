# After creating the dataset of at least 1 person using create_dataset.py 
# now it is time to predict person by reading real time frames

import cv2, numpy, os

size = 1
haar_file = 'haarcascade_frontalface_default.xml'
dataset = 'dataset'
print('\nModel is Training...........')
