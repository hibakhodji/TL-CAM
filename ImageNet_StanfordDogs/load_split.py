import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
from pathlib import Path
import numpy as np
import matplotlib.pyplot as plt

import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers, models
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import splitfolders



#Loadind dataset

DOWNLOAD_ROOT = "http://vision.stanford.edu/aditya86/ImageNetDogs/"
FILENAME = "images.tar"
filepath = keras.utils.get_file(FILENAME, DOWNLOAD_ROOT + FILENAME, extract=True)
data_dir = Path(filepath).parent / "Images"
print(data_dir)
class_names = os.listdir(data_dir)
n_classes = len(os.listdir(data_dir))

n_images = 0
for i in range(n_classes):
    n_images += len(os.listdir(data_dir / class_names[i]))
print("Number of images: ", n_images)
print("Number of classes: ", n_classes)

#Splitting dataset
splitfolders.ratio(data_dir, output="../StanfordDogs", seed=1337, ratio=(.8, 0.1,0.1))

