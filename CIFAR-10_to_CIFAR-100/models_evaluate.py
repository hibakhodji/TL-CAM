import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import tensorflow as tf
import sys
sys.path.append('../TLCAM')
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import numpy as np
from sklearn.model_selection import train_test_split
from tlcam_layer import ScoreCAM, Callbacks
from utils import evaluate

seed_value = 42
np.random.seed(42)
tf.random.set_seed(42)

#outputDir=sys.argv[1]

sourceModel = load_model("../cifar10_model.h5")
sourceModel.summary()


# Loading CIFAR-100
(x_train, y_train), (x_test, y_test)=tf.keras.datasets.cifar100.load_data()


x_train, x_valid, y_train, y_valid = train_test_split(x_train, y_train, test_size=0.2, random_state=seed_value, stratify=y_train)


# Preparing
x_train = x_train / 255
x_valid = x_valid / 255
x_test = x_test / 255

y_train = keras.utils.to_categorical(y_train, 100)
y_valid = keras.utils.to_categorical(y_valid, 100)
y_test = keras.utils.to_categorical(y_test, 100)



print("x_train shape", x_train.shape)
print("x_test shape", x_test.shape)

baseline = load_model('baseline/baseline_model.h5')
cam_k5   = load_model('cam_k5/cam_k5_model.h5')
cam_k25  = load_model('cam_k25/cam_k25_model.h5')
cam_k50  = load_model('cam_k50/cam_k50_model.h5')
tl_model = load_model('tl_model/tl_model_model.h5')

models_dict = {
    'baseline':baseline,
    'cam_k5':cam_k5,
    'cam_k25':cam_k25,
    'cam_k50':cam_k50,
    'tl_model':tl_model,
}

for _, model in models_dict.items():
   model.summary()



for model_name, model in models_dict.items():
    outputDir='./'+model_name

    if model_name in ['baseline','cam_k5', 'cam_k25', 'cam_k50']:
       batchSize = 16
    else:
       batchSize = 32

    if model_name in ['cam_k5', 'cam_k25', 'cam_k50']:
       callbacks = [Callbacks()]
    else:
       callbacks = []

    evaluate(model, model_name, [x_test, y_test], outputDir, batchSize)



