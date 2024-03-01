import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import tensorflow as tf
import sys
sys.path.append('../TL-CAM')
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Reshape, Conv2D, Dense, UpSampling2D, Flatten, Input, BatchNormalization, Dropout, Activation, MaxPool2D, GlobalAveragePooling2D, MaxPooling2D
from tensorflow.keras.optimizers import Adam
from tlcam_layer import ScoreCAM
from utils import check_initial_weights, compileNtrain, test, evaluate
import datetime

seed_value = 42
np.random.seed(42)
tf.random.set_seed(42)


dir = "../../../Preliminay_TL-CAM/ImageNet_to_Stanford/StanfordDogs/"

def create_generators(size, batch_size=32, preprocess_input=None, seed=None):
    if seed is not None:
       np.random.seed(seed)
       tf.random.set_seed(seed)

    rescale=1/255 if preprocess_input is None else None

    test_datagen  = ImageDataGenerator(rescale=rescale, preprocessing_function=preprocess_input)


    test_generator       = test_datagen.flow_from_directory(
                             dir+'test/',
                             batch_size=batch_size,
                             target_size=(size[0], size[1]),
                             class_mode = 'categorical',
                             shuffle=False,
                             seed=seed)
    return test_generator


baseline      = load_model('baseline/baseline_model.h5')
cam_k5        = load_model('cam_k5/cam_k5_model.h5')
concat_cam_k5 = load_model('concat_cam_k5/concat_cam_k5_model.h5')
resnet_tl     = load_model('resnet_tl/resnet_tl_model.h5')

models_dict = {
    'baseline':baseline,
    'cam_k5':cam_k5,
    'concat_cam_k5':concat_cam_k5,
    'resnet_tl':resnet_tl,
}

for _, model in models_dict.items():
    model.summary()




for model_name, model in models_dict.items():
    outputDir='./'+model_name

    if model_name in ['baseline', 'cam_k5', 'concat_cam_k5']:
       x_test = create_generators([224, 224], batch_size=8, preprocess_input=None, seed=seed_value)


    if model_name == 'resnet_tl':
       x_test = create_generators([224, 224], batch_size=32, preprocess_input=None, seed=seed_value)



    evaluate(model, model_name, [x_test], outputDir)

