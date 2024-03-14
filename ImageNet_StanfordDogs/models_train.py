import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import tensorflow as tf
import sys
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.layers import Reshape, Conv2D, Dense, UpSampling2D, Flatten, Input, BatchNormalization, Dropout, Activation, MaxPool2D, GlobalAveragePooling2D, MaxPooling2D, Concatenate
from tensorflow.keras.optimizers import Adam
from TLCAM.tlcam_layer import ScoreCAM, Callbacks
from TLCAM.utils import check_initial_weights, compileNtrain, test, evaluate, get_conv_layer_name, SaveImageLayer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard, ReduceLROnPlateau
import datetime
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_ResNet152V2
from tensorflow.keras.applications.inception_v3 import InceptionV3
from tensorflow.keras.applications.inception_v3 import preprocess_input as preprocess_input_InceptionV3
from tf_explain.callbacks.grad_cam import GradCAMCallback
from tensorflow.keras import regularizers
from tensorflow.keras.layers.experimental.preprocessing import RandomFlip, RandomTranslation
print("TensorFlow version:", tf.__version__)

seed_value = 42
LR = 0.003
data_augmentation = tf.keras.Sequential([
    RandomFlip("horizontal_and_vertical", seed=seed_value),
])



sourceModel = ResNet152V2(weights='imagenet', include_top=True)


def get_vis_data(generator, label):
    generator.batch_size=9
    for x_batch, y_batch in generator:
        if all(idx  == label for idx in tf.argmax(y_batch, axis=1)): 
           return x_batch, y_batch

def combine_generator(gen1, gen2):
    while True:
        x1, y1 = gen1.next()
        x2, y2 = gen2.next()
        yield [x1, x2], y1 


dir = "../../TL-CAM/ImageNet_to_Stanford/StanfordDogs/"
cam_dir = 'CAM_training_images/'

def create_generators(traindir, size, batch_size=32, preprocess_input=None, seed=None):
    if seed is not None:
       np.random.seed(seed)
       tf.random.set_seed(seed)

    rescale=1/255 if preprocess_input is None else None

    train_datagen = ImageDataGenerator(rescale=rescale, preprocessing_function=preprocess_input)
    valid_datagen = ImageDataGenerator(rescale=rescale, preprocessing_function=preprocess_input)
    test_datagen  = ImageDataGenerator(rescale=rescale, preprocessing_function=preprocess_input)

    train_generator      = train_datagen.flow_from_directory(
                             traindir,
                             batch_size=batch_size,
                             target_size=(size[0], size[1]),
                             shuffle = True,
                             class_mode = 'categorical',
                             seed=seed)


    validation_generator = valid_datagen.flow_from_directory(
                             dir+'val/',
                             batch_size=batch_size,
                             target_size=(size[0], size[1]),
                             class_mode ='categorical',
                             shuffle= False,
                             seed=seed)

    test_generator       = test_datagen.flow_from_directory(  
                             dir+'test/',
                             batch_size=batch_size,
                             target_size=(size[0], size[1]),
                             class_mode = 'categorical',
                             shuffle=False,
                             seed=seed)
    return train_generator, validation_generator, test_generator 



def create_TL_model(base, input_shape):
    tf.keras.backend.clear_session()
    model = base(weights='imagenet', input_shape=(input_shape[0], input_shape[1], 3), include_top=False)
    model.trainable = False
    x = model.output
    x = GlobalAveragePooling2D()(x)
    x = Dense(120, activation='softmax')(x)
    return Model(model.input, x)


def convBlock(pInput, size):
    x = pInput
    x = Conv2D(filters=size, kernel_size=3, activation='relu', kernel_initializer='he_normal')(x)
    x = BatchNormalization()(x)
    x = MaxPooling2D(pool_size=(2, 2))(x)
    return x

def createModel(placeholder, outputSize, seed=None):
    tf.keras.backend.clear_session()
    if seed is not None:
       np.random.seed(seed)
       tf.random.set_seed(seed)

    x = placeholder
    x = data_augmentation(x)
    #x = SaveImageLayer(output_dir='intermediate_baseline_train')(x)
    x = convBlock(x, 16)
    x = convBlock(x, 32)
    x = convBlock(x, 64)
    x = convBlock(x, 128)
    x = convBlock(x, 256)
    x = GlobalAveragePooling2D()(x)
    x = Dropout(0.4)(x)
    x = Dense(outputSize, activation='softmax')(x)  
    return x

def concatModel(placeholder, outputSize, sourceModel=None, layer=None, threshold=None,  preprocess_input=None, seed=None):
    tf.keras.backend.clear_session()
    if seed is not None:
       np.random.seed(seed)
       tf.random.set_seed(seed)

    tlcam_output = ScoreCAM(layer, threshold)(placeholder, sourceModel, preprocess_input=preprocess_input)

    #x = SaveImageLayer(output_dir='intermediate_concat_train_original')(placeholder, eval=False)
    #placeholder = x

    task1 = data_augmentation(placeholder)
    task1 = convBlock(task1, 16)
    task1 = convBlock(task1, 32)
    task1 = convBlock(task1, 64)
    task1 = convBlock(task1, 128)
    task1 = convBlock(task1, 256)

    #x = SaveImageLayer(output_dir='intermediate_concat_train')(tlcam_output, eval=False)
    #tlcam_output = x

    task2 = data_augmentation(tlcam_output)
    task2 = convBlock(task2, 16)
    task2 = convBlock(task2, 32)
    task2 = convBlock(task2, 64)
    task2 = convBlock(task2, 128)
    task2 = convBlock(task2, 256)

    concat = Concatenate(axis=-1)([task1, task2])
    x = GlobalAveragePooling2D()(concat)
    x = Dropout(0.4)(x)
    x = Dense(outputSize, activation='softmax')(x)  
    return x

originalInput = Input([224, 224, 3], name='OriginalInput')
camInput      = Input([224, 224, 3], name='CAMInput')

baseline      = Model(inputs=originalInput, outputs=createModel(originalInput, 120, seed=seed_value))
concat_cam_k5 = Model(inputs=originalInput, outputs=concatModel(originalInput, 120, sourceModel=sourceModel, layer=-1, threshold=5, preprocess_input=preprocess_input_ResNet152V2, seed=seed_value))
cam_k5        = Model(inputs=camInput, outputs=createModel(camInput, 120, seed=seed_value))
resnet_tl     = create_TL_model(ResNet152V2, [224, 224])
inception_tl  = create_TL_model(InceptionV3, [299, 299])


numberOfEpochs  = 100
loss        = 'categorical_crossentropy'
metrics         = ['accuracy',                tf.keras.metrics.Precision(),               tf.keras.metrics.Recall(),                  tf.keras.metrics.FalsePositives(),                tf.keras.metrics.FalseNegatives(),                tf.keras.metrics.TruePositives(),                 tf.keras.metrics.TrueNegatives(),                 tf.keras.metrics.TopKCategoricalAccuracy(k=3)]

models_dict = {
    'baseline':baseline,
    'cam_k5':cam_k5,
    'concat_cam_k5':concat_cam_k5,
    'inception_tl':inception_tl,
    'resnet_tl':resnet_tl,
}


for _, model in models_dict.items():
   model.summary()


check_initial_weights(models = [baseline, cam_k5])
check_initial_weights(models = [baseline, cam_k5, concat_cam_k5])


for model_name, model in models_dict.items():
    outputDir='./'+model_name
    if os.path.exists(outputDir):
       os.rmdir(outputDir)

    os.mkdir(outputDir)

    if model_name in ['baseline']:
       batchSize = 8
       x_train, x_val, x_test = create_generators(dir+'train/', [224, 224], batch_size=batchSize, preprocess_input=None, seed=seed_value)
       layerGradCam = get_conv_layer_name(model, -1)
       optimizer    = Adam(lr = LR)
       reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
       es_callback  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    if model_name == 'resnet_tl':
       batchSize = 32
       x_train, x_val, x_test = create_generators(dir+'train/', [224, 224], batch_size=batchSize, preprocess_input=None, seed=seed_value)
       layerGradCam = get_conv_layer_name(model, -1)
       optimizer    = 'adam'
       reduce_lr    = []
       es_callback  = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if model_name == 'inception_tl':
       batchSize = 32
       x_train, x_val, x_test = create_generators(dir+'train/', [299, 299], batch_size=batchSize, preprocess_input=None, seed=seed_value)
       layerGradCam = get_conv_layer_name(model, -1)
       optimizer    = 'adam'
       reduce_lr    = []
       es_callback  = EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True)

    if model_name == 'cam_k5':
       batchSize = 8
       x_train, x_val, x_test = create_generators(cam_dir, [224, 224], batch_size=batchSize, preprocess_input=None, seed=seed_value)
       layerGradCam = get_conv_layer_name(model, -1) 
       optimizer    = Adam(lr = LR)
       reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
       es_callback  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
   
    if model_name ==  'concat_cam_k5':
       batchSize = 8
       x_train, x_val, x_test = create_generators(dir+'train/', [224, 224], batch_size=batchSize, preprocess_input=None, seed=seed_value)
       layerGradCam = get_conv_layer_name(model, -1)
       optimizer    = Adam(lr = LR)
       reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=5)
       es_callback  = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)

    log_dir = outputDir+'/logs_'+model_name+'/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)

    with file_writer.as_default():
      tf.summary.image("Training data", next(x_train)[0], max_outputs=8, step=0)

    with file_writer.as_default():
      tf.summary.image("Validation data", get_vis_data(x_val, 0)[0], max_outputs=8, step=0)

    with file_writer.as_default():
      tf.summary.image("Test data", next(x_test)[0], max_outputs=8, step=0)

    vis = [
          GradCAMCallback(
                    validation_data=get_vis_data(x_val, 99),
                    class_index=99,
                    layer_name = layerGradCam,
                    output_dir=log_dir,)]

    if model_name in ['baseline', 'cam_k5', 'resnet_tl', 'inception_tl']:
       tensorboard_callback =  TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=1)
       callbacks_train = [tensorboard_callback, vis, es_callback, reduce_lr]
       callbacks_test = []

    else:
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=True, update_freq='epoch', embeddings_freq=0)
        callbacks_train = [Callbacks(), tensorboard_callback, vis, es_callback, reduce_lr]
        callbacks_test  = [Callbacks()]


    compileNtrain(optimizer, loss, metrics, model, model_name, [x_train], [x_val], numberOfEpochs, outputDir, model_name + '_model.h5', callbacks=callbacks_train)
    test(model, model_name, [x_test], outputDir, callbacks=callbacks_test)

    #x, y = next(x_train)
    #model.compile(optimizer, loss, metrics, run_eagerly=True)
    #model.fit(x, y, epochs=1)
    #model.fit(x, y, epochs=1, validation_data=x_val)
    #xt,yt = next(x_test)
    #model.evaluate(xt, yt, callbacks=callbacks_test)

