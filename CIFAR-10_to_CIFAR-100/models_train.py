import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = "0, 1"
import tensorflow as tf
import sys
sys.path.append('../TLCAM')
from tensorflow import keras
from tensorflow.keras.models import Model, load_model
import numpy as np
from tensorflow.keras.layers import Reshape, Conv2D, Dense, UpSampling2D, Flatten, Input, BatchNormalization, Dropout, Activation, MaxPool2D, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from sklearn.model_selection import train_test_split
from tlcam_layer import ScoreCAM, Callbacks
from utils import check_initial_weights, compileNtrain, test, evaluate, get_conv_layer_name, SaveImageLayer
from tensorflow.keras.callbacks import EarlyStopping, TensorBoard
import datetime
from tf_explain.callbacks.grad_cam import GradCAMCallback

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

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

def create_TL_model(outputSize, sourceModel):
    sourceModel.trainable = False
    #print(sourceModel.layers[17].name)
    x = sourceModel.layers[17].output
    x = Conv2D(32, 3, activation='relu', name='conv2d_')(x)
    x = BatchNormalization(name = 'batch_norm_')(x)
    x = MaxPool2D(2, name='maxpool_', padding="same")(x)
    x = Flatten(name='flatten_')(x)
    x = Dense(512, name='dense_', activation='relu')(x)
    x = Dense(outputSize, activation='softmax', name='dense_output')(x)
    return Model(inputs=sourceModel.input,outputs=x)


def create_TL_model2(outputSize, sourceModel):
    sourceModel.trainable = False
    #print(sourceModel.layers[17].name)
    x = sourceModel.layers[17].output
    x = Activation('relu', name='activation_5')(x) 
    x = BatchNormalization(name = 'batch_norm_5')(x)
    x = MaxPool2D(2, name='maxpool_5', padding="same")(x)
    x = Dropout(0.3, name='dropout_5')(x)
    x = Flatten(name='flatten_')(x)
    x = Dense(100, name='dense_')(x)
    x = Activation('relu', name='activation_')(x)
    x = Dropout(0.5, name='dropout_')(x)
    x = Dense(outputSize, activation='softmax', name='dense_output')(x)
    return Model(inputs=sourceModel.input,outputs=x)



## Convblock
def convBlock(pInput, size):
    x = pInput
    x = Conv2D(size, 3, padding='same')(x)
    x = Activation('elu')(x)
    x = BatchNormalization()(x)
    x = MaxPool2D(2)(x)
    x = Dropout(0.2)(x)
    return x



def createModel(placeholder, outputSize, sourceModel=None, layer=None, threshold=None, add_ScoreCAM=False, seed=None):
    if seed is not None:
       np.random.seed(seed)
       tf.random.set_seed(seed)

    x = placeholder

    if add_ScoreCAM:
       x = ScoreCAM(layer, threshold)(x, sourceModel)

    #x = SaveImageLayer(output_dir='intermediate_')(x)
    x = convBlock(x, 16)
    x = convBlock(x, 32)
    x = convBlock(x, 64)
    x = Flatten()(x)
    x = Dense(100, activation='elu')(x)
    x = Dropout(0.3)(x)
    x = Dense(outputSize, activation='softmax')(x)
    return x


inputPlaceholderSource = Input([32, 32, 3], name='input')

#baseline= Model(inputs=inputPlaceholderSource, outputs=createModel(inputPlaceholderSource, 100, seed=seed_value))
#cam_k5  = Model(inputs=inputPlaceholderSource, outputs=createModel(inputPlaceholderSource, 100, sourceModel=sourceModel, layer='conv2d_4', threshold=5, add_ScoreCAM=True, seed=seed_value))
#cam_k25 = Model(inputs=inputPlaceholderSource, outputs=createModel(inputPlaceholderSource, 100, sourceModel=sourceModel, layer='conv2d_4', threshold=25, add_ScoreCAM=True, seed=seed_value))
#cam_k50 = Model(inputs=inputPlaceholderSource, outputs=createModel(inputPlaceholderSource, 100, sourceModel=sourceModel, layer='conv2d_4', threshold=50, add_ScoreCAM=True, seed=seed_value))
tl_model  = create_TL_model2(100, sourceModel=sourceModel)

numberOfEpochs  = 100
loss        = 'categorical_crossentropy'
metrics         = ['accuracy',                tf.keras.metrics.Precision(),               tf.keras.metrics.Recall(),                  tf.keras.metrics.FalsePositives(),                tf.keras.metrics.FalseNegatives(),                tf.keras.metrics.TruePositives(),                 tf.keras.metrics.TrueNegatives(),                 tf.keras.metrics.TopKCategoricalAccuracy(k=3)]
es_callback = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)




models_dict = {
    #'baseline':baseline,
    #'cam_k5':cam_k5,
    #'cam_k25':cam_k25,
    #'cam_k50':cam_k50,
    'tl_model':tl_model,
}


for _, model in models_dict.items():
   model.summary()

#check_initial_weights(models = [baseline, cam_k5, cam_k25, cam_k50])


x = x_valid[np.where(np.argmax(y_valid, axis=1) == 58)[0]]
y = y_valid[np.where(np.argmax(y_valid, axis=1) == 58)[0]]

for model_name, model in models_dict.items():
    outputDir='./'+model_name
    if os.path.exists(outputDir):
       os.rmdir(outputDir)

    os.mkdir(outputDir)

    log_dir = outputDir+'/logs_'+model_name+'/fit/' + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    file_writer = tf.summary.create_file_writer(log_dir)
    gradCAMcb = [GradCAMCallback(
                    validation_data=(x[0:9], y[0:9]),
                    class_index=58,
                    layer_name = get_conv_layer_name(model, -1),
                    output_dir=log_dir,)]

    with file_writer.as_default():
      tf.summary.image("Train data", x_train, max_outputs=8, step=0)

    with file_writer.as_default():
      tf.summary.image("Validation data", x, max_outputs=8, step=0)

    with file_writer.as_default():
      tf.summary.image("Test data", x_test, max_outputs=8, step=0)


    if model_name in ['baseline','tl_model']:
       tensorboard_callback =  TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=True, write_images=True, update_freq='epoch', embeddings_freq=1)
       callbacks_train = [es_callback, tensorboard_callback, gradCAMcb]
       callbacks_test  = []

    else: 
        tensorboard_callback = TensorBoard(log_dir=log_dir, histogram_freq=1, write_graph=False, write_images=True, update_freq='epoch', embeddings_freq=0)
        callbacks_train = [tensorboard_callback, Callbacks(), es_callback, gradCAMcb]
        callbacks_test  = [Callbacks()]

    if model_name in ['baseline','cam_k5', 'cam_k25', 'cam_k50']:
       batchSize = 16
       optimizer = Adam(1e-3, decay=1e-6)
    else:
       batchSize = 32
       optimizer = Adam(2e-4, decay=1e-6)

    compileNtrain(optimizer, loss, metrics, model, model_name, [x_train, y_train], [x_valid, y_valid], numberOfEpochs, outputDir, model_name + '_model.h5', batchSize, callbacks_train)
    test(model, model_name, [x_test, y_test], outputDir, batchSize, callbacks_test)


