import os
import tensorflow as tf
import sys
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tf_keras_vis.utils.model_modifiers import ExtractIntermediateLayerForGradcam as ModelModifier
from scipy.ndimage import zoom
from tf_keras_vis.utils import normalize, zoom_factor
import tensorflow.keras.backend as K
import random


@tf.keras.utils.register_keras_serializable()
class ScoreCAM(keras.layers.Layer):

    def clear_session(self):
       tf.keras.backend.clear_session()

    # Saving inputs and outputs of TLCAM layer during training
    def saveFigs(self, images, cams, fm, th, layer):
        random.seed(42)
        path = './savedTrainingFigs'+str(th)+'_'+str(layer)
        if not os.path.exists(path):
           os.makedirs(path)


        for i in range(fm.shape[0]):

            id = str(i)

            # Saving original images
            orig = plt
            orig.imshow(images[i])
            orig.axis('off')
            print('\nSAVING INPUTS TO TLCAM LAYER\n')
            orig.savefig(path+'/TLCAM_input'+id+'.png')
            orig.close()

            # Saving generated Class Activation Maps
            cam = plt
            cam.imshow(images[i])
            cam.imshow(cams[i], cmap='jet', alpha=0.5)
            cam.axis('off')
            print('\nSAVING GENERATED CAMs\n')
            cam.savefig(path+'/CAM'+id+'.png')
            cam.close()

            # Saving generated transformed inputs
            inpt = plt
            inpt.imshow(fm[i])
            inpt.axis('off')
            print('\nSAVING TRANSFORMED INPUTS\n')
            inpt.savefig(path+'/TLCAM_output'+id+'.png')
            inpt.close()

    def get_scores(self, seed_inputs, model):
        y_pred  = model(seed_inputs, training=False)
        scores = tf.argmax(y_pred, axis=1)
        return scores


    def __init__(self, penultimate_layer=None, threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.clear_session()
        self.penultimate_layer = penultimate_layer
        self.threshold = threshold


    def get_config(self):
        config = {
            "threshold": self.threshold,
        }
        base_config = super().get_config()
        return dict(list(base_config.items()) + list(config.items()))    

    @tf.autograph.experimental.do_not_convert
    def call(self,
             seed_inputs,
             model=None,
             seek_penultimate_conv_layer=True,
             activation_modifier=lambda cam: K.relu(cam),
             preprocess_input=None,
             training=False):

        if training in [None, False]:
           return seed_inputs

        else:
            threshold          = self.threshold
            penultimate_layer  = self.penultimate_layer

            print('TRAINING WITH THRESHOLD = ', threshold)
            print('TRAINING WITH LAYER= ', penultimate_layer)
            print('TRAINING WITH model= ', model.name)

           # Creating model where output layer is last conv layer of pre-trained model
            CAModel = ModelModifier(penultimate_layer, seek_penultimate_conv_layer, False)(model)
            inputs = preprocess_input(seed_inputs * 255) if preprocess_input is not None else seed_inputs
            scores = self.get_scores(inputs, model)

            # Feeding input to new model to extract feature maps of last conv layer
            penultimate_output = CAModel(inputs, training=False)
            #add this line for cifar-100
            #penultimate_output = tf.math.scalar_mul(-1, penultimate_output)


            activation_variances = tf.math.reduce_variance(penultimate_output,
                                                 axis=tf.range(start=1,
                                                               limit=(tf.keras.backend.ndim(penultimate_output)-1),
                                                               delta=1), keepdims=True)


            top_k_var, top_k_indices = tf.math.top_k(activation_variances, threshold)
            top_k_indices   = [tf.reshape(index, (-1)) for index in top_k_indices]
            penultimate_outputs = [tf.gather(output, index, axis=-1) for output, index in zip(penultimate_output, top_k_indices)]
            penultimate_output  = tf.stack(penultimate_outputs, axis=0)
            #print("penultimate_output", penultimate_output.shape)


            nsamples = penultimate_output.shape[0]
            channels = penultimate_output.shape[-1]


            #print("nsamples", nsamples)
            #print("channels", channels)



            #Upsampling activations

            zoom_factors = zoom_factor(penultimate_output.shape[1:-1], seed_inputs.shape[1:-1])
            zoom_factors = (1, ) + zoom_factors + (1, )
            upsampled_activations = zoom(penultimate_output, zoom_factors, order=1, mode='nearest')

            #print("upsampled_activations", upsampled_activations[1].shape)
            #print("upsampled_activations [0]", upsampled_activations[0])



            #Normalizing activations

            min_activations = [tf.math.reduce_min(upsampled_activation,
                                               axis=tuple(range(upsampled_activation.ndim)[1:-1]),
                                               keepdims=True) for upsampled_activation in upsampled_activations]



            max_activations = [tf.math.reduce_max(upsampled_activation,
                                               axis=tuple(range(upsampled_activation.ndim)[1:-1]),
                                               keepdims=True) for upsampled_activation in upsampled_activations]


            normalized_activations = [tf.math.divide_no_nan(tf.subtract(upsampled_activation, min_activation),
                                            tf.subtract(
                                                max_activation,
                                                min_activation)) for upsampled_activation, min_activation, max_activation in zip(upsampled_activations, min_activations, max_activations)]


            # Obtain masked images

            input_imgs = tf.unstack(seed_inputs, axis=0)
            feature_maps = [tf.unstack(normalized_activation, axis=-1) for normalized_activation in normalized_activations]

            masked_seed_inputs = []
            for img, f_map in zip(input_imgs, feature_maps):
                masked_inputs = []
                for normalized in f_map:
                    masked_inputs.append(tf.math.multiply(tf.expand_dims(normalized, axis=-1), tf.cast(img, tf.float32)))
                masked_seed_inputs.append(tf.stack(masked_inputs, axis=0))



            #Predicting masked seed-inputs

            preds = [model(masked_inputs, training=False) for masked_inputs in masked_seed_inputs]



             # Extracting weights


            fm_weights = [tf.gather(prediction, [score], axis=-1) for prediction, score in zip(preds, scores)]
            fm_weights = tf.stack(fm_weights, axis=0)


            # Generate cam


            class_act_maps = K.batch_dot(upsampled_activations, fm_weights)

            if activation_modifier is not None:
                class_act_maps = activation_modifier(class_act_maps)



            _output_ = tf.math.multiply(class_act_maps, tf.cast(seed_inputs, tf.float32))
            _output_ = normalize(_output_)
           # self.saveFigs(seed_inputs, class_act_maps, _output_, threshold, penultimate_layer)

            return _output_




class Callbacks(keras.callbacks.Callback):

    def on_train_batch_begin(self, batch, logs=None):
        tf.config.run_functions_eagerly(True)


    def on_train_batch_end(self, batch, logs=None):
        tf.config.run_functions_eagerly(False)



