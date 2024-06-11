import os
import tensorflow as tf
import sys
from tensorflow import keras
import numpy as np
from scipy.ndimage.interpolation import zoom
from tf_keras_vis.utils import normalize, zoom_factor
import tensorflow.keras.backend as K
from scipy.special import rel_entr
from tensorflow.keras.models import Model
import math



class EvalX:
    
    def jaccard_distance(self, preds, qxp):

        jaccard = len(set(preds).intersection(qxp)) / len(set(preds).union(qxp))
        return 1-jaccard 
    


    def jd_explainability(self, input_image, model, predictions):
        max_preds = tf.argmax(predictions, axis=-1)
        unique_max_preds = set(max_preds.numpy())
        qxp_len = len(unique_max_preds)
        sorted_pred_vector_top_indices = (tf.argsort(pred_vector, direction='DESCENDING').numpy())[:qxp_len]
        distance= self.jaccard_distance(sorted_pred_vector_top_indices, unique_max_preds)
        return distance, qxp_len
        
    
    
    def kl_loss_features(self, ground_truth, predictions):
        nb_channels=predictions.shape[0]
        max_preds = tf.argmax(predictions, axis=-1)
        unique_max_preds = set(max_preds.numpy())
        nbpreds = [tf.unstack(max_preds).count(i) for i in unique_max_preds]
        qxp = [0 for element in range(predictions.shape[-1])]
        for index, value in zip(unique_max_preds, nbpreds):
            qxp[int(index)] = float(value/nb_channels)
        return sum(rel_entr(ground_truth, qxp))
        

        

    def __call__(self,
                 seed_inputs,
                 model=None,
                 penultimate_layer=None,
                 threshold=None,
                 ground_truths=None,
                 compute_kl = True,
                 training = False):


        # Extract feature maps of selected layer
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(penultimate_layer).output)
        intermediate_layer_model.summary()
        penultimate_output = intermediate_layer_model(seed_inputs, training=False)
        #add this line for cifar-100
        #penultimate_output = tf.math.scalar_mul(-1, penultimate_output)


        if threshold is not None:
           activation_variances = tf.math.reduce_variance(penultimate_output,
                                                     axis=tf.range(start=1,
                                                                   limit=(tf.keras.backend.ndim(penultimate_output)-1),
                                                                   delta=1), keepdims=True)


           top_k_var, top_k_indices = tf.math.top_k(activation_variances, threshold)
           print('top_k_var, top_k_indices', top_k_var, top_k_indices)
           top_k_indices   = [tf.reshape(index, (-1)) for index in top_k_indices]
           print('top_k_indices', top_k_indices)
           penultimate_outputs = [tf.gather(output, index, axis=-1) for output, index in zip(penultimate_output, top_k_indices)]
           penultimate_output  = tf.stack(penultimate_outputs, axis=0)


        nsamples = penultimate_output.shape[0]
        channels = penultimate_output.shape[-1]

        for p in penultimate_output:
            print('p.shape', p.shape)

        zoom_factors = zoom_factor(penultimate_output.shape[1:-1], seed_inputs.shape[1:-1])
        zoom_factors = (1, ) + zoom_factors + (1, )
        upsampled_activations = zoom(penultimate_output, zoom_factors, order=1, mode='nearest')

  

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
       
        preds = [model(masked_inputs, training=training) for masked_inputs in masked_seed_inputs]
                
        #Evaluation
        # Jaccard Distance
        avg_jaccard_distance = []
        avg_nb_qxp_preds = []
        for input_image, prediction in zip(seed_inputs, preds):
            avg_jd, avg_nb_qxp = self.jd_explainability(input_image, model, prediction) 
            avg_jaccard_distance.append(avg_jd)
            avg_nb_qxp_preds.append(avg_nb_qxp)
        
        # Average loss of target-specific features
        if compute_kl: 
            avg_target_features = []
            for ground_truth, prediction in zip(ground_truths, preds):
                avg_target_features.append(self.kl_loss_features(ground_truth, prediction))

        if not compute_kl : return avg_jaccard_distance, avg_nb_qxp_preds
        return avg_target_features, avg_jaccard_distance, avg_nb_qxp_preds

