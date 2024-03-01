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
        print('predictions.shape', predictions)
        print('input_image.shape', input_image.shape)
        pred_vector = model(tf.expand_dims(input_image, axis=0), training=False)[0]
        max_preds = tf.argmax(predictions, axis=-1)
        print('max_preds', max_preds)
        print('max_preds shape', max_preds.shape)
        unique_max_preds = set(max_preds.numpy())
        qxp_len = len(unique_max_preds)

        print('sorted_pred_vector_top_indices = (tf.argsort(pred_vector, direction=DESCENDING).numpy())[:qxp_len]')        
        sorted_pred_vector_top_indices = (tf.argsort(pred_vector, direction='DESCENDING').numpy())[:qxp_len]
        print('sorted_pred_vector_top_indices', sorted_pred_vector_top_indices)   
        print("unique_max_preds set:", unique_max_preds)
        distance= self.jaccard_distance(sorted_pred_vector_top_indices, unique_max_preds)
        print('distance--->', distance)
        return distance, qxp_len
        
    
    
    def kl_loss_features(self, ground_truth, predictions, nb_channels):
        print('predictions.shape', predictions)
        print('nb_channels', nb_channels)
        max_preds = tf.argmax(predictions, axis=-1)
        print('max_preds shape', max_preds.shape)
        print('max_preds', max_preds)
        unique_max_preds = set(max_preds.numpy())
        print("unique_max_preds set:", unique_max_preds)
        nbpreds = [tf.unstack(max_preds).count(i) for i in unique_max_preds]
        print('nbpreds:', nbpreds)  
        print('predictions.shape[-1]', predictions.shape[-1])        
        qxp = [0 for element in range(predictions.shape[-1])]
        for index, value in zip(unique_max_preds, nbpreds):
            qxp[int(index)] = float(value/nb_channels)

        print('ground_truth -->', ground_truth)
        print('type(ground_truth)--->', type(ground_truth))
        print('index grouns truth--->', np.where(ground_truth==1))
        print('sum gt ---->', np.sum(ground_truth))
        print("qxp---------->", qxp)
        print('sum qxp --------->', sum(qxp))
        print('len(qxp)', len(qxp))
        print('len(ground_truth)', len(ground_truth))
        print('KL---->', sum(rel_entr(ground_truth, qxp)))
        return sum(rel_entr(ground_truth, qxp))
        

        

    def __call__(self,
                 seed_inputs,
                 model=None,
                 penultimate_layer=None,
                 threshold=None,
                 ground_truths=None,
                 compute_kl = True,
                 training = False):


        print('layer used ---->', penultimate_layer)
        # Extract feature maps of selected layer
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(penultimate_layer).output)
        intermediate_layer_model.summary()
        penultimate_output = intermediate_layer_model(seed_inputs, training=False)
        #add this line for cifar-100
        #penultimate_output = tf.math.scalar_mul(-1, penultimate_output)


        if threshold is not None:
           print('activation_variances = tf.math.reduce_variance(penultimate_output')            
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
        print('penultimate_output.shape', penultimate_output.shape)
        print('channels', channels)
        print('samples', nsamples)

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
                
        print('masked_seed_inputs', np.shape(masked_seed_inputs))
        for m in masked_seed_inputs:
            print('masked shape', m.shape)
       #Predicting masked seed-inputs
       
        preds = [model(masked_inputs, training=training) for masked_inputs in masked_seed_inputs]
        print('predsh shape', np.shape(preds))
        for pred in preds:
            print('pred shape', pred.shape)
                
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
                avg_target_features.append(self.kl_loss_features(ground_truth, prediction, prediction.shape[0]))

        if not compute_kl : return avg_jaccard_distance, avg_nb_qxp_preds
        print('avg_jaccard_distance', avg_jaccard_distance)
        print('avg_nb_qxp_preds', avg_nb_qxp_preds)
        print('avg_target_features', avg_target_features) 
        return avg_target_features, avg_jaccard_distance, avg_nb_qxp_preds

