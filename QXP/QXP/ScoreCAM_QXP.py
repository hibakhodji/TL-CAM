import numpy as np
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.models import Model
from scipy.ndimage.interpolation import zoom
from tf_keras_vis.utils import normalize, zoom_factor
import tensorflow.keras.backend as K
from Quantitative_eXplainability import QXP

class ScoreCAM_QXP:

    def get_quantitative_explanation(self, scores, predctions, labels, shape):
        QXP(scores, predctions, labels, shape).explain()
        
        
    def get_scores(self, seed_inputs, model):
        y_pred  = model(seed_inputs, training=False)
        scores = tf.argmax(y_pred, axis=-1)
        return scores


    def __init__(self, model=None, penultimate_layer=None, threshold=None, **kwargs):
        super().__init__(**kwargs)
        self.model=model
        self.penultimate_layer=penultimate_layer
        self.threshold = threshold

  

    def __call__(self,
                 seed_inputs,
                 scores=None,
                 seek_penultimate_conv_layer=True,
                 quantitative_explanation = True,
                 labels = None,
                 activation_modifier=True,
                 training = False):
        
        model=self.model
        penultimate_layer=self.penultimate_layer
        threshold = self.threshold

        if quantitative_explanation == True and labels is None:
            raise Exception("Please provide a list of labels for the Quantitative eXPlanation")
            
        if scores is None: scores = self.get_scores(seed_inputs, model)

        # Creating model where output layer is last conv layer of pre-trained model
        
        intermediate_layer_model = Model(inputs=model.input, outputs=model.get_layer(penultimate_layer).output)
        #intermediate_layer_model.summary()
        penultimate_output = intermediate_layer_model(seed_inputs, training=False)
        #add this line for CIFAR-100
        #penultimate_output = tf.math.scalar_mul(-1, penultimate_output)

        if threshold is not None:
            
            activation_variances = tf.math.reduce_variance(penultimate_output,
                                                     axis=tf.range(start=1,
                                                                   limit=(tf.keras.backend.ndim(penultimate_output)-1),
                                                                   delta=1), keepdims=True)


            top_k_var, top_k_indices = tf.math.top_k(activation_variances, threshold)
            #print('top_k_var, top_k_indices', top_k_var, top_k_indices)
            top_k_indices= [tf.reshape(index, (-1)) for index in top_k_indices]
            #print('top_k_indices var', top_k_indices)
            penultimate_outputs = [tf.gather(output, index, axis=-1) for output, index in zip(penultimate_output, top_k_indices)]
            penultimate_output  = tf.stack(penultimate_outputs, axis=0)
            
       
        nsamples = penultimate_output.shape[0]
        channels = penultimate_output.shape[-1]


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
        #print('preds shape ------>', np.shape(preds))
       
        # Extracting weights


        fm_weights = [tf.gather(prediction, score, axis=-1) for prediction, score in zip(preds, scores)]
        fm_weights = tf.stack(fm_weights, axis=0)
        #print('fm_weights -------------->', fm_weights)
        


        # Generate cam


        class_act_maps = K.batch_dot(upsampled_activations, fm_weights)

        if activation_modifier:
            activation_modifier = lambda cam: K.relu(cam)
            class_act_maps = activation_modifier(class_act_maps)

        class_act_maps = normalize(class_act_maps) 
        
        
        if quantitative_explanation == True and labels is not None:
            
            for score, prediction in zip(scores, preds):
                self.get_quantitative_explanation(score, prediction, labels, penultimate_output.shape[3])

        return class_act_maps, masked_seed_inputs
