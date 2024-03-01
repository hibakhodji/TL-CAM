import numpy as np
import tensorflow as tf
import os

def check_initial_weights(models):
    all_weights_match = True
    weights_first_model = models[0].get_weights()

    for i in range(1, len(models)):
        weights_current_model = models[i].get_weights()
        if not all(np.array_equal(w1, w2) for w1, w2 in zip(weights_first_model, weights_current_model)):
           all_weights_match = False
           break

    if all_weights_match:
       print("The initial weights of all models are the same.")
    else:
      print("The initial weights of at least one model are different.")



def compileNtrain(optimizer, loss, metrics, model, modelName, trainData, validData, nbEpochs, path, filename, batchSize=None, callbacks=[]): 

    print('Compiling ...')
    model.compile(optimizer, loss, metrics)
    print("Training model ...")

    if len(trainData) ==2 and len(validData)==2:
       x_train, y_train = trainData
       x_valid, y_valid = validData
       history=model.fit(x_train, y_train, batch_size=batchSize, epochs=nbEpochs, callbacks=callbacks, validation_data=(x_valid, y_valid))

    elif len(trainData) == 1 and len(validData)==1:
         train_generator = trainData[0]
         valid_generator = validData[0]
         history=model.fit(train_generator, batch_size=batchSize, epochs=nbEpochs, callbacks=callbacks, validation_data=valid_generator)

    weights = path+'/'+modelName+'_weights.h5'
    np.save(path+'/'+modelName+'_history.npy',history.history)
    print("Saving model parameters")
    model.save_weights(weights)
    model.save(path+'/'+filename)



def test(model, modelName, testData, path, batchSize=None, callbacks=[]):
    print("Testing model...")
    if len(testData) ==2: 
       x_test, y_test = testData
       metrics = model.evaluate(x_test, y_test, batch_size=batchSize, callbacks=callbacks)

    elif len(testData)==1:
         test_generator=testData[0]
         metrics = model.evaluate(test_generator, batch_size=batchSize, callbacks=callbacks)

    f = open(path+'/metrics_'+str(modelName)+'.txt', 'w')
    f.write(str(metrics)+'\n')
    f.close()
    print("Final testing accuracy :" +str(metrics[1])+"\n")
    return metrics[1] # accuracy


def evaluate(model, modelName, testData, path, batchSize=None, callbacks=[]):
    print("Evaluating model...")
    if len(testData) ==2: 
       x_test, y_test = testData
       metrics = model.evaluate(x_test, y_test, batch_size=batchSize, callbacks=callbacks)

    elif len(testData)==1:
         test_generator=testData[0]
         metrics = model.evaluate(test_generator, batch_size=batchSize, callbacks=callbacks)

    f = open(path+'/eval_metrics_'+str(modelName)+'.txt', 'w')
    f.write(str(metrics)+'\n')
    f.close()
    print("Final testing accuracy :" +str(metrics[1])+"\n")
    return metrics[1] # accuracy


def get_last_conv_layer_name(model):
    for layer in reversed(model.layers):
        if isinstance(layer, tf.keras.layers.Conv2D):
            return layer.name
    return None

def get_conv_layer_name(model, index):
    if index == -1:
       return get_last_conv_layer_name(model)

    count_conv_layers = 0
    for layer in model.layers:
        if isinstance(layer, tf.keras.layers.Conv2D):
            count_conv_layers += 1
            if count_conv_layers == index:
                return layer.name




class SaveImageLayer(tf.keras.layers.Layer):

    def __init__(self, output_dir, image_name_prefix="image_", **kwargs):
        super(SaveImageLayer, self).__init__(**kwargs)
        self.output_dir = output_dir
        self.image_name_prefix = image_name_prefix

    @tf.autograph.experimental.do_not_convert
    def call(self, inputs, training=None, eval=True):

        if training in [None, eval]: return inputs

        for i, image in enumerate(inputs):
            image_name = f"{self.image_name_prefix}{i}.png"
            image_path = os.path.join(self.output_dir, image_name)
            tf.keras.preprocessing.image.save_img(image_path, image)
        return inputs  

    def get_config(self):
        config = super(SaveImageLayer, self).get_config()
        config.update({
            'output_dir': self.output_dir,
            'image_name_prefix': self.image_name_prefix,
        })
        return config
