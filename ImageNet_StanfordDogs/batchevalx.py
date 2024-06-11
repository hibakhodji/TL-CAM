import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import tensorflow as tf
import sys
sys.path.append('../TL-CAM')
sys.path.append('../EvalX')
from computeEvalX import EvalX
import numpy as np
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
import random
from functools import reduce
from tlcam_layer import ScoreCAM 
from utils import get_conv_layer_name
import re
import math
import datetime

seed_value = 42
np.random.seed(seed_value)
tf.random.set_seed(seed_value)

#mn       = 53
#mx       = 70


modelpath = './'

def _inf_(input_list):

    if all(math.isinf(elem) for elem in input_list): return float('inf')
    max_value = max([elem for elem in input_list if not math.isinf(elem)])
    return [max_value if math.isinf(elem) else elem for elem in input_list]

def batch_processing(data, model, steps, threshold, ground_truths, layer):
    avg_kl    = []
    avg_jd    = []
    avg_preds = []

    for i in range(0, len(data), steps): 
        batch_x = data[i:i+steps]
        batch_y = ground_truths[i:i+steps]
        kl, jd, preds = EvalX()(batch_x, model=model, penultimate_layer=layer, threshold=threshold, ground_truths=batch_y)
        avg_kl.extend(kl)
        avg_jd.extend(jd)
        avg_preds.extend(preds)

    max_value = max([elem for elem in avg_kl if not math.isinf(elem)])
    print('----------------------------------------------------------------------------------------------')
    print('avg_kl', avg_kl)
    print('_inf_(avg_kl)', _inf_(avg_kl))
    print('Max value in avg_kl', max_value)
    print('len avg_kl', len(avg_kl))
    print('len avg_jd', len(avg_jd))
    print('len avg_preds', len(avg_preds))
    print('tf.math.reduce_mean(_inf_(avg_kl))', tf.math.reduce_mean(_inf_(avg_kl)))
    print('tf.math.reduce_mean(avg_jd)', tf.math.reduce_mean(avg_jd))
    print('tf.math.reduce_mean(avg_preds)', tf.math.reduce_mean(avg_preds))
    return tf.math.reduce_mean(_inf_(avg_kl)), tf.math.reduce_mean(avg_jd), tf.math.reduce_mean(avg_preds), tf.reduce_sum(tf.cast(tf.math.is_inf(avg_kl), tf.int32)).numpy(), max_value


print("Loading data")
test_datagen  = ImageDataGenerator(rescale=1/255)
test_generator = test_datagen.flow_from_directory(
        'StanfordDogs/test/',
        batch_size=2153,
        target_size=(224, 224),
        class_mode = 'sparse', 
        shuffle=False,
        seed=seed_value
)

x_test, y_test = next(test_generator)
print(x_test.shape)



print("Loading baseline model")
baseline = load_model(modelpath+"baseline/baseline_model.h5")
baseline.summary()
#print(baseline.evaluate(x_test, y_test, batch_size=8))


print("Loading TL-CAM CONCAT model with k=", 5)
tlcam_model_concat = load_model(modelpath+"concat_cam_k5/concat_cam_k5_model.h5")
tlcam_model_concat.summary()
#print(tlcam_model_concat.evaluate(x_test, y_test, batch_size=8))


print("Loading ResNet model")
resnet = load_model(modelpath+"resnet_tl/resnet_tl_model.h5")
resnet.summary()
#print(resnet.evaluate(x_test, y_test, batch_size=32))


print("Extracting correct predictions by baseline for EvalX")
predictions_baseline = tf.argmax(baseline.predict(x_test), axis=-1)
indices_correct_preds_baseline = [i for i, pred in enumerate(predictions_baseline) if pred==y_test[i]]


print("Extracting correct predictions by TL-CAM CONCAT for EvalX with k=", 5)
predictions_tlcam = tf.argmax(tlcam_model_concat.predict(x_test), axis=-1)
indices_correct_preds_tlcam = [i for i, pred in enumerate(predictions_tlcam) if pred==y_test[i]]


print("Extracting correct predictions by ResNet for EvalX")
predictions_tl = tf.argmax(resnet.predict(x_test), axis=-1)
indices_correct_preds_tl = [i for i, pred in enumerate(predictions_tl) if pred==y_test[i]]


print("Creating test set for EvalX")
x_test_evalx       = np.array([x_test[i] for i in reduce(np.intersect1d, (indices_correct_preds_baseline, indices_correct_preds_tlcam, indices_correct_preds_tl))])
y_test_evalx       = np.array([y_test[i] for i in reduce(np.intersect1d, (indices_correct_preds_baseline, indices_correct_preds_tlcam, indices_correct_preds_tl))])
print("Done")


print('x_test_evalx', x_test_evalx.shape)


subset_x = x_test_evalx
subset_y = y_test_evalx

print("Checking if models' predictions are the same on a subset of created test set for EvalX")
print(tf.argmax(baseline.predict(subset_x), axis=-1))
print(tf.argmax(tlcam_model_concat.predict(subset_x), axis=-1))
print(tf.argmax(resnet.predict(subset_x), axis=-1))
print("Completed")

b = list(tf.argmax(baseline.predict(subset_x), axis=-1))
c = list(tf.argmax(tlcam_model_concat.predict(subset_x), axis=-1))
r = list(tf.argmax(resnet.predict(subset_x), axis=-1))

print('checking if predictions are the same with lists --->', b==c==r)

models_dict = {
    'baseline':baseline,
    'concat_k5':tlcam_model_concat,
#    'resnet_model_1024':resnet,
#    'resnet_model_256':resnet,
    'resnet_model_128':resnet, 
}

print('subset_x', subset_x.shape)
print('subset_y:', subset_y)


subset_y_enc = tf.keras.utils.to_categorical(subset_y, 120)
print('subset_y_enc[:10]', subset_y_enc[:10])

print('subset_y_enc shape', subset_y_enc.shape)

for model_name, model in models_dict.items():

    threshold = 1024 if model_name == 'resnet_model_1024' else (256 if model_name == 'resnet_model_256' else None)
    if model_name == 'resnet_model_128': threshold = 128

    avg_kl, avg_jaccard_distance, avg_qxp_preds, nbinf, maxvaluetf = batch_processing(subset_x, model, 2, threshold, subset_y_enc, get_conv_layer_name(model, -1))
    end_time = datetime.datetime.now()

    file_path = f"{model_name}_QXP_metrics.txt"
    with open(file_path, 'w') as file:
        file.write(f"EvalX Results for {model_name}:\n")
        file.write(f"Average loss of target-specific features: {avg_kl}\n")
        file.write(f"Nb of inf values: {nbinf}\n")
        file.write(f"Max value: {maxvaluetf}\n")
        file.write(f"Avg Jaccard Distance: {avg_jaccard_distance}\n")
        file.write(f"Avg QXP preds: {avg_qxp_preds}\n")
        file.write(f"End Time: {end_time}\n")           

