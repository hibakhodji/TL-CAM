import os
os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"
os.environ["CUDA_VISIBLE_DEVICES"] = " "
import sys
import tensorflow as tf
sys.path.append('../EvalX')
from computeEvalX import EvalX
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import load_model
from functools import reduce
sys.path.append('../TL-CAM')
from tlcam_layer import ScoreCAM
from utils import get_conv_layer_name
import math
import datetime


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

(_, _), (x_test, y_test)=tf.keras.datasets.cifar100.load_data()

print("Normalizing ...")
x_test = x_test / 255


print("Loading baseline model")
baseline = load_model(modelpath+"baseline/baseline_model.h5")
#baseline.summary()
print(baseline.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 100)))

print("Loading TL-CAM model")
tlcam_model = load_model(modelpath+"cam_k50/cam_k50_model.h5")
#tlcam_model.summary()
print(tlcam_model.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 100)))


print("Loading TL model")
tl = load_model(modelpath+"tl_model/tl_model_model.h5")
#tl.summary()
print(tl.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 100)))

print("Loading TL model conv")
tlconv = load_model(modelpath+"tl_model_conv/tl_model_model.h5")
#tlconv.summary()  
#print(tlconv.evaluate(x_test, tf.keras.utils.to_categorical(y_test, 100)))


print('x_test shape', x_test.shape)


print("Extracting correct predictions by baseline for EvalX")
predictions_baseline = tf.argmax(baseline.predict(x_test), axis=-1)
indices_correct_preds_baseline = [i for i, pred in enumerate(predictions_baseline) if pred==y_test[i]]


print("Extracting correct predictions by TL-CAM for EvalX with k=", 50)
predictions_tlcam = tf.argmax(tlcam_model.predict(x_test), axis=-1)
indices_correct_preds_tlcam = [i for i, pred in enumerate(predictions_tlcam) if pred==y_test[i]]


print("Extracting correct predictions by TL for EvalX")
predictions_tl = tf.argmax(tl.predict(x_test), axis=-1)
indices_correct_preds_tl = [i for i, pred in enumerate(predictions_tl) if pred==y_test[i]]


print("Extracting correct predictions by TL conv for EvalX")
predictions_tlconv = tf.argmax(tlconv.predict(x_test), axis=-1)
indices_correct_preds_tlconv = [i for i, pred in enumerate(predictions_tlconv) if pred==y_test[i]]



print("Creating test set for EvalX")
x_test_evalx = np.array([x_test[i] for i in reduce(np.intersect1d, (indices_correct_preds_baseline, indices_correct_preds_tlcam, indices_correct_preds_tl, indices_correct_preds_tlconv))])
y_test_evalx = np.array([y_test[i] for i in reduce(np.intersect1d, (indices_correct_preds_baseline, indices_correct_preds_tlcam, indices_correct_preds_tl, indices_correct_preds_tlconv))])
print("Done")


print('x_test_evalx', x_test_evalx.shape)


subset_x = x_test_evalx
subset_y = y_test_evalx

print("Checking if models' predictions are the same on a subset of created test set for EvalX")
print(tf.argmax(baseline.predict(subset_x), axis=-1))
print(tf.argmax(tlcam_model.predict(subset_x), axis=-1))
print(tf.argmax(tl.predict(subset_x), axis=-1))
print(tf.argmax(tlconv.predict(subset_x), axis=-1))
print("Completed")

b = list(tf.argmax(baseline.predict(subset_x), axis=-1))
c = list(tf.argmax(tlcam_model.predict(subset_x), axis=-1))
t = list(tf.argmax(tl.predict(subset_x), axis=-1))


print('checking if predictions are the same ----->', b==c==t)
ok=[]
for i,j,k in zip(b,c,t):
    #print(int(i),int(j),int(k))
    if not i==j==k: ok.append('not ok')
    if i==j==k: ok.append('ok')

print('ok.count(not ok)', ok.count('not ok'))
print('ok.count(ok)', ok.count('ok'))

models_dict = {
    'tl_model_64':tl,
    'baseline':baseline,
    'cam_k50':tlcam_model,
    'tl_model':tl,
}

print('subset_x', subset_x.shape)

print('subset_y[:20]:', subset_y[:20])
print('subset_y shape', subset_y.shape)

# Ground truth (One-hot encoding)
subset_y_enc = tf.keras.utils.to_categorical(subset_y, 100)

print('subset_y_enc[:20]', subset_y_enc[:20])


for model_name, model in models_dict.items():
    threshold = 64 if model_name == 'tl_model_64' else None

    print('------------------------------------ model:', model_name)
    avg_kl, avg_jaccard_distance, avg_qxp_preds, nbinf, maxvaluetf = batch_processing(subset_x, model, 8, threshold, subset_y_enc, get_conv_layer_name(model, -1))
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


