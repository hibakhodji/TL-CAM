import os
import glob
import cv2
import tensorflow as tf
import sys
from tensorflow import keras
import numpy as np
import matplotlib.pyplot as plt
from tensorflow.keras.models import Model
from tqdm import tqdm 
from .tlcam_layer import ScoreCAM
from tensorflow.keras.applications.resnet_v2 import ResNet152V2
from tensorflow.keras.applications.resnet_v2 import preprocess_input as preprocess_input_ResNet152V2

class ImageDatasetProcessor:
    def __init__(self, input_generator, output_folder, sourceModel, layer, threshold, preprocess_input):
        self.input_generator = input_generator
        self.output_folder = output_folder
        self.sourceModel = sourceModel
        self.layer = layer
        self.threshold = threshold
        self.preprocess_input = preprocess_input

    def prepare(self, inputs):
        return tf.expand_dims(inputs, axis=0)

    def process_batch(self, batch):
        processed_batch = []
        for image in batch:
            processed_image = ScoreCAM(self.layer, self.threshold)(self.prepare(image), self.sourceModel, preprocess_input=self.preprocess_input, training=True)
            processed_batch.append(processed_image)
        return processed_batch

    def save_data(self, processed_data, filenames):

        for processed_image, original_image_path in zip(processed_data, filenames):
            #rel_input_path = os.path.relpath(original_image_path, self.input_generator.directory)
            output_path = os.path.join(self.output_folder, original_image_path)
            os.makedirs(os.path.dirname(output_path), exist_ok=True)
            plt.imshow(processed_image[0])
            plt.axis('off')
            plt.tight_layout(pad=0)
            plt.savefig(output_path, bbox_inches='tight', pad_inches = 0)
            plt.close()

    def process_dataset(self):
        num_batches = len(self.input_generator)
        for batch_num, batch in tqdm(enumerate(self.input_generator), desc="Processing batches", total=num_batches, unit="batch", ascii=True):
            processed_batch = self.process_batch(batch)
            idx_start = batch_num * self.input_generator.batch_size
            idx_end = idx_start + len(batch)
            self.save_data(processed_batch, self.input_generator.filenames[idx_start:idx_end])
            if batch_num == num_batches:
               break
