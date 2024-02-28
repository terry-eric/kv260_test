# Copyright 2019 Xilinx Inc.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
# ==============================================================================
import tempfile
import os

import tensorflow as tf
import numpy as np
import datetime

from tensorflow import keras

# Load MNIST dataset
mnist = keras.datasets.mnist
(train_images, train_labels), (test_images, test_labels) = mnist.load_data()

# Normalize the input image so that each pixel value is between 0 to 1.
train_images = train_images / 255.0
test_images = test_images / 255.0

# Expand the dimensions of the training and testing images to include the channel dimension
train_images = np.repeat(train_images[..., np.newaxis], 3, axis=-1)  # 從 (60000, 28, 28, 1) 變為 (60000, 28, 28, 3)
test_images = np.repeat(test_images[..., np.newaxis], 3, axis=-1)  # 從 (10000, 28, 28, 1) 變為 (10000, 28, 28, 3)

# train_images = np.expand_dims(train_images, axis=-1)  # From (60000, 28, 28) to (60000, 28, 28, 1)
# test_images = np.expand_dims(test_images, axis=-1)  # From (10000, 28, 28) to (10000, 28, 28, 1)


from tensorflow_model_optimization.quantization.keras.vitis.layers import vitis_activation

model_name = "yolov3"
model = tf.keras.models.load_model(f'../{model_name}.h5')
model.summary()

# Post-Training Quantize
from tensorflow_model_optimization.quantization.keras import vitis_quantize

quantizer = vitis_quantize.VitisQuantizer(model)
# Quantized
quantized_model = quantizer.quantize_model(
    calib_dataset=train_images[0:10],
    include_cle=True,
    cle_steps=10,
    include_fast_ft=True)

quantized_model.save(f'{model_name}-quantized.h5')


# # Load Quantized Model
# quantized_model = keras.models.load_model('yolo-quantized.h5')

# # Evaluate Quantized Model
# quantized_model.compile(
#     loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=False),
#     metrics=['sparse_categorical_accuracy'])
# quantized_model.evaluate(test_images, test_labels, batch_size=6)

# # Dump Quantized Model
# quantizer.dump_model(
#     quantized_model, dataset=train_images[0:1], dump_float=True)
