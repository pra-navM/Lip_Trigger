import tensorflow as tf
import os
import numpy as np
from PIL import Image

representative_data_dir = 'repdataset'

#preprocess
def preprocess_image(image_path):
    image = Image.open(image_path).resize((224, 224))  # Resize image to 224x224
    image = np.array(image).astype(np.float32) / 255.0  # Normalize the image
    image = np.expand_dims(image, axis=0)  # Add batch dimension
    return image

#generate rep data
def representative_dataset_gen():
    image_paths = [os.path.join(representative_data_dir, f) for f in os.listdir(representative_data_dir) if f.endswith(('.jpg', '.jpeg', '.png'))]
    
    for image_path in image_paths:
        image = preprocess_image(image_path)
        yield [image]

#load model
model = tf.keras.models.load_model('new_new_mouth_state_model.h5')


converter = tf.lite.TFLiteConverter.from_keras_model(model)
converter.optimizations = [tf.lite.Optimize.DEFAULT]

#supported operations
converter.target_spec.supported_ops = [tf.lite.OpsSet.TFLITE_BUILTINS_INT8]
converter.inference_input_type = tf.uint8  # or tf.int8
converter.inference_output_type = tf.uint8  # or tf.int8

#rep dataset
converter.representative_dataset = representative_dataset_gen
tflite_quant_model = converter.convert()

#save
with open('mouth_state_model_quant.tflite', 'wb') as f:
    f.write(tflite_quant_model)

print("Quantized model saved as 'mouth_state_model_quant.tflite'")

