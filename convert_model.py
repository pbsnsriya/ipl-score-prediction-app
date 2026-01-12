import tensorflow as tf
import os

model_path = 'model.h5'
tflite_path = 'model.tflite'

try:
    # Load the Keras model
    model = tf.keras.models.load_model(model_path)
    
    # Convert the model
    converter = tf.lite.TFLiteConverter.from_keras_model(model)
    tflite_model = converter.convert()

    # Save the model
    with open(tflite_path, 'wb') as f:
        f.write(tflite_model)
    
    print(f"Successfully converted {model_path} to {tflite_path}")

except Exception as e:
    print(f"Error converting model: {e}")
