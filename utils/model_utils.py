import torch
import tensorflow as tf

def load_torch_model(model_path):
    # Load a pre-trained PyTorch model
    model = torch.load(model_path)
    model.eval()
    return model

def load_tf_model(model_path):
    # Load a pre-trained TensorFlow model
    model = tf.keras.models.load_model(model_path)
    return model

def save_model(model, model_path):
    # Save the model to the specified path
    model.save(model_path)
