import numpy as np
from tensorflow import keras
from keras.utils import to_categorical
import segmentation_models as sm
from sklearn.preprocessing import MinMaxScaler

scaler = MinMaxScaler()

def preprocess_img(img, backbone):
  # Flatten images along the channel dimension as MinMaxScaler expects
  # array dimension <= 2
  flattened_img = img.reshape(-1, img.shape[-1])
  # Scale images
  flattened_img_scaled = scaler.fit_transform(flattened_img)
  # Reshape scaled flattened image to original shape
  img_scaled = flattened_img_scaled.reshape(img.shape)
  # Preprocess input based on the pretrained backbone for transfer learning
  preprocess_input = sm.get_preprocessing(backbone)
  img_preprocessed = preprocess_input(img_scaled)
  # return the pre-processed image
  return img_preprocessed

def preprocess_mask(mask, num_classes):
  # Convert mask to one-hot
  mask_one_hot = to_categorical(mask, num_classes)
  # return the one-hot encoded mask
  return mask_one_hot

def postprocess_mask(pred_mask):
  # Convert the predicted one-hot encoded mask back to normal
  pred_mask_postprocessed = np.argmax(pred_mask, axis=2)
  # return the one-hot decoded mask
  return pred_mask_postprocessed