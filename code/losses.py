import numpy as np
from keras import backend as K


# Custom dice coefficient
def dice_coef(y_true, y_pred):
  mask = K.cast(K.tf.count_nonzero(y_true, [0,1,2]), K.floatx()) # -> of size len(z-axis), having 0 if a slice has only 0s
  mask_binary = K.cast(K.not_equal(mask, 0.0), K.floatx()) # boolean tensor, having 0 if a slice has only 0s

  # the usual dice loss
  y_true_f = K.flatten(y_true)
  y_pred_f = K.flatten(y_pred * mask_binary)
  intersection = K.sum(y_true_f * y_pred_f)
  return (2. * intersection + 1) / (K.sum(y_true_f) + K.sum(y_pred_f) + 1)

def dice_coef_np(y_true, y_pred):
  y_true = np.asarray(y_true)
  y_pred = np.asarray(y_pred)
  for i in range(y_true.shape[3]):
    if np.count_nonzero(y_true[:,:,:,i,:]) == 0:
      y_pred[:,:,:,i,:] = 0.
  y_true_f = y_true.flatten()
  y_pred_f = y_pred.flatten()
  intersection = np.sum(y_true_f * y_pred_f)
  return (2. * intersection + 1) / (np.sum(y_true_f) + np.sum(y_pred_f) + 1)

# Losses
def dice_coef_loss(y_true, y_pred):
  return 1.-dice_coef(y_true, y_pred)

def masked_mse(y_tup, y_pred):
  y_true = y_tup[0]
  y_inp = y_tup[1]
  mask_true = K.cast(K.equal(y_inp, 0.0), K.floatx())
  
  masked_squared_error = K.square(mask_true * (y_true - y_pred))
  masked_mse = K.sum(masked_squared_error, axis=-1) / K.maximum(K.sum(mask_true, axis=-1), 1)
  return masked_mse