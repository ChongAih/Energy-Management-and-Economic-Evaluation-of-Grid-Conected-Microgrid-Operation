# -*- coding: utf-8 -*-
"""tx.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1A0XzDsfO7bmzYmSdVMTxXzLZq9D67tvA
"""

import time
import numpy as np
import tensorflow as tf
import transformer # import library rectified using reference - https://www.tensorflow.org/tutorials/text/transformer
from sklearn.metrics import mean_absolute_error as mae

# Transformer - class object
# Transformer(num_layers, d_model, num_heads, dff, input_vocab_size, target_vocab_size, pe_input, pe_target, kernel_size, rate=0.1)
  # num_layers = number of encoder & decoder layers
  # d_model = number of features to be created from input for multihead-attention
  # num_heads = number of heads (depth = d_model / num_heads)
  # dff = hidden neuron number of feed forward network
  # input_vocab_size = 1 (there is only 1 feature)
  # target_vocab_size = 1 (there is only 1 feature)
  # pe_input = maximum position encoding length (not affect result as long as it is longer than the sequence)
  # pe_target = maximum position encoding length (not affect result as long as it is longer than the sequence)
  # kernel_size = convolution kernel size (NLP transformer embedding layer is replaced with convolution layer)
  # rate = dropout rate

# create_padding_mask - function (no needed in this work as there is no padding - all input are of the same length)
# create_look_ahead_mask - function (needed for decoder to mask the future target value to prevent information leak)

class tx_model():
  def __init__(self, num_layers, d_model, num_heads, dff,
               kernel_size, dropout_rate, timesteps_in, 
               timesteps_out, item, directory):

    self.timesteps_in = timesteps_in
    self.timesteps_out = timesteps_out

    # fixed transformer parameters
    input_vocab_size = 1 # there is only 1 feature
    target_vocab_size = 1 # there is only 1 feature
    pe_input = 10 * timesteps_in
    pe_target = 10 * timesteps_out

    # create transformer model
    self.model = transformer.Transformer(num_layers, d_model, num_heads, dff,
                                         input_vocab_size, target_vocab_size, 
                                         pe_input, pe_target, kernel_size,
                                         dropout_rate)

    # training parameters
    self.weights_dir = directory + "TX_" + str(item) + "_k" + str(kernel_size) + \
                       "_dm" + str(d_model) + "_df" + str(dff) + "_l" + str(num_layers) + \
                       "_h" + str(num_heads) + "_weights.h5py"
    self.epochs = 30
    self.threshold = 0.005
    self.batch_size = 64
    self.optimizer = tf.keras.optimizers.Adam(0.0001)
    self.train_loss = tf.keras.metrics.Mean(name='train_loss')

  # loss function for training
  def loss_function(self, real, pred):
    mask = tf.math.logical_not(tf.math.equal(real, 0))
    loss_ = tf.expand_dims(tf.keras.losses.MAE(real, pred), axis=-1)
    mask = tf.cast(mask, dtype=loss_.dtype)
    loss_ *= mask
    return tf.reduce_sum(loss_)/tf.reduce_sum(mask)

  # create padding msks for encoder and decoder & look ahead mask for decoder
  def create_masks(self, inp, tar):

    # Create look ahead mask - location of future token will be 1, 
    # Since there is no need for padding mask, the output will all be zero
    # Combine the look ahead and padding mask
    # For NLP transformer, the input is (x,y)
    # For modified time series transformer, the input is (x,y,1)
    # Thus the dimension is different when creating padding and modification is required

    # Encoder padding mask
    enc_padding_mask = transformer.create_padding_mask(inp)
    enc_padding_mask = enc_padding_mask[:,:,:,:,0] # ensure consistent dimension
    
    # Used in the 2nd attention block in the decoder.
    # This padding mask is used to mask the encoder outputs.
    dec_padding_mask = transformer.create_padding_mask(inp)
    dec_padding_mask = dec_padding_mask[:,:,:,:,0] # ensure consistent dimension
    
    # Used in the 1st attention block in the decoder.
    # It is used to pad and mask future tokens in the input received by the decoder.
    look_ahead_mask = transformer.create_look_ahead_mask(tf.shape(tar)[1])
    dec_target_padding_mask = transformer.create_padding_mask(tar)
    dec_target_padding_mask = dec_target_padding_mask[:,:,:,:,0] # ensure consistent dimension

    combined_mask = tf.maximum(dec_target_padding_mask, look_ahead_mask)

    return enc_padding_mask, combined_mask, dec_padding_mask

  # fit the transformer model
  def fit_model(self, x_train, y_train, x_valid, y_valid):

    def get_batch(inp, tar, batch_size=1):
        l = len(inp)
        for i in range(0, l, batch_size):
            yield i / batch_size, inp[i:min(i + batch_size, l)], tar[i:min(i + batch_size, l)]

    # change to the default dtype of tensorflow layer
    x_train = tf.cast(x_train, dtype=tf.float32)
    y_train = tf.cast(y_train, dtype=tf.float32)

    val_loss_bef = np.inf
    for epoch in range(self.epochs):
      start = time.time()
      train_loss = 0

      # batch training
      for batch, inp, tar in get_batch(x_train, y_train, batch_size=self.batch_size):
          inp = inp[:, :, :]
          tar_inp = inp[:, -self.timesteps_out:, :]
          tar_real = tar[:, :, 0:1]  
          
          # enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(inp, tar_inp)
          enc_padding_mask, combined_mask, dec_padding_mask = None, None, None
          
          with tf.GradientTape() as tape:
            predictions, _ = self.model(inp, tar_inp, 
                                        True,
                                        enc_padding_mask, 
                                        combined_mask, 
                                        dec_padding_mask)
            loss = self.loss_function(tar_real, predictions) # Find the loss between target and prediction

          gradients = tape.gradient(loss, self.model.trainable_variables) # Compute the first derivative of the weights
          self.optimizer.apply_gradients(zip(gradients, self.model.trainable_variables)) # Backpropagation 
          train_loss += loss

      # Save the weights of best model
      y_valid_pred, _, _ = self.predict(x_valid, True)
      val_loss = mae(y_valid[:, :, 0], y_valid_pred[:, :, 0])
      
      if val_loss < val_loss_bef:
        self.model.save_weights(self.weights_dir)
        val_loss_bef = val_loss

      print('Epoch {} training loss - {:.4f}, validation loss - {:.4f}'.format(epoch + 1, train_loss / (batch + 1), val_loss))
      print('Best validation loss - {:.4f}, time taken for 1 epoch {:.4f} sec\n'.format(val_loss_bef, time.time() - start))

  # make prediction
  def predict(self, x_test, training):

    def get_batch(inp, batch_size=1):
        l = len(inp)
        for i in range(0, l, batch_size):
            yield i / batch_size, inp[i:min(i + batch_size, l)]
    
    # only load the save weights only if the model is being trained
    if not training:
      self.model.load_weights(self.weights_dir)

    # change to the default dtype of tensorflow layer
    x_test = tf.cast(x_test, dtype=tf.float32)

    for batch, inp in get_batch(x_test, batch_size=self.batch_size):
      # set the last t step of input as the decoder input to make prediction
      encoder_input = inp[:, :, :]
      decoder_input = inp[:, -self.timesteps_out:, :]

      # enc_padding_mask, combined_mask, dec_padding_mask = self.create_masks(encoder_input, decoder_input)
      enc_padding_mask, combined_mask, dec_padding_mask = None, None, None

      # predictions.shape == (batch_size, seq_len, vocab_size)
      # set dropout to be false
      predictions, attention_weights = self.model(encoder_input, 
                                                  decoder_input,
                                                  False, 
                                                  enc_padding_mask,
                                                  combined_mask,
                                                  dec_padding_mask)

      if batch == 0:
        y_test_pred = predictions
        #attentions = attention_weights
      else:
        y_test_pred = tf.concat([y_test_pred, predictions], axis=0)
        #attentions = tf.concat([attentions, attention_weights], axis=0)

    # attention_weights are saved for the last batch only
    # attention_weights contain the attention of decoder input to encoder output & decoder input self attention
    return y_test_pred, attention_weights, decoder_input