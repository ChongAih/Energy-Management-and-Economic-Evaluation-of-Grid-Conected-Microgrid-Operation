# -*- coding: utf-8 -*-
"""seq2seq.ipynb

Automatically generated by Colaboratory.

Original file is located at
    https://colab.research.google.com/drive/1mx-AXlnF8CHAk961BgAtTmOlBZNc0pjh
"""

import h5py
import tensorflow as tf
tf.keras.backend.set_floatx('float32')

class seq2seq_model():
  def __init__(self, encoder_units, decoder_units, timesteps_in, timesteps_out, num_features, directory):
    self.encoder_units = encoder_units
    self.decoder_units = decoder_units
    self.dropout_rate = 0.2
    self.timesteps_in = timesteps_in
    self.timesteps_out = timesteps_out
    self.num_features = num_features
    self.weights_dir = directory + "seq2seq.h5py"
    self.epochs = 20
    self.batch_size = 64
    self.create_model()

  def create_model(self):
    input_ = tf.keras.Input(shape=(self.timesteps_in, self.num_features))
    # encoder
    # x, state_h, state_c - (batch_size * encoder_units)
    x, state_h, state_c = tf.keras.layers.LSTM(self.encoder_units, 
                                               return_sequences=False, 
                                               return_state=True)(input_)
    # decoder
    # feed the output from last step of encoder to every unrolled cell of lstm
    # x - (batch_size, timestep_out, encoder_units)
    x = tf.keras.layers.RepeatVector(self.timesteps_out)(x)
    # feed the state information from last step of encoder to the decoder
    x = tf.keras.layers.LSTM(self.decoder_units, 
                             return_sequences=True, 
                             return_state=False)(x, initial_state=[state_h, state_c])
    output = tf.keras.layers.Conv1D(1, kernel_size=1, strides=1, 
                                    padding="same", activation="linear")(x)                  
    self.model = tf.keras.Model(inputs=input_, outputs=output)
    self.model.compile(loss="mae", optimizer="adam")
    #self.model.summary()
  
  def fit_model(self, x_train, y_train, x_valid, y_valid):
    modelcheckpoint = tf.keras.callbacks.ModelCheckpoint(self.weights_dir, monitor='val_loss', 
                                                         verbose=0, save_best_only=True,
                                                         save_weights_only=True)
    earlystopping = tf.keras.callbacks.EarlyStopping(monitor='val_loss', patience=10)
    callbacks = [modelcheckpoint, earlystopping]
    self.history = self.model.fit(x_train, y_train, batch_size=self.batch_size, 
                                  epochs=self.epochs, verbose=1, callbacks=callbacks, 
                                  validation_data=(x_valid, y_valid))

  def predict(self, x_test):
    self.model.load_weights(self.weights_dir)
    y_pred = self.model.predict(x_test, batch_size=self.batch_size)
    return y_pred