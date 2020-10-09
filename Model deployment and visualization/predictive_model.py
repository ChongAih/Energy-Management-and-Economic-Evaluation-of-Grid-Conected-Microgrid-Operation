#!/usr/bin/env python
# coding: utf-8

# In[1]:


from tx import tx_model
from seq2seq import seq2seq_model


# In[2]:


def retrieve_model():
    # global hyperparameters
    # 2 weeks of lagged observation to predict next 24 hours
    timesteps_in = 2 * 7 * 24
    timesteps_out = 24

    # Hyperparameter for load TX model
    num_layers = 1
    d_model = 64
    num_heads = 4
    dff = 64
    kernel_size = 5
    dropout_rate = 0.1
    item = 'Load'
    directory = "trained model/"
    model_load = tx_model(num_layers, d_model, num_heads, dff,
                          kernel_size, dropout_rate, timesteps_in, 
                          timesteps_out, item, directory)

    # Hyperparameter for price TX model
    num_layers = 1
    d_model = 64
    num_heads = 4
    dff = 64
    kernel_size = 5
    dropout_rate = 0.1
    item = 'Price'
    directory = "trained model/"
    model_price = tx_model(num_layers, d_model, num_heads, dff,
                           kernel_size, dropout_rate, timesteps_in, 
                           timesteps_out, item, directory)

    # Hyperparameter for PV seq2seq model
    encoder_units = 32
    decoder_units = 32
    num_features = 41
    item = 'PV'
    directory = "trained model/"
    model_pv = seq2seq_model(encoder_units, decoder_units, timesteps_in, timesteps_out, num_features, directory)
    
    return model_load, model_price, model_pv


# In[3]:


if __name__ == "__main__":
    model_load, model_price, model_pv = retrieve_model()

