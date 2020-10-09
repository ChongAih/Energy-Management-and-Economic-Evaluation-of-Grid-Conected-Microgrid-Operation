#!/usr/bin/env python
# coding: utf-8

# In[1]:


import tensorflow as tf


# In[2]:


def retrieve_model():
    model_averse = tf.keras.models.load_model("trained model/RL_averse.h5")
    model_seeking = tf.keras.models.load_model("trained model/RL_seeking.h5")
    return model_averse, model_seeking


# In[3]:


if __name__ == "__main__":
    
    import ess # self-built module for battery related function
    import numpy as np
    
    battery_rl_averse = ess.battery_discrete()
    battery_rl_seeking = ess.battery_discrete()
    
    model_averse, model_seeking = retrieve_model()
    x = np.array([[1, 1, 1, 1, 1]]) # per unit pv, load, price, SOC, average price of past 24 hours
    
    # Discrete solution ranging from [-1, 1] with interval of 0.2
    action_averse = np.argmax(model_averse.predict(x)) 
    action_seeking = np.argmax(model_seeking.predict(x))
    
    # Based on the declared action set, return the desired action and power output = action * battery.P_rated
    print ("Risk averse action: {}".format(battery_rl_averse.action_set[action_averse]))
    print ("Risk seeking action: {}".format(battery_rl_seeking.action_set[action_seeking]))

