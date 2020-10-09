#!/usr/bin/env python
# coding: utf-8

# In[1]:


import pickle
import numpy as np


# In[2]:


class battery():
    def __init__(self):
        """
        P_rated - charge/ discharge rate (kW)
        E_rated - rated capacity (kWh)
        C_E - energy capital cost ($/kWh)
        LC - life cycle
        eta - efficiency
        DOD - depth of discharge
        wear_cost - wear & operation cost ($/kWh/operation)
        wear_cost = (C_E * E_rated) / (eta * E_rated * LC * DOD)

        As rated power smaller than rated energy, initial SOC is set to be maximumly 
        rated power away from the rated energy, if not LP does not work as constraint 
        violated --> rated_power / rated_energy + initial SOC < target SOC
        """ 
        with open ("trained model/sc_energy.pkl", "rb") as file:
            self.sc_energy = pickle.load(file)
        with open ("trained model/sc_price.pkl", "rb") as file:
            self.sc_price = pickle.load(file)
        
        self.P_rated = self.sc_energy.transform(np.array([[1000]]))[0] # pu
        self.E_rated = self.sc_energy.transform(np.array([[5000]]))[0] # pu 
        self.C_E = self.sc_price.transform(np.array([[171]]))[0] # pu
        self.LC = 4996
        self.eta = 1.
        self.DOD = 1.
        self.wear_cost = self.C_E / self.eta / self.DOD / self.LC
        self.target_SOC = 0.5 # Decide the backup energy required
        self.initial_SOC = self.target_SOC 
        self.current_SOC = self.initial_SOC
        
    def update_SOC(self, action):
        self.current_SOC = self.current_SOC + action * self.P_rated[0] / self.E_rated[0]
        return self.current_SOC


# In[3]:


class battery_continuous(battery):
    def __init__(self):
        super(battery_continuous, self).__init__()


# In[4]:


class battery_discrete(battery):
    def __init__(self):
        super(battery_discrete, self).__init__()
        action_size = 11
        self.action_set = np.linspace(-1, 1, num = action_size, endpoint = True)


# In[5]:


if __name__ == "__main__":
    battery_lp = battery_continuous()
    battery_rl = battery_discrete()

