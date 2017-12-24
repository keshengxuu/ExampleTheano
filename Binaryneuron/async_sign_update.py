#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:36:56 2017
keshengxuu@gmail.com
@author: ksxuu
"""

import numpy as np

state_s0 = np.ones(1000)
nr_neurons = 4
s = np.random.poisson(10, size=(nr_neurons,30))
Tetime =np.zeros(s.shape)
Tetime = np.cumsum(s,axis = 1)

# nevent is the flag of  updating binary neuron state 
nevent = np.zeros(s.shape[0],dtype=int)
for i in range(np.amax(Tetime)): # looping for the simulation time
    for j in range(s.shape[0]):   # looping for the numbe of the neurons
        if nevent[j] <s.shape[1] and  i == Tetime[j,nevent[j]]:
            nevent[j] = nevent[j] + 1
            print i,j
