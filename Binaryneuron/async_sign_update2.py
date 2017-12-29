#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:42:59 2017
keshengxuu@gmail.com
@author: ksxu
"""
from __future__ import division 
import numpy as np
import time

# The parameters for building the balanced networks
# Ne1,Ne2,Ni1,Ni2 are the total number od neurons in the excitatory and inhibitory populations
Ne1 = 8000; 
Ni1 = 2000; 
Ne2 = 8000; 
Ni2 = 2000;
# tauE(tauI) is the mean time interval between updates for neurons in the excitatory(inhibitory) populations
tauE =10;
tauI = 8;
Nevent = 10; # the length of simulation time
K = 1000;# K is the average number of inputs a neuron from each population
E0 = 0.3; # E0 is of the order unity for the external neurons
m0 = 1.0 ;  # 0<m0<1 represents the mean acticity of the external neurons
# connection strengths as following
J_EE = 1
J_IE = 1
JE = 4
JI = 2.5
J_EI = -JE
J_II = -JI
Jtilde = 1.5
TK = 4
TE = 1
TI = 0.7
# cp is the connection probability to bulid the netowrks with random,sparse architcture.
cp = K/(Ne1+Ni1)
start_time = time.time()

#conection strengths
jee = J_EE/np.sqrt(K) 
jei = J_EI/np.sqrt(K) 
jie = J_IE/np.sqrt(K)
jtilde = -Jtilde*np.sqrt(K)/(Ne1+Ni1)
jii = J_II/np.sqrt(K)



# intinal states for excitatory and inbibitory population neurons  of 1th balcanced networks
sigmaE1 = np.random.randint(2,size = Ne1)
sigmaI1 = np.random.randint(2,size = Ni1)
# excitatory and inbibitory neurons for  population of 2th balcanced networks
sigmaE2 = np.random.randint(2,size = Ne2)
sigmaI2 = np.random.randint(2,size = Ni2)


# excitatory population of the first banlaced networks
# the connection patterns for E-E  neurons
Cee1 = np.random.binomial(1,cp,(Ne1,Ne1))
Cee2 = np.random.binomial(1,cp,(Ne2,Ne2))
Cee1[np.diag_indices(Ne1)] = 0
Cee2[np.diag_indices(Ne2)] = 0
# the connection patterns for I-E neurons
Cei1 = np.random.binomial(1,cp,(Ne1,Ni1))
Cei2 = np.random.binomial(1,cp,(Ne2,Ni2))
Cie1 = np.random.binomial(1,cp,(Ni1,Ne1))
Cie2 = np.random.binomial(1,cp,(Ni2,Ne2))
Cii1 = np.random.binomial(1,cp,(Ni1,Ni1))
Cii2 = np.random.binomial(1,cp,(Ni2,Ni2))
Cii1[np.diag_indices(Ni1)] = 0
Cii2[np.diag_indices(Ni2)] = 0
# the mutual inhibtion connection from the inhibitory neurons of 2th balanced netwotks
Cmut_inh1= np.ones((Ne1,Ni2))
Cmut_inh2= np.ones((Ne2,Ni1))



# start for the simulation 
Ini_TimeEve = np.concatenate(( np.random.poisson(tauE, size=(Ne1,1)),
                        np.random.poisson(tauE, size=(Ne2,1)),
                        np.random.poisson(tauI, size=(Ni1,1)),
                        np.random.poisson(tauI, size=(Ni2,1))),axis=0)
for i in range(Nevent): # looping for the simulation time
    # uE1, uE2(uI1,uI2)  are the total synaptic input  of the excitatory population(inhibitory population)
    uE1 = jee*np.dot(Cee1, sigmaE1) + np.sqrt(K)*E0 *m0+ jei*np.dot(Cei1, sigmaI1) + jtilde*np.dot(Cmut_inh1, sigmaI2)-TK
    uI1 = jie*np.dot(Cie1, sigmaE1)  + jii*np.dot(Cii1, sigmaI1) -TK
    uE2 = jee*np.dot(Cee2, sigmaE2) + np.sqrt(K)*E0*m0 + jei*np.dot(Cei2, sigmaI2) + jtilde*np.dot(Cmut_inh2, sigmaI1) -TK
    uI2 = jie*np.dot(Cie2,sigmaE2) + jii*np.dot(Cii2, sigmaI2) -TK
    
    
    U1 = np.concatenate((uE1, uE2, uI1,  uI2), axis=0)
    sigma = np.concatenate((sigmaE1, sigmaE2, sigmaI1, sigmaI2), axis=0)

    # indice  are positions for the updating  neurons at time t.
    indice = np.where(Ini_TimeEve[:,0]==i)[0]
    #Heaviside function of updating rules
    sigma[indice[U1[indice]<=0]] = 0
    sigma[indice[U1[indice]>0]] = 1 
    
    
    # time events of updating  excitatory neurons.
    Ini_TimeEve[indice[indice<(Ne1+Ne2)]] += np.random.poisson(tauE, size=(len(indice[indice<(Ne1+Ne2)]),1)) 
    # time events of updating  inhibitory neurons.
    Ini_TimeEve[indice[indice>=(Ne1+Ne2)]] += np.random.poisson(tauI, size=(len(indice[indice>=(Ne1+Ne2)]),1))
    
    sigmaE1, sigmaE2, sigmaI1, sigmaI2 = np.split(sigma,[Ne1, (Ne1+Ne2), (Ne1+Ni1+Ne2)])
    
    #print  indice

print("Indice--- %s seconds ---" % (time.time() - start_time))
