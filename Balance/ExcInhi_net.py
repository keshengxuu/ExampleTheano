#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Mon Dec  4 17:57:21 2017

@author: ksxu
"""
import numpy as np
from conne_matrix import Connectivity

#-----------------------------------------------------------------------------------------
# Define E/I populations
#-----------------------------------------------------------------------------------------
def generate_ei(N, pE=0.8):
    """
    E/I signature.

    Parameters
    ----------

    N : int
        Number of recurrent units.

    pE : float, optional
         Fraction of units that are excitatory. Default is the usual value for cortex.

    """
    assert 0 <= pE <= 1

    Nexc = int(pE*N)
    Ninh = N - Nexc

    idx = range(N)
    EXC = idx[:Nexc]
    INH = idx[Nexc:]

    ei       = np.ones(N, dtype=int)
    ei[INH] *= -1
    print EXC

    return ei, EXC, INH

#-----------------------------------------------------------------------------------------
# Functions for generating connection matrices
#-----------------------------------------------------------------------------------------

def generate_Crec(ei, p_exc=0.2, p_inh=0.3, rng=None, seed=1, allow_self=False):
    if rng is None:
        rng = np.random.RandomState(seed)

    N    = len(ei)
    exc, = np.where(ei > 0)
    inh, = np.where(ei < 0)

    C = np.zeros((N, N))
    for i in exc:
        C[i,exc] = 1*(rng.uniform(size=len(exc)) < p_exc)
        if not allow_self:
            C[i,i] = 0
        C[i,inh]  = 1*(rng.uniform(size=len(inh)) < p_inh)
        C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])
    for i in inh:
        C[i,exc] = 1*(rng.uniform(size=len(exc)) < p_exc)
        C[i,inh] = 1*(rng.uniform(size=len(inh)) < p_inh)
        if not allow_self:
            C[i,i] = 0
        C[i,inh] *= np.sum(C[i,exc])/np.sum(C[i,inh])
    C /= np.linalg.norm(C, axis=1)[:,np.newaxis]

    return C

        

##/////////////////////////////////////////////////////////////////////////////////////
#
def init_weights(rng, C, m, n, distribution):
    """
    Initialize weights from a distribution.

    Parameters
    ----------

    rng : numpy.random.RandomState
          Random number generator.

    C : Connectivity
        Specify which weights are plastic and nonzero.

    m, n : int
           Number of rows and columns, respectively.

    distribution : str
                   Name of the distribution.

    """
    # Account for plastic and fixed weights.
    if C is not None:
        mask = C.plastic
        size = C.nplastic
    else:
        mask = 1
        size = m*n

    # Distributions
    if distribution == 'uniform':
        w = 0.1*rng.uniform(-mask, mask, size=size)
    elif distribution == 'normal':
        w = rng.normal(np.zeros(size), mask, size=size)
    elif distribution == 'gamma':
        k     = 2
        theta = 0.1*mask/k
        w     = rng.gamma(k, theta, size=size)
    elif distribution == 'lognormal':
        mean  = 0.5*mask
        var   = 0.1
        mu    = np.log(mean/np.sqrt(1 + var/mean**2))
        sigma = np.sqrt(np.log(1 + var/mean**2))
        w     = rng.lognormal(mu, sigma, size=size)

    if C is not None:
        W = np.zeros(m*n)
        W[C.idx_plastic] = w
    else:
        W = w

    return W.reshape((m, n))




if __name__=="__main__":
    import matplotlib.pyplot as plt
    ##    
    seed =1
    rng = np.random.RandomState(seed)
    
    ## Summary
    #W    = self.Wrec
    #Wexc =  W[np.where(W > 0)]
    #Winh = -W[np.where(W < 0)]
    
    N = 100
    #E/I signature.
    ei,_,_=generate_ei(N)
    # the connection matrix
    C_or_N = generate_Crec(ei)
    C_or_N[80:95,80:95]=0
    #the weight will be not training
    Cfixed = np.zeros((N,N))
    Cfixed[80:95,80:95] = 0.8
    #Mask for plastic and fixed weights
    connectivity_mask  = Connectivity(C_or_N, Cfixed=Cfixed )
    # Store the pre-synaptic neurons to each plastic neuron
    W_plastic = [list(np.nonzero(connectivity_mask.mask_plastic[i, :])[0]) for i in range(N)]
    # Wrec_plastic is the weight after trainning
    Wrec_plastic = init_weights(rng,connectivity_mask,m=N,n=N,distribution = 'normal')
    # Wrec_fixed is a matrix of fixed weights
    Wrec_fixed = connectivity_mask.mask_fixed
    
    Wrec_plus = np.abs(connectivity_mask.mask_plastic*Wrec_plastic+ Wrec_fixed)
    #Weights between the recurrent units
    Wrec = Wrec_plus * ei
    
    
    fig=plt.figure(1,figsize=(7,5))
    plt.clf()
    cmap=plt.get_cmap('jet')
    plt.subplot(221)
    plt.imshow( connectivity_mask.mask_plastic,cmap=cmap)
    plt.colorbar()
    plt.title('plastic mask')
    plt.subplot(222)
    plt.imshow( Wrec_plastic,cmap=cmap)
    plt.colorbar()
    plt.title('Wrec plastic')
    plt.subplot(223)
    plt.imshow( Wrec_plus,cmap=cmap)
    plt.colorbar()
    plt.title('$W^{rec,+}$')
    plt.subplot(224)
    plt.imshow( Wrec,cmap=cmap)
    plt.colorbar()
    plt.title('$W^{rec}$')
    
    plt.tight_layout()
    
    plt.savefig('net-con.png',dpi= 300)
#


