#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec  7 13:12:30 2017

@author: ksxu
"""

import numpy as np
import matplotlib.pyplot as plt
from conne_matrix import Connectivity
import ExcInhi_net as EXIN


#Example 1
seed =1
rng = np.random.RandomState(seed)

## Summary
#W    = self.Wrec
#Wexc =  W[np.where(W > 0)]
#Winh = -W[np.where(W < 0)]

N = 100
fig=plt.figure(1,figsize=(10,5))
plt.clf()
cmap=plt.get_cmap('jet')

Cfixed = np.zeros((N,N))
Cfixed[80:95,80:95] = 0.8
#E/I signature.
ei,_,_=EXIN.generate_ei(N)

C_or_N = EXIN.generate_Crec(ei)

ax1 = plt.subplot(231)
plt.imshow(C_or_N*ei)
plt.colorbar()
plt.title('Connection matrices',fontsize='small')


ax2 = plt.subplot(232)
plt.imshow(Cfixed,vmin=0, vmax=1,cmap=cmap)
plt.colorbar()
plt.title('Cfixed')




C_or_N[80:95,80:95]=0

C  = Connectivity(C_or_N, Cfixed=Cfixed )


Wrec=EXIN.init_weights(rng,C,m=N,n=N,distribution = 'normal')

WrecFULL =C.mask_plastic*Wrec+ C.mask_fixed
#

ax3 = plt.subplot(233)
plt.imshow(C.mask_plastic,cmap=cmap)
plt.colorbar()
plt.title('Plastic Mask')


ax4 = plt.subplot(234)
plt.imshow(C.mask_fixed,cmap=cmap)
plt.colorbar()
plt.title('Fixed Mask')

ax5 = plt.subplot(235)
plt.imshow(np.abs(WrecFULL),cmap=cmap)
plt.colorbar()
plt.title('$W^{rec,+}$')

ax6 = plt.subplot(236)
plt.imshow(np.abs(WrecFULL)*EXIN.generate_ei(N)[0],cmap=cmap)
plt.colorbar()
plt.title('$W^{rec}$')


plt.tight_layout()

plt.savefig('net-con.png',dpi= 300)


# example2
## Summary
#W    = self.Wrec
#Wexc =  W[np.where(W > 0)]
#Winh = -W[np.where(W < 0)]

N = 100

# the connection matrix
C_or_N =EXIN.generate_Crec(ei)
C_or_N[80:95,80:95]=0
#the weight will be nor training
Cfixed = np.zeros((N,N))
Cfixed[80:95,80:95] = 0.8
#Mask for plastic and fixed weights
C  = Connectivity(C_or_N, Cfixed=Cfixed )
Wrec0=EXIN.init_weights(rng,C,m=N,n=N,distribution = 'normal')
WrecPlus =np.abs(C.mask_plastic*Wrec0+ C.mask_fixed)
Wrec = WrecPlus * ei


fig=plt.figure(2,figsize=(7,5))
plt.clf()
cmap=plt.get_cmap('jet')

ax1=plt.subplot(221)
plt.imshow(C.mask_plastic,cmap=cmap)
plt.colorbar()
plt.title('$M^{rec}$')

ax2=plt.subplot(222)
plt.imshow(Wrec0,cmap=cmap)
plt.colorbar()
plt.title('$W^{rec,plastic}$')

ax3=plt.subplot(223)
plt.imshow(WrecPlus,cmap=cmap)
plt.colorbar()
plt.title('$W^{rec,+}$',fontsize='medium')

ax4=plt.subplot(224)
plt.imshow( Wrec,cmap=cmap)
plt.colorbar()
plt.title('$W^{rec}$')


plt.tight_layout()

plt.savefig('Wrec.png',dpi= 300)
#
