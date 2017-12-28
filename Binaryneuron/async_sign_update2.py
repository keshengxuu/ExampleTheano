#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Thu Dec 28 16:42:59 2017

@author: ksxu
"""

#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Sat Dec 23 17:36:56 2017
keshengxuu@gmail.com
@author: ksxuu
"""
import numpy as np

Ne1 = 8000 ;
Ni1 = 2000 ;
Ne2 = 8000 ;
Ni2 = 2000 ;
tauE =10;
tauI = 8;
Ntime = 10;
K = 1000;
E0 = 0.3;
J_EE = 1
J_IE = 1
JE = 4
JI = 2.5
J_EI = -JE
J_II = -JI
Jtilde = 1.5
TK = 4
pc = K/(Ne1+Ni1)
Nevent = 1000


#aa= []
# start for the simulation 
Ini_TimeEve = np.concatenate(( np.random.poisson(tauE, size=(Ne1,1)),
                        np.random.poisson(tauE, size=(Ne2,1)),
                        np.random.poisson(tauI, size=(Ni1,1)),
                        np.random.poisson(tauI, size=(Ni2,1))),axis=0)
for i in range(Nevent): # looping for the simulation time
    # updating the neurons using poisson statistics
    indice = np.where(Ini_TimeEve[:,0]==i)[0]
    # time events of updating  excitatory neurons.
    Ini_TimeEve[indice[indice<(Ne1+Ne2)]] += np.random.poisson(tauE, size=(len(indice[indice<(Ne1+Ne2)]),1)) 
    # time events of updating  inhibitory neurons.
    Ini_TimeEve[indice[indice>=(Ne1+Ne2)]] += np.random.poisson(tauI, size=(len(indice[indice>=(Ne1+Ne2)]),1))
    
    print  indice
