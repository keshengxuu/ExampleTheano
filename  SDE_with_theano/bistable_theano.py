#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:18:26 2017
This code is from:
http://www.nehalemlabs.net/prototype/blog/2013/10/17/solving-stochastic-differential-equations-with-theano/
@author: ksxu
"""

import theano
import theano.tensor as T
from theano.tensor.shared_randomstreams import RandomStreams
import numpy as np
import matplotlib.pyplot as plt
import time
 
#define the ode function
#dc/dt  = f(c, lambda)
#c is a vector with n components
def evolve(c, n, k, l):
	return T.pow(c, n)/(T.pow(c, n)+T.pow(k,n)) - l*c
 
def euler(c, n, k, l, dt):
	return T.cast(c + dt*evolve(c, n, k, l) + T.sqrt(dt)*c*rv_n, 'float32')
 
def rk4(c, n, k, l, dt):
	'''
	Adapted from
	http://people.sc.fsu.edu/~jburkardt/c_src/stochastic_rk/stochastic_rk.html
	'''
	a21 =   2.71644396264860
	a31 = - 6.95653259006152
	a32 =   0.78313689457981
	a41 =   0.0
	a42 =   0.48257353309214
	a43 =   0.26171080165848
	a51 =   0.47012396888046
	a52 =   0.36597075368373
	a53 =   0.08906615686702
	a54 =   0.07483912056879
 
	q1 =   2.12709852335625
	q2 =   2.73245878238737
	q3 =  11.22760917474960
	q4 =  13.36199560336697
 
	x1 = c
	k1 = dt * evolve(x1, n, k, l) + T.sqrt(dt) * c * rv_n
 
	x2 = x1 + a21 * k1
	k2 = dt * evolve(x2, n, k, l) + T.sqrt(dt) * c * rv_n
 
	x3 = x1 + a31 * k1 + a32 * k2
	k3 = dt * evolve(x3, n, k, l) + T.sqrt(dt) * c * rv_n
 
	x4 = x1 + a41 * k1 + a42 * k2
	k4 = dt * evolve(x4, n, k, l) + T.sqrt(dt) * c * rv_n
 
	return T.cast(x1 + a51 * k1 + a52 * k2 + a53 * k3 + a54 * k4, 'float32')
 
if __name__ == '__main__':
	#random
	srng = RandomStreams(seed=31415)
 
	#define symbolic variables
	dt = T.fscalar("dt")
	k = T.fscalar("k")
	l = T.fscalar("l")
	n = T.fscalar("n")
	c = T.fvector("c")
 
	#define numeric variables
	num_samples = 50000
	c0 = theano.shared(0.5*np.ones(num_samples, dtype='float32'))
	n0 = 6
	k0 = 0.5
	l0 = 1/(1+np.power(k0, n0))
	dt0 = 0.1
	total_time = 8
	total_steps = int(total_time/dt0)
	rv_n = srng.normal(c.shape, std=0.05) #is a shared variable
 
	#create loop
	#first symbolic loop with everything
	(cout, updates) = theano.scan(fn=rk4,
									outputs_info=[c], #output shape
									non_sequences=[n, k, l, dt], #fixed parameters
									n_steps=total_steps)
	#compile it
	sim = theano.function(inputs=[n, k, l, dt], 
						outputs=cout, 
						givens={c:c0}, 
						updates=updates,
						allow_input_downcast=True)

	print "running sim..."
	start = time.clock()
	cout = sim(n0, k0, l0, dt0)
	diff = (time.clock() - start)
	print "done in", diff, "s at ", diff/num_samples, "s per path"
	downsample_factor_t = int(0.1/dt0) #always show 10 points per time unit
	downsample_factor_p = num_samples/50
	x = np.linspace(0, total_time, total_steps/downsample_factor_t)
	plt.subplot(211)
	plt.plot(x, cout[::downsample_factor_t, ::downsample_factor_p])
	plt.subplot(212)
	bins = np.linspace(0, 1.2, 50)
	plt.hist(cout[int(1/dt0)], bins, alpha = 0.5, 
				normed=True, histtype='bar',  
				label=['Time one'])
	plt.hist(cout[int(2/dt0)], bins, alpha = 0.5, 
				normed=True, histtype='bar',  
				label=['Time two'])
	plt.hist(cout[-1], bins, alpha = 0.5, 
				normed=True, histtype='bar',  
				label=['Time eight'])
	plt.legend()
	plt.show()