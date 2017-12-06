#!/usr/bin/env python2
# -*- coding: utf-8 -*-
"""
Created on Wed Dec  6 13:23:54 2017
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
def evolve(c, s, k, l):
	return T.cast((1-l)*T.pow(s, 2)/(T.pow(s, 2)+T.pow(k,2)) + l - c,'float32')
 
def average(c, r, g):
	n = c.shape[0]
	return T.cast(r*T.sum(c)/n + g*c,'float32')
 
def system(c, s, a, k, l, r, g, dt):
	return [T.cast(c + dt*evolve(c, s, k, l) + T.sqrt(dt)*c*rv_n,'float32'), 
		T.cast(average(c, r, g),'float32'), 
		T.cast(T.sum(c)/c.shape[0],'float32')]
 
if __name__ == '__main__':
	#random
	srng = RandomStreams(seed=31415)
 
	#define symbolic variables
	dt = T.fscalar("dt")
	k = T.fvector("k")
	l = T.fscalar("l")
	r = T.fscalar("r")
	g = T.fscalar("g")
	c = T.fvector("c")
	s = T.fvector("s")
	a = T.fscalar("a")
 
	#define numeric variables
	n_cells = 10
	c0 = theano.shared(np.ones(n_cells, dtype='float32')*0.05)
	s0 = theano.shared(np.ones(n_cells, dtype='float32'))
	k0 = np.random.normal(loc = 0.3, scale = 0.2, size = n_cells)
	l0 = 1/2
	r0 = 0.8
	g0 = 0.4
	dt0 = 0.01
	total_steps = 500
	rv_n = srng.normal(c.shape, std=0.1) #is a shared variable
 
	#create loop
	#first symbolic loop with everything
	([cout, sout, aout], updates) = theano.scan(fn=system,
											outputs_info=[c,s,a], #output shape
											non_sequences=[k,l,r,g,dt], #fixed parameters
											n_steps=total_steps)
	#compile it
	sim = theano.function(inputs=[a, k, l, r, g, dt], 
						outputs=[cout, sout, aout], 
						givens={c:c0, s:s0}, 
						updates=updates,
						allow_input_downcast=True)
 
	print "running sim..."
	start = time.clock()
	[cout, sout, aout] = sim(0, k0, l0, r0, g0, dt0)
	diff = (time.clock() - start)
	print "done in", diff, "s at ", diff/n_cells, "s per path"
	x = np.linspace(0, total_steps*dt0, total_steps)
	plt.subplot(311)
	plt.plot(x, cout)
	plt.subplot(312)
	plt.plot(x, sout)
	plt.subplot(313)
	plt.plot(x, aout)
	plt.show()