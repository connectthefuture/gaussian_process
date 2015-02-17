"""
Computing a realisation from a family of functions for a 
Gaussian Process. 
(Emille requires this for computation , its also a good exercise to learn the method from scratch)
Depends on george, which requires eigen


Notes (17/02)
--> not as computationally intensive as PyMC (probably due to no MCMC sampling and simple frequentist statistics)
"""
import numpy as np
import george
import emcee
import sys
import matplotlib.pyplot as plt

#import the exponential squared kernel. (Can also do Rational Quadratic and Matern; see RW2006 for more theory)
from george.kernels import ExpSquaredKernel


from time import time

def main():
	"""
	Meat of the code

	"""
	#insn=sys.argv[1]
	#inband=sys.argv[2]
	#par=np.loadtxt(pt+'tmax_dm15.dat', dtype='string')
        #lc=read_inp_lc('SDSS_SN00013689.DAT')
	
	#return 0

	pt=sys.argv[2]
    #input the data 
	lc=np.loadtxt(pt+'SN13689_data.DAT')
	mag1=lc[:,1]
	ph1=lc[:,0]
	mag=lc[:,1]; magerr=lc[:,2]

    	#normalize it
    	magerr/=max(mag)
	mag/=max(mag)

	# Set up the Gaussian process.
	param=float(sys.argv[1])	
	kernel = ExpSquaredKernel(param)
	gp = george.GP(kernel)
	
	#lc=set_arr(insn, inband)

	
	start=time()

	

	
	print "Fitting with george"

	#print max(mag)
	
	
	# Pre-compute the factorization of the matrix.
	

        gp.compute(ph1, magerr)
	
	t = np.linspace(ph1.min(), ph1.max(), 500)
	#predict the GP fit	
	mu, cov = gp.predict(mag, t)
	std=np.sqrt(np.diag(cov))
	


	real=gp.sample_conditional(mag, ph1)

	#prints  realisation length (should equal data points)
	print np.shape(real), len(mu)
	assert len(real) ==  len(ph1)
	end=time()
	print "start to end it took", end-start
	plt.errorbar(ph1, mag, magerr, fmt=".k", capsize=2, label='data')
	plt.plot(ph1, real, 'r:', linewidth=3, label='realisation', )
	plt.plot(t, mu, 'k:', label='best fit')
	plt.fill_between(t, mu-std, mu+std, alpha=0.3)
	plt.legend(loc=0)
	plt.xlabel('MJD(days)')
	plt.ylabel('Normalised flux ')
	plt.show()
main()
