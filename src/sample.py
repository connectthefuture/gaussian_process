"""
George, sampling and marginalization for the hyperparameters

Uses emcee to sample the posterior distribution and obtain a set of realisations 

--> this approach allows you to take into account the uncertainties in the hyperparameters, which may not be well constrained by the data, which could be the case here. 

"""
import george
import numpy as np
import scipy.optimize as op
import matplotlib.pyplot as plt
import sys
import emcee

from george.kernels import ExpSquaredKernel
from time import time

def main():
    print "Running fit"
    param=float(sys.argv[1])
    kernel=ExpSquaredKernel(param)
   
    gp=george.GP(kernel)
    lc=np.loadtxt('../files/SN13689_data.dat')
    ph=lc[:,0]; mag=lc[:,1]; magerr=lc[:,2]
    magerr/=max(mag);mag=mag/max(mag)
    
    gp.compute(ph, magerr)

    t=np.linspace(ph.min(), ph.max(), 500)
    
    def lnprob(p):
        if np.any((-10 > p) + (p > 10)):
            return -np.inf
        lnprior=float(sys.argv[2])
        kernel.pars=np.exp(p)
        return lnprior+ gp.lnlikelihood(mag, quiet=True)
    #setup the sampler
    nwalkers, ndim = 10, len(kernel)
    sampler=emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   

    #initialise the walkers
    p0= [np.log(kernel.pars) + 1e-4 * np.random.randn(ndim) 	for i in range(nwalkers)]
    print "Running burn-in"
    st=time()
    p0, _, _ =sampler.run_mcmc(p0, 2000)
    end=time()
    print "It took", end-st, "to burn in "
    print "Running produciton chain "

    
    sampler.run_mcmc(p0, 2000)
    prod=time()
    print 'it took', prod-st, 'seconds'    
    param_arr=[]
    for i in range(50):
        w = np.random.randint (sampler.chain.shape[0])
        n = np.random.randint (2000, sampler.chain.shape[1])

        gp.kernel.pars  = np.exp(sampler.chain[w, n])
	#param_arr.append(gp.kernel.value)
	plt.errorbar(ph, mag, magerr, fmt='ro')
        plt.plot(t, gp.sample_conditional(mag,t), "k", alpha=0.3)
    
    print 'The kernel parameter is:', gp.kernel.value
    plt.savefig('../img/gaussfit_margin_'+sys.argv[1]+'_'+sys.argv[2]+'.png')
    plt.show()
	

if len(sys.argv) == 3:
	main()
else:
	print "Usage: python", sys.argv[0], '<input parameter> <lnprior>'
