"""
George, sampling and marginalization for the hyperparameters

Uses emcee to co
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
        lnprior=0.0
        kernel.pars=np.exp(p)
        return lnprior+ gp.lnlikelihood(mag, quiet=True)

    nwalkers, ndim = 10, len(kernel)
    sampler=emcee.EnsembleSampler(nwalkers, ndim, lnprob)
   
    p0= [np.log(kernel.pars) + 1e-4 * np.random.randn(ndim) 	for i in range(nwalkers)]
    print "Running burn-in"
    st=time()
    p0, _, _ =sampler.run_mcmc(p0, 2000)
    end=time()
    print "It took", end-st, "to burn in "
    print "Running produciton chain "

    
    sampler.run_mcmc(p0, 2000)
    
    for i in range(50):
        w = np.random.randint (sampler.chain.shape[0])
        n = np.random.randint (2000, sampler.chain.shape[1])

        gp.kernel.pars  = np.exp(sampler.chain[w, n])
	plt.errorbar(ph, mag, magerr, fmt='ro')
        plt.plot(t, gp.sample_conditional(mag,t), "k", alpha=0.3)
    plt.show()


main()
