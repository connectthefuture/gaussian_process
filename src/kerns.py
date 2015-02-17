"""
Gaussian Process light curve fits for SNIa (NIR, t2 fits )

To use on CSP (Contreras+2010 & Stritzinger+2011), but also for the upcoming CfA data release (Friedman+ 2014)

Based on george (Dan Foreman-Mackey)

Q1: Why do I even have the first two command line arguments?


"""

#package that contains the proper light curve reading functions

from pack import dist	


import emcee 
import numpy as np
import matplotlib.pyplot as plt
import sys
import george
import scipy.optimize as op

from george.kernels import ExpSquaredKernel, WhiteKernel, ExpKernel, ConstantKernel, RationalQuadraticKernel, Matern32Kernel, Matern52Kernel


pt='/Users/lapguest/all_paper/files_snpy/'
def set_arr(insn, inband):
	"""
	Read the light curve 
	"""	
	lc=dist.rd_lc(insn, inband)	
	return lc

def read_inp_lc(fname):
	fin=open(fname, 'r')
	ls=[]
	for row in fin:
		ls.append(row.split())
	ls=np.array(ls)
	lssub=ls[32:-1]
	lc=[]
	for ll in lssub:
		if ll[2]=='i':
			lc.append([float(ll[1]), float(ll[4]), float(ll[5])])	
	return np.array(lc)



#define model, likelihood, prior and posterior if you want to do a full Bayesian treatment 

#good for correlated noise. If you don't have correlated skip to the end 

def model1(params, t):
    m, b, amp, loc, sig2 = params
    return m*t + b + amp * np.exp(-0.5 * (t - loc) ** 2 / sig2)

def lnlike1(p, t, y, yerr):
    return -0.5 * np.sum(((y - model1(p, t))/yerr) ** 2)

def lnprior1(p):
    m, b, amp, loc, sig2 = p
    if (-10 < m < 10 and  -10 < b < 10 and -10 < amp < 10 and
            -5 < loc < 5 and 0 < sig2 < 3):
        return 0.0
    return -np.inf

def lnprob1(p, x, y, yerr):
    lp = lnprior1(p)
    return lp + lnlike1(p, x, y, yerr) 
#negative log likelihood function



def main():
	"""
	Meat of the code

	"""
	insn=sys.argv[1]
	inband=sys.argv[2]
	par=np.loadtxt(pt+'tmax_dm15.dat', dtype='string')
        lc=read_inp_lc('../files/SDSS_SN00013689.DAT')
	print lc[0]
	#return 0
	lc=np.loadtxt('../files/SN13689_data.DAT')
	mag1=lc[:,1]	
	ph1=lc[:,0]
	mag=lc[:,1]; magerr=lc[:,2]
	magerr/=max(mag)	
	mag/=max(mag)

	# Set up the Gaussian process.
	param=float(sys.argv[3])
	kernel = RationalQuadraticKernel(param, 100)+Matern32Kernel(param)#ExpSquaredKernel(param)
	gp = george.GP(kernel)
	#lc=set_arr(insn, inband)

	def nll(p):
                # Update the kernel parameters and compute the likelihood.
                gp.kernel[:] = p
                ll = gp.lnlikelihood(mag, quiet=True)

                # The scipy optimizer doesn't play well with infinities.
                return -ll if np.isfinite(ll) else 1e25
        # And the gradient of the objective function.
        def grad_nll(p):
    # Update the kernel parameters and compute the likelihood.
                gp.kernel[:] = p
                return -gp.grad_lnlikelihood(mag, quiet=True)
	#ph=lc['MJD']-tbmax
	#condition for second maximum 
	##TODO: GUI for selecting region
	#cond=(ph>=10.0) & (ph<=40.0)
	#define the data in the region of interest

	"""
	ph1=ph[cond]
	
	mag=lc[inband][cond]

	magerr=lc['e_'+inband][cond]
	"""

	

	
	print "Fitting with george"

	#print max(mag)
	
	
	# Pre-compute the factorization of the matrix.
	gp.compute(ph1, magerr)
        print gp.lnlikelihood(mag), gp.grad_lnlikelihood(mag)

        gp.compute(ph1, magerr)
        if sys.argv[4] == 'mle':
        	p0=gp.kernel.vector
        	results=op.minimize(nll, p0, jac=grad_nll)

        	gp.kernel[:]=results.x
        
        
        print gp.lnlikelihood(mag), gp.kernel.value
	#vertically stack it for no apparent reason 
	arr=np.vstack([ph1, mag, magerr]).T
	
	#define x array for applying the fit
	t = np.linspace(ph1.min(), ph1.max(), 500)

	#print t.min()#gp.lnlikelihood(mag)
	mu, cov = gp.predict(mag, t)
	
	#condition for peak
	mpeak=(mu==min(mu))
	
	#calculate the standard deviation from covariance matrix 
	std = np.sqrt(np.diag(cov))
	
	#as a check for parameters, print the array at max out
	print t[mpeak][0], max(mu), std[mpeak][0]

	#return 0
	arr=np.vstack([t, mu, std]).T

	np.savetxt("mle_gpfit_SDSS_new.txt", arr)
	plt.errorbar(ph1, mag, magerr, fmt=".k", capsize=2, label='data')
	
	plt.plot(t, mu, 'k:', label='best fit')
	plt.fill_between(t, mu-std, mu+std, alpha=0.3)
	plt.legend(loc=0)

	
	#plt.ylim(plt.ylim()[::-1])
	plt.xlabel("MJD")
	plt.ylabel("flux")
	plt.savefig("mle_SDSS_gpfit_george.pdf")
	plt.show()
if len(sys.argv)==5:
	main()
else:
	print "Usage: python"+sys.argv[0]+"<SN> <band> <kern param> <maximum likelihood or not>"

#some additional sampling code
"""
	initial = np.array([0, 0, -1.0, 0.1, 0.4])
	ndim = len(initial)
	nwalkers=100

	p0 = [np.array(initial) + 1e-8 * np.random.randn(ndim)
      	for i in xrange(nwalkers)]
	sampler = emcee.EnsembleSampler(nwalkers, ndim, lnprob1, args=(ph1, mag ,magerr))

	print("Running burn-in...")
	p0, _, _ = sampler.run_mcmc(p0, 500)
	sampler.reset()

	print("Running production...")
	sampler.run_mcmc(p0, 1000)
	"""

