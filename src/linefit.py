"""
Linefitting using a Gaussian Process and the kernels as defined in the George docs
"""
import george
import numpy as np
import sys
import scipy.optimize as op
import matplotlib.pyplot as plt 


from george import kernels


def main():
    k1=66.0**2 * kernels.ExpSquaredKernel(67.0**2)
    k2= 2.4**2 * kernels.ExpSquaredKernel(90**2) * kernels.ExpSine2Kernel(2.0/1.3 ** 2, 1.0)

    k3 = 0.66**2 * kernels.RationalQuadraticKernel (0.78, 1.2**2)

    k4 = 0.18 ** 2 * kernels.ExpSquaredKernel(1.6**2) + kernels.WhiteKernel(0.19)

    kernel= k2+k4#k1 + k2 + k3 + k4

    gp=george.GP(kernel)
    
    indata=np.loadtxt('/Users/lapguest/newbol/bol_ni_ej/out_files/err_bivar_regress.txt', usecols=(5, 6, 1, 2), skiprows=1)
    
    
    def nll(p):
        # Update the kernel parameters and compute the likelihood.
        gp.kernel[:] = p
        ll = gp.lnlikelihood(indata[:,2], quiet=True)
        
        # The scipy optimizer doesn't play well with infinities.
        return -ll if np.isfinite(ll) else 1e25
            # And the gradient of the objective function.
    def grad_nll(p):
                # Update the kernel parameters and compute the likelihood.
        gp.kernel[:] = p
        return -gp.grad_lnlikelihood(indata[:,2], quiet=True)
            #ph=lc['MJD']-tbmax
            #condition for second maximum
            ##TODO: GUI for selecting region
            #cond=(ph>=10.0) & (ph<=40.0)
            #define the data in the region of interest
          
            
            
            
            
    print "Fitting with george"
            
            #print max(mag)
            
            
            # Pre-compute the factorization of the matrix.
    gp.compute(indata[:,0], indata[:,2])
    #print gp.lnlikelihood(mag), gp.grad_lnlikelihood(mag)
    gp.compute(indata[:,0], indata[:,2])        
   
    if sys.argv[1] == 'mle':
        p0=gp.kernel.vector
        results=op.minimize(nll, p0, jac=grad_nll)
        
        gp.kernel[:]=results.x
            
            
    print  gp.kernel.value

    t2=indata[:,0]
    t=np.linspace(t2.min(), t2.max(), 100)

    mu, cov = gp.predict(indata[:,1], t)

    print gp.predict(indata[:,1], 31.99)[0]
    std=np.sqrt(np.diag(cov))

    plt.plot(t2, indata[:,2], 'bo')

    plt.plot(t, mu, 'r:', linewidth=3)
    plt.fill_between(t, mu-std, mu+std, alpha=0.3)
    plt.show()
main()