import numpy as np
import sys
import scipy as sci
from scipy import special
import matplotlib.pyplot as plt
import matrix
import potentials
# import 1d solvers
sys.path.insert(0, '../solvers')
import utilities
import spec
import solvers
def getExactLambda(n,mass,alpha):
    '''
    Exact eigenvalues of the HO equation. -f''(x) + alpha x^2 f(x) = 2 m E f(x)
    Used for checking the solution method
    Lambda = 2 m E
    E = (n + .5) \omega
    \alpha = m^2 \omega^2
    \omega = \sqrt{alpha/m^2}
    Parameters
    ----------
    n : float
        principle quantum number. float integers
    alpha: float
        oscillator width
    mass: float
        mass of particle in well
        
    Returns
    -------
    float
        oscillator energy 2mE.
    '''
    return 2*mass*(.5 + n)*np.sqrt(alpha/mass**2)
def getPsi_x(n,alpha):
    '''
    Definition of exact HO wavefunction taken from Zettili page 240.
    Used for checking the solution method
    Parameters
    ----------
    n : TYPE
        principle quantum number for SE equation
    alpha : TYPE
        harmonic oscillator parameter (mass*omega)^2 from the potential term \mu^2 \omega^{2} x^{2} in the SE.
    Returns
    -------
    wf : function
        1-d wavefunction for the 1d-harmonic oscillator as a function of position x.
    '''
    herm = special.hermite(n)
    def wf(x):
        result = (1/np.sqrt(np.sqrt(np.pi)*2**(n)*np.math.factorial(n)*(alpha)**(.25)))*np.exp(-x**(2)*np.sqrt(alpha)/2)*herm(x*alpha**(.25))
        return(result)
    return wf
'''
Fourier basis seems to produce unstable time evolution beyond N=100
'''
###############################################################################
## Self Consistent Solve parameters
# Note as you turn up the coupling q on the non-linear term, you will probably
# need more self consisten iterations
niter = 50 # number of self consistent iterations

## WARNING: MIXING DOESN'T SEEM TO BE WORKING
## Sigma is the mixing parameter. sigma = 1.0 means full replacement. 
## The mixing currently does not seem to be working
sigma = 1.0

###############################################################################
## Propagation parameters
prop_order = 6 #Expansion order of the propagator e^(-i \Delta t h(t)).
delta_t = 10**(-3) # time step length rule of thumb is delta_t < 1/N^(1.5)
nt_steps = 10**(5) #number of time steps

###############################################################################
## Domain properties
#step_size = .2 ## for finite difference schemes
N = 100 # number of collocation points.
lb = -5 # left boundary
rb = 5 # right boundary

###############################################################################
## Interaction parameters
mass = 1.0
alpha = 1.0 # interaction strength for HO potential (alpha*x^2)
q = 1.0 # interaction strength for |psi|^2 term in GP Hamiltonian

###############################################################################
e2 = 1.4399784 # e^2 charge in MeV fm
hb2m0 = 20.735530
params = {'mass':mass,'nt_steps':nt_steps,'niter':niter,'hb2m0':hb2m0,'e2':e2,'alpha':alpha,\
          'q':q,'sigma':sigma}


###############################################################################
currentInterval = (0,2*np.pi) # interval the derivatives are defined on. (we start here)
targetInterval = (lb,rb) ## interval on real space to solve on

## Define fourier GaussLobatto points in the interval [0,2pi]
GL_func = spec.GaussLobatto()
CPnts,weights = GL_func.fourier(N)


coord_trans = spec.coord_transforms(currentInterval,targetInterval)
CPnts_mapped,dgdy_affine = coord_trans.inv_affine_gen(CPnts)
D_1 = spec.DerMatrix().getFourier(CPnts)/dgdy_affine

# define integration weights
int_weights = spec.GaussLobatto().getDx(CPnts)*dgdy_affine

# make HO potential grid
V_HO_grid = potentials.V_HO(CPnts_mapped,alpha=params['alpha'])

## initialize psi and psiStar with HO solutions
psiGet = getPsi_x(0,alpha=params['alpha'])
psi_HO_exact = psiGet(CPnts_mapped)
psiStar_HO_exact = np.conjugate(psi_HO_exact)

psi_norm = utilities.normalize(psi_HO_exact,int_weights)
psi_HO_exact = psi_HO_exact/psi_norm

E_GS = getExactLambda(0,mass=params['mass'],alpha=params['alpha'])

### check solver against HO solutions
H_constructor = matrix.spec_H_func(N, CPnts_mapped,D_1,params)

H_HO = H_constructor.HO(psi_HO_exact,psi_HO_exact,BC=True) #the psis don't do anything here
evals,evects = sci.linalg.eig(H_HO) #linalg.eigh doesn't work here since the derivative matrices are not hermitian
idx = evals.argsort()
evals = evals[idx]
evects = evects[:,idx]
evects = evects.T

sol = np.concatenate([[0],np.real(evects[0]),[0]])
E_GS_numerical = evals[0]

sol_norm = utilities.normalize(sol,int_weights)
sol = sol/sol_norm
print('Harmonic Oscillator Check:')
print(50*'=')
print(f'Numerical Harmonic Oscillator E_GS: {np.real(E_GS_numerical)}')
print('L2 Difference of HO ground state WF: ',np.linalg.norm(psi_HO_exact-np.abs(sol))) #abs is there since the numerical solver solves up to a sign
print('Difference in HO ground state energy: ', np.abs(E_GS - E_GS_numerical))
print(50*'=')
plt.plot(CPnts_mapped,sol,label='Numerical Ground State',color='red')
plt.plot(CPnts_mapped,psi_HO_exact,label='Exact Ground State')
plt.plot(CPnts_mapped,V_HO_grid,label='HO Potential')
plt.legend()
plt.ylim([0,1.5])
plt.xlim([-3,3])
plt.show()


#################################################################################

### Now solve the nonlinear GP problem self-consistently
H_constructor = matrix.spec_H_func(N, CPnts_mapped,D_1,params)
H_func = H_constructor.GP_HO
E_gsSeries, psiSeries = solvers.MatrixSolve_SC_hermitian(H_func,psi_HO_exact,psiStar_HO_exact,int_weights,params)

print('GP Results :')
print(50*'=')
print(f'Final E_GS: {E_gsSeries[-1]}')
'''
for i in range(len(psiSeries))[::2]:
    plt.plot(CPnts_mapped,psi_HO_exact,label='Exact HO GS')
    plt.plot(CPnts_mapped,np.abs(psiSeries[i]),label='Numerical GP GS',color='red')
    plt.plot(CPnts_mapped,V_HO_grid,label='HO Potential')
    plt.legend()
    plt.title(f'iteration: {i}')
    plt.ylim([0,1.5])
    #plt.xlim([-3,3])
    plt.show()
'''
plt.plot(np.arange(0,niter),E_gsSeries,linestyle='-',marker='o')
plt.xlabel('Self-consistent Iterations')
plt.ylabel('E_GS')
plt.show()

################################################################################
### Time Propagation
################################################################################
## intialize the t = 0 wavefunction
psi_0 =  psiSeries[-1].copy()
psi_series_forward = np.zeros((nt_steps+1,len(CPnts)),dtype=complex) # -2 for len since we remove boundary
psi_series_forward[0] = psi_0
bndyErr = []
for i in range(nt_steps):
    H =  H_func(psi_series_forward[i],psi_series_forward[i],BC=True)
    # compute predictor step using chebyshev expansion of the propagator
    psi_predict = solvers.prop_cheb(psi_series_forward[i],H,dt=0.5*delta_t,prop_order=prop_order,weights=int_weights)
    #psi_predict = solvers.prop(psi_series_forward[i],H,dt=0.5*delta_t,prop_order=prop_order,weights=int_weights)
    
    # recompute the Hamiltonian with the predictor WFs
    H = H_func(psi_predict,psi_predict,BC=True)
    
    # Now compute the full time step.
    psi_series_forward[i+1] = solvers.prop_cheb(psi_series_forward[i],H,dt=delta_t,prop_order=prop_order,weights=int_weights)
    #psi_series_forward[i+1] = solvers.prop(psi_series_forward[i],H,dt=delta_t,prop_order=prop_order,weights=int_weights)
    
    if i % 500 == 0:
        plt.plot(CPnts_mapped,np.real(psi_series_forward[i+1]),label='real')
        plt.plot(CPnts_mapped,np.imag(psi_series_forward[i+1]),label='img')
        plt.plot(CPnts_mapped,np.abs(psi_series_forward[i+1]),label='rho')
        plt.plot(CPnts_mapped,V_HO_grid,label='HO Potential')
        plt.title(f't = {round(i*delta_t,4)}')
        plt.ylim([-1.5,1.5])
        #plt.xlim([-3,3])
        plt.legend()
        plt.show()
    
