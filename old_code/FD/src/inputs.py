###############################################################################
## Basic System inputs
N = 8 # number of neutrons
Z = 8 # number of protons
A = N + Z
interaction = 'bkn' # interaction functions ()
coulomb = True # include coulomb
initial_wf = 'HO' # initial single particle wavefunctions for static solution

#nmax = 2 # set max principle number for each nucleus manually for now.
#lmax = 2

###############################################################################
## Interaction parameters ( This will change depeneding on the functional.)
###############################################################################


Vls = 22 - 14*(N - Z)/A# spin orbit strength taken from Bohr and Mottelson
V0 = -51 + 33*(N - Z)/(N + Z) # MeV woods-saxon strength

###############################################################################
#Spectral parameters
###############################################################################
spec_basis = 'chebyshev'
beta = 1.0 # the stretching parameter for the arcsin coordinate transform of GL points. Needed for cheby. polys


## Domain properties
#step_size = .2 ## for finite difference schemes
N_Col = 200 # number of collocation Gauss-Lobatto points.
lb = 10**(-3)# left boundary
rb = 20 # right boundary

###############################################################################
## Self Consistent Solve parameters
###############################################################################
niter = 20 # number of self consistent iterations
sigma = .5 #mixing parameter sigma \in [0,1]. simga = 0.0 means nothing happens.

###############################################################################
## Propagation parameters
###############################################################################
prop_order = 6 #Expansion order of the propagator e^(-i \Delta t h(t)).

delta_t = 1.0*10**(-5) # time step length. rule of thumb is delta_t < 1/N^(1.5)
nt_steps = 10 #number of time steps

###############################################################################
## Define parameter arrays
e2 = 1.4399784 # e^2 charge in MeV fm
hb2m0 = 20.735530 # rest mass of hbar^2 /2m
Rp = 0.831 # radius of the proton in fm. used for saturation density

V = -51.+33.*(N-Z)/nt
vws[1] = -51.-33.*(N-Z)/nt

params = {'V0': V0,'R':R,'r0':r0,'a':a,'e2':e2,'r_cutoff': r_cutoff,'Vls': Vls,\
          'Z':Z,'Nneu':Nneu,'hb2m0':hb2m0,'kappa':kappa}


