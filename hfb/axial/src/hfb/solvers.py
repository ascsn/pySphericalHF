import numpy as np
from scipy import special
import sys, os
import math

import warnings
import traceback
import tqdm

# from jax import jit, lax
# import jax
# jax.config.update("jax_enable_x64", True)
# import jax.numpy as jnp
from functools import partial

import misc_physics as misc_phys
import utils

import itertools
import pandas as pd
#See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.copy_on_write = True

import time
from scipy import optimize, linalg
import h5py

import matplotlib.pyplot as plt

warnings.filterwarnings('ignore',message='Calling float on a single element Series')

class CylindricalIntegral:
    def __init__(self,xi,eta,wz,wp):
        self.xi = xi
        self.eta = eta
        self.wz = wz
        self.wp = wp
        
    def integrate(self,arr,bz,bp):
        #Compiling this with jit slows down computations considerably, maybe
        #from converting data types
        #Also, I just figured out that this works as long as arr.shape ends in
        #( len(self.wp), len(self.wz) ) - even if there's other dimensions
        return np.pi*bz*bp**2 * (self.wp @ arr @ self.wz)

class Normalization(CylindricalIntegral):
    def __call__(self,rho,bz,bp):
        return self.integrate(rho,bz,bp)
    
class AxialMultipoleMoment(CylindricalIntegral):
    def __init__(self,*args):
        super().__init__(*args)
        
        self.ee, self.xx = np.meshgrid(self.eta,self.xi)
        
        self._cache = {}
        
    def _coeff(self,l):
        if l == 0:
            coeff = np.sqrt(4*np.pi)
        elif l == 1:
            coeff = np.sqrt(4*np.pi/3)/10
        elif l == 2:
            coeff = np.sqrt(16*np.pi/5)/100
        else:
            coeff = 1/10.**l
        return coeff
    
    def __call__(self,l,inputArr,bz,bp):
        """
        The array is cached because I need to call this a large number of times
        when updating my Lagrange multipliers. I don't see anywhere else it makes
        sense to cache this array.
        
        I don't really know how to free the cache memory, but it's totally
        unnecessary, because each constraint is (for the default precision we're
        using) only 3200 values
        """
        
        #Try-catch is about as fast as an if-else statement, except the
        #profiler like this method more
        try:
            arr = self._cache[l]
        except KeyError:
            coeff = self._coeff(l)
            
            s = np.sqrt(bp**2*self.ee + bz**2 * self.xx**2)
            
            arr = s**l * special.eval_legendre(l,bz*self.xx/s)
            arr = arr.T * coeff * np.sqrt((2*l+1)/(4*np.pi))
            self._cache[l] = arr
        
        return self.integrate(inputArr*arr,bz,bp)
    
class BaseSkyrme(CylindricalIntegral):
    """
    Note that the method 'rho' computes the variation with respect to $\rho$,
    and so on and so forth
    """
    def __init__(self,*args):
        super().__init__(*args)
        
    def get_energy(self,listOfFields,bz,bp):
        raise NotImplementedError
        
    def rho(self,listOfFields,bz,bp,**kwargs):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def dr_rho(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def dz_rho(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def del_rho(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def tau(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def divJ(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def J_fz(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def J_zf(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def J_fr(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def J_rf(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def rho_tilde(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
class AxialMultipoleConstraint(BaseSkyrme):
    def _coeff(self,l):
        if l == 0:
            coeff = np.sqrt(4*np.pi)
        elif l == 1:
            coeff = np.sqrt(4*np.pi/3)/10
        elif l == 2:
            coeff = np.sqrt(16*np.pi/5)/100
        else:
            coeff = 1/10.**l
        return coeff
    
    def rho(self,listOfFields,bz,bp,l=None,lagrangeMultiplier=None):
        if l is None:
            raise ValueError('Provide an l value')
        if lagrangeMultiplier is None:
            raise ValueError('Provide Lagrange multiplier')
    
        rhop, rhon = listOfFields
        
        coeff = self._coeff(l)
        
        ee, xx = np.meshgrid(self.eta,self.xi)
        s = np.sqrt(bp**2*ee + bz**2 * xx**2)
        
        arr = s**l * special.eval_legendre(l,bz*xx/s)
        
        return 2*(-lagrangeMultiplier*arr.T * coeff * np.sqrt((2*l+1)/(4*np.pi)),)
    
"""
Can be abstracted a bit by supplying a list of fields, each a proton and a neutron,
but that seems excessive right now. Could also be abstracted by a more detailed
base class

"""
class Skyrme_Kinetic(BaseSkyrme):
    def __init__(self,*args,
                 coeff=2*[utils.GlobalVariables.h2m,]):
        """
        The normal kinetic energy density, $\hbar^2/2m \tau$. By default,
        assumes equal proton and neutron mass, with value
        $\hbar^2/2m = 20.73553$ MeV [CITE]
        """
        super().__init__(*args)
        self.coeff = coeff
        
        self.inputs = [['p','tau'],['n','tau']]
        
    def get_eneg(self,listOfFields,bz,bp):
        taup, taun = listOfFields
        return self.coeff[0] * self.integrate(taup,bz,bp) + self.coeff[1] * self.integrate(taun,bz,bp)
    
    def tau(self,listOfFields,bz,bp):
        taup, taun = listOfFields
        return self.coeff[0]*np.ones(taup.shape), self.coeff[1]*np.ones(taun.shape)
    
class Skyrme_rho_rho(BaseSkyrme):
    def __init__(self,B1,B2,*args):
        super().__init__(*args)
        self.B1 = B1
        self.B2 = B2
        
        self.inputs = [['p','rho'],['n','rho']]
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rhoTot = rhop + rhon
        
        return self.B1*self.integrate(rhoTot**2,bz,bp) + self.B2 * \
            self.integrate(rhop**2+rhon**2,bz,bp)
            
    def rho(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rhoTot = rhop+rhon
        
        return 2*self.B1*rhoTot + 2*self.B2*rhop, 2*self.B1*rhoTot + 2*self.B2*rhon
            
class Skyrme_rho_tau(BaseSkyrme):
    def __init__(self,B3,B4,*args):
        super().__init__(*args)
        self.B3 = B3
        self.B4 = B4
        
        self.inputs = [['p','rho'],['n','rho'],['p','tau'],['n','tau']]
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,taup,taun = listOfFields
        rho = rhop + rhon
        tau = taup + taun
        
        return self.B3 * self.integrate(rho*tau,bz,bp) + self.B4 * \
            self.integrate(rhop*taup + rhon*taun,bz,bp)
            
    def rho(self,listOfFields,bz,bp):
        rhop,rhon,taup,taun = listOfFields
        tau = taup+taun
        
        return self.B3*tau + self.B4*taup, self.B3*tau + self.B4*taun
    
    def tau(self,listOfFields,bz,bp):
        rhop,rhon,taup,taun = listOfFields
        
        rho = rhop + rhon
        return self.B3*rho + self.B4*rhop, self.B3*rho + self.B4*rhon
            
class Skyrme_rho_dRho(BaseSkyrme):
    def __init__(self,B5,B6,*args):
        super().__init__(*args)
        self.B5 = B5
        self.B6 = B6
        
        self.inputs = [['p','rho'],['n','rho'],['p','del_rho'],['n','del_rho']]
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,drhop,drhon = listOfFields
        rho = rhop + rhon
        drho = drhop + drhon
        
        return self.B5 * self.integrate(rho*drho,bz,bp) + self.B6*\
            self.integrate(rhop*drhop + rhon*drhon,bz,bp)
            
    def rho(self,listOfFields,bz,bp):
        rhop,rhon,drhop,drhon = listOfFields
        drho = drhop + drhon
        
        return self.B5*drho + self.B6*drhop, self.B5*drho + self.B6*drhon
    
    def del_rho(self,listOfFields,bz,bp):
        rhop,rhon,drhop,drhon = listOfFields
        rho = rhop + rhon
        
        return self.B5*rho + self.B6*rhop, self.B5*rho + self.B6*rhon

class Skyrme_rho_alpha(BaseSkyrme):
    def __init__(self,B7,B8,alpha,*args):
        super().__init__(*args)
        self.B7 = B7
        self.B8 = B8
        self.alpha = alpha
        
        self.inputs = [['p','rho'],['n','rho']]
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rho = rhop + rhon
        
        return self.B7 * self.integrate(rho**(2+self.alpha),bz,bp) + self.B8*\
            self.integrate(rho**self.alpha*(rhop**2 + rhon**2),bz,bp)
            
    def rho(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rho = rhop + rhon
        
        common = (2+self.alpha)*self.B7*rho**(1+self.alpha)
        common += self.alpha*self.B8*rho**(self.alpha-1)*(rhop**2+rhon**2)
        
        return common + 2*self.B8*rho**self.alpha * rhop, common + 2*self.B8*rho**self.alpha * rhon
            
class Skyrme_rho_divJ(BaseSkyrme):
    def __init__(self,B9,B9p,*args):
        super().__init__(*args)
        self.B9 = B9
        self.B9p = B9p
        
        self.inputs = [['p','rho'],['n','rho'],['p','divJ'],['n','divJ']]
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,divJp,divJn = listOfFields
        return self.B9*self.integrate((rhop+rhon)*(divJp+divJn),bz,bp) + \
            self.B9p*self.integrate(rhop*divJp+rhon*divJn,bz,bp)
            
    def rho(self,listOfFields,bz,bp):
        rhop,rhon,divJp,divJn = listOfFields
        divJ = divJp + divJn
        
        return self.B9*divJ + self.B9p*divJp, self.B9*divJ + self.B9p*divJn
    
    def divJ(self,listOfFields,bz,bp):
        rhop,rhon,divJp,divJn = listOfFields
        rho = rhop + rhon
        return self.B9*rho + self.B9p*rhop, self.B9*rho + self.B9p*rhon

class CoulombDirect_Gaussian(BaseSkyrme):
    def __init__(self,*args,nLeg=160,b=50,Iarr=None):
        super().__init__(*args)
        
        self.e2 = utils.GlobalVariables.e2
        
        self.nLeg = nLeg
        #Legendre nodes are $\zeta'$ from my notes
        legNodes, legWeights = np.polynomial.legendre.leggauss(self.nLeg)
        #Want only nodes and weights in interval [0,1], to match HFBTHO
        self.legNodes = legNodes[nLeg//2:]
        self.legWeights = legWeights[nLeg//2:]
        self.b = b
        
        self.zeta = self.legNodes
        self.a = 1/self.b * self.zeta/np.sqrt(1-self.zeta**2)
        
        self.Iarr = Iarr
        
        self.inputs = [['p','rho'],]

    def get_Iarr(self,bz,bp):
        if self.Iarr is None:
            self.bz = bz
            self.bp = bp
            self.Iarr = np.zeros(2*(self.eta.size,)+2*(self.xi.size,))
            
            rVals = np.sqrt(self.eta)*bp
            zVals = self.xi*bz

            d = (rVals[:,None,None,None]-rVals[None,:,None,None])**2 \
                + (zVals[None,None,:,None]-zVals[None,None,None,:])**2
            for (aIter,a) in enumerate(self.a):
                toAdd = np.exp(-d * a**2)
                toAdd *= special.i0e(2*rVals[:,None]*rVals[None,:]*a**2)[:,:,None,None]
                toAdd *= 1/(1-self.zeta[aIter]**2)**(3/2) * self.legWeights[aIter]
                self.Iarr += toAdd
            
            self.Iarr *= self.e2/(self.b*np.sqrt(np.pi))
            self.Iarr = np.swapaxes(self.Iarr,1,2)

        #Iarr changes with harmonic oscillator widths.
        #WARNING: not tested
        if not hasattr(self,'bz'): self.bz = bz
        if not hasattr(self,'bp'): self.bp = bp

        if bz != self.bz or bp != self.bp:
            self.Iarr = None
            self.Iarr = self._get_Iarr(bz,bp)

        return self.Iarr
    
    def _get_field(self,rhop,bz,bp):
        Iarr = self.get_Iarr(bz,bp)
        return self.integrate(Iarr*rhop,bz,bp)
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.integrate(rhop*self._get_field(rhop,bz,bp),bz,bp)
    
    def rho(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return 2*self._get_field(rhop,bz,bp), np.zeros(rhop.shape)
    
class CoulombDirect_Gaussian_Reduced(BaseSkyrme):
    def __init__(self,rhopPSInv,rhopBasis,*args,nLeg=160,b=50,Iarr=None):
        super().__init__(*args)

        self.rhopPSInv = rhopPSInv
        self.rhopBasis = rhopBasis
        
        self.e2 = utils.GlobalVariables.e2
        
        self.nLeg = nLeg
        #Legendre nodes are $\zeta'$ from my notes
        legNodes, legWeights = np.polynomial.legendre.leggauss(self.nLeg)
        #Want only nodes and weights in interval [0,1], to match HFBTHO
        self.legNodes = legNodes[nLeg//2:]
        self.legWeights = legWeights[nLeg//2:]
        self.b = b
        
        self.zeta = self.legNodes
        self.a = 1/self.b * self.zeta/np.sqrt(1-self.zeta**2)
        
        self.Iarr = Iarr
        
        self.inputs = [['p','rho'],]

    def get_Iarr(self,bz,bp):
        if self.Iarr is None:
            self.bz = bz
            self.bp = bp
            self.Iarr = np.zeros(2*(self.eta.size,)+2*(self.xi.size,))
            
            rVals = np.sqrt(self.eta)*bp
            zVals = self.xi*bz

            d = (rVals[:,None,None,None]-rVals[None,:,None,None])**2 \
                + (zVals[None,None,:,None]-zVals[None,None,None,:])**2
            for (aIter,a) in enumerate(self.a):
                toAdd = np.exp(-d * a**2)
                toAdd *= special.i0e(2*rVals[:,None]*rVals[None,:]*a**2)[:,:,None,None]
                toAdd *= 1/(1-self.zeta[aIter]**2)**(3/2) * self.legWeights[aIter]
                self.Iarr += toAdd
            
            self.Iarr *= self.e2/(self.b*np.sqrt(np.pi))
            self.Iarr = np.swapaxes(self.Iarr,1,2)

        #Iarr changes with harmonic oscillator widths.
        #WARNING: not tested
        if not hasattr(self,'bz'): self.bz = bz
        if not hasattr(self,'bp'): self.bp = bp

        if bz != self.bz or bp != self.bp:
            self.Iarr = None
            self.Iarr = self._get_Iarr(bz,bp)

        return self.Iarr
    
    def _get_field(self,rhop,bz,bp):
        Iarr = self.get_Iarr(bz,bp)

        try:
            coeffs = self.rhopPSInv @ rhop[0]
            rhop = coeffs @ np.swapaxes(self.rhopBasis,0,1)
        except ValueError:
            pass

        return self.integrate(Iarr*rhop,bz,bp)
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.integrate(rhop*self._get_field(rhop,bz,bp),bz,bp)
    
    def rho(self,listOfFields,bz,bp):
        rhop, = listOfFields
        fieldOut = 2*self._get_field(rhop,bz,bp)
        return fieldOut, np.zeros(fieldOut.shape)
    
class CoulombDirect_Laplace(BaseSkyrme):
    #TODO: factors of 2 floating about that need to be fixed
    def __init__(self,*args):
        super().__init__(*args)
        warnings.warn('Stop using this - CoulombDirect_Gaussian is now fast!')
        
        self.inputs = [['p','rho'],['p','del_rho']]
        
    def _get_field(self,delRho,bz,bp):
        ret = np.zeros(delRho.shape)
        
        rVals = np.sqrt(self.eta)*bp
        zVals = self.xi*bz
        
        for (j,r) in enumerate(rVals):
            d = (r+rVals[None,:,None])**2 + (zVals[:,None,None]-zVals[None,None,:])**2
            
            ellipticArg = 4*r*rVals[None,:,None]/d
            ellipticEval = special.ellipe(ellipticArg)
            
            ret[j] = self.integrate(delRho*np.sqrt(d)*ellipticEval,bz,bp)
        return 2*utils.GlobalVariables.e2*ret/(2*np.pi)
    
    def get_eneg(self,listOfFields,bz,bp):
        rhop, delRhop = listOfFields
        return self.integrate(rhop*self._get_field(delRhop,bz,bp),bz,bp)/2
        
    def rho(self,listOfFields,bz,bp):
        rhop, delRhop = listOfFields
        
        return self._get_field(delRhop,bz,bp), np.zeros(delRhop.shape)
    
class Approximate_CoulombDirect(BaseSkyrme):
    def __init__(self,coordInds,*args,shp=(40,80)):
        raise NotImplementedError('Needs to be made consistent with CoulombDirect_Gaussian')
        self.coordInds = coordInds
        super().__init__(*args)
        
        self.inputs = [['p','rho'],['p','del_rho']]
        self.shp = shp
        
    def _get_field(self,delRho,bz,bp):
        ret = np.zeros(delRho.shape)
        
        rVals = np.sqrt(self.eta)*bp
        zVals = self.xi*bz
        
        for (j,r) in enumerate(rVals):
            d = (r+rVals[None,:,None])**2 + (zVals[:,None,None]-zVals[None,None,:])**2
            
            ellipticArg = 4*r*rVals[None,:,None]/d
            ellipticEval = special.ellipe(ellipticArg)
            
            ret[j] = self.integrate(delRho*np.sqrt(d)*ellipticEval,bz,bp)
            
        del d #May or may not help
        return 2*utils.GlobalVariables.e2*ret/(2*np.pi)
    
    def get_eneg(self,listOfFields,bz,bp):
        raise NotImplementedError
    
    def rho(self,listOfFields,bz,bp):
        rhop, delRhop = listOfFields
        
        fullDelRhop = np.zeros(self.shp)
        fullDelRhop[self.coordInds] = delRhop
        
        field = self._get_field(fullDelRhop,bz,bp)
        
        # print(field.shape)
        # hfb.plot_field(field,eta,xi)
        # sys.exit()
        return field, np.zeros(delRhop.shape)
    
class CoulombExchange(BaseSkyrme):
    def __init__(self,*args):
        super().__init__(*args)
        
        e2 = utils.GlobalVariables.e2
        self.coeff = -3/4*e2*(3/np.pi)**(1/3)
        
        self.inputs = [['p','rho'],]
    
    def get_eneg(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.coeff * self.integrate(rhop**(4/3),bz,bp)
    
    def rho(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.coeff*4/3*rhop**(1/3), np.zeros(rhop.shape)

class Pairing_delta(BaseSkyrme):
    """
    Warning: factors of 2 are generally inconsistent in the literature. I can't
    guarantee that they're all correct, but I've tried. The pairing form explicitly
    is
    
    $$  E[\rho,\tilde{\rho}] = \sum_q V_0^q \bigg[ 1 - V_1^q \frac{(\rho_p+\rho_n)}{2} \bigg] \times \sum_{q'}\tilde{\rho}_{q'}^2  $$
    
    Often in the literature one sees $V_0^q/2$, but I think HFBTHO does not
    You see that the energy contribution does not sum over $\tilde{\rho}_{q'}^2$. 
    This now agrees with the HFB energy to $10^{-5}$ MeV.
    """
    def __init__(self,V0,V1,g,rhoc,*args):
        super().__init__(*args)
        self.V0 = V0
        self.V1 = V1
        self.g = g
        self.rhoc = rhoc
        
        if g != 1:
            raise NotImplementedError
            
        self.inputs = [['p','rho'],['n','rho'],['p','rho_tilde'],['n','rho_tilde']]
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,rhopPair,rhonPair = listOfFields
        
        # rhoPairSum = rhopPair**2 + rhonPair**2
        # arrp = self.V0[0]*(1-self.V1[0] * (rhop+rhon)/self.rhoc) * rhoPairSum
        # arrn = self.V0[1]*(1-self.V1[1] * (rhop+rhon)/self.rhoc) * rhoPairSum
        
        arrp = self.V0[0]*(1-self.V1[0] * (rhop+rhon)/self.rhoc) * rhopPair**2
        arrn = self.V0[1]*(1-self.V1[1] * (rhop+rhon)/self.rhoc) * rhonPair**2
        
        return self.integrate(arrp+arrn,bz,bp)
    
    def rho(self,listOfFields,bz,bp):
        rhop,rhon,rhopPair,rhonPair = listOfFields
        
        common = -1/self.rhoc*(rhopPair**2 + rhonPair**2)
        return self.V0[0]*self.V1[0]*common, self.V0[1]*self.V1[1]*common
        #Below agrees worse with HFBTHO's results
        # return -self.V0[0]*self.V1[0]/self.rhoc*rhopPair**2, -self.V0[1]*self.V1[1]/self.rhoc*rhonPair**2
    
    def rho_tilde(self,listOfFields,bz,bp):
        rhop,rhon,rhopPair,rhonPair = listOfFields
        
        common = (rhop+rhon)/self.rhoc
        
        #Factors of 2 float around in HFBTHO
        return self.V0[0]*(1-self.V1[0]*common) * rhopPair, \
            self.V0[1]*(1-self.V1[1]*common) * rhonPair
        # return 2*self.V0[0]*(1-self.V1[0]*common) * rhopPair, \
        #     2*self.V0[1]*(1-self.V1[1]*common) * rhonPair
    
class SkyrmeEDF:
    def __init__(self,edfParams,xi,eta,wz,wr,directCoulombObj):
        Mv = 1.24983857423226952
        rhocPair = 0.16

        g, C00pp, C0Dpp, C0pt, C1pt, C10pp, C1Dpp = \
            misc_phys.EDFParams.inm_to_isospin(edfParams['rho0'],
                                               edfParams['Ms_inv'],
                                               Mv,
                                               edfParams['K'],
                                               edfParams['EoA'],
                                               edfParams['a'],
                                               edfParams['L'])

        B = misc_phys.EDFParams.isospin_to_pn(C00pp,C10pp,C0pt,C1pt,edfParams['Crdr0'],edfParams['Crdr1'],C0Dpp,C1Dpp,
                                              edfParams['CrdJ0'],edfParams['CrdJ1'])

        #Pairing parameters (proton, neutron)
        V0 = [edfParams['Vp'],edfParams['Vn']]
        V1 = [0.5,0.5]
        sigma = 1
        
        self.terms = {'kinetic':Skyrme_Kinetic(xi,eta,wz,wr),
                      'rho_rho':Skyrme_rho_rho(B[0],B[1],xi,eta,wz,wr),
                      'rho_tau':Skyrme_rho_tau(B[2],B[3],xi,eta,wz,wr),
                      'rho_dRho':Skyrme_rho_dRho(B[4],B[5],xi,eta,wz,wr),
                      'rho_alpha':Skyrme_rho_alpha(B[6],B[7],g,xi,eta,wz,wr),
                      'coulomb_exchange':CoulombExchange(xi,eta,wz,wr),
                      'coulomb_direct':directCoulombObj,
                      'rho_divJ':Skyrme_rho_divJ(B[8],B[9],xi,eta,wz,wr),
                      'pairing_delta':Pairing_delta(V0,V1,sigma,rhocPair,xi,eta,wz,wr)
                      }
        
        #EDF arguments
        self.integArgs = {key:getattr(term,'inputs') for (key,term) in self.terms.items()}

    def hfb_energy_wrapper(self,densities,bz,bp,display=False):
        eneg = {}
        
        for (key,term) in self.terms.items():
            if display:
                print(term.__class__)
            t0 = time.time()
            integArgs = [densities[tup[0]][tup[1]] for tup in self.integArgs[key]]
            eneg[key] = term.get_eneg(integArgs,bz,bp)
            t1 = time.time()
            if display:
                print(eneg[key],t1-t0)
            
        totalEneg = 0.
        for val in eneg.values():
            totalEneg += val

        eneg['volume'] = eneg['rho_rho'] + eneg['rho_tau'] + eneg['rho_alpha']
        eneg['surface'] = eneg['rho_dRho']
        eneg['coulomb'] = eneg['coulomb_direct'] + eneg['coulomb_exchange']
        eneg['spin-orbit'] = eneg['rho_divJ']
        eneg['pairing'] = eneg['pairing_delta']
        eneg['net'] = eneg['kinetic'] + eneg['volume'] + eneg['surface'] + eneg['coulomb'] + \
            eneg['spin-orbit'] + eneg['pairing']
                
        return eneg
            
def ho_weights_and_nodes(nr=40,nz=80):
    #TODO: should be made part of HarmonicOscillatorBasis
    eta, wr = special.roots_laguerre(nr)
    wr = wr * np.exp(eta)
    
    xi, wz = special.roots_hermite(nz)
    wz = wz * np.exp(xi**2)
    
    return eta, wr, xi, wz
            
class HarmonicOscillatorBasis:
    """ 
    ===========================================================================
    I've checked the basis states against HFBTHO, and they agree perfectly.
    
    Note that HFBTHO stores basis states weighted by the integration weights
    (the radial states also divide by $\sqrt{2}$), so that integrations
    involving the basis functions are actually just a sum. I'm not going to
    do that here, I don't think - not when I've already got an integration
    routine defined in CylindricalIntegral.
    
    The caveat to the above is that the total basis function is divided by
    $\sqrt{2\pi}$, so that it cancels out in the integration routine.
    ===========================================================================
    """
    def __init__(self,bp,bz,hbzero=20.735530000000000,hoMaxQuanta=100):
        self.hoPerp = 2*hbzero/bp**2
        self.hoZ = 2*hbzero/bz**2
        self.hoMaxQuanta = hoMaxQuanta
        
        self.betap = 1/bp
        self.betaz = 1/bz

        self.bp = bp
        self.bz = bz
        
    def sp_eneg(self,nr,nz,lamd,spin):
        return (2*nr + lamd + 1)*self.hoPerp + (nz+0.5)*self.hoZ
    
    def get_quantum_numbers(self,nShells):
        """
        Validated gainst an example from HFBTHO
        
        Not sure this is quite general enough to use elsewhere, although
        any similar code will look like a copy-paste
        """
        nrVals = np.arange(self.hoMaxQuanta)
        nzVals = np.arange(self.hoMaxQuanta)
        lambdaVals = np.arange(self.hoMaxQuanta)

        allVals = np.array(list(itertools.product(nrVals,nzVals,lambdaVals,[-1,1])))

        hoEneg = self.sp_eneg(*allVals.T)

        df = pd.DataFrame(data=allVals,columns=['nr','nz','m_l','2m_s'])
        df['E'] = hoEneg
        df = df.loc[np.argsort(df['E'])]
        df['k'] = df['m_l'] + (df['2m_s'] + 1)//2
        df['parity'] = (df['nz'] + df['m_l']) % 2 + 1

        df['HO-quanta'] = 2*df['nr'] + df['nz'] + df['m_l']

        df = df.reset_index(drop=True)

        df = df[df['HO-quanta']<=self.hoMaxQuanta].reset_index(drop=True)
        df = df[df['k'] > 0].reset_index(drop=True)

        Nvals = (nShells+1) * (nShells+2) * (nShells+3)//6

        maxE = np.sort(df['E'])[Nvals]
        df = df[df['E']<=maxE]

        columnOrder = ['k','nr','nz','parity','m_l','2m_s','E','HO-quanta']
        df = df[columnOrder]

        #Sort this way to have the same ordering as HFBTHO
        self.quantumNumbers = df.sort_values(by=['k','parity','nr','nz'],ignore_index=True)
        return None
    
    def split_by_block(self,r,z):
        quantNumbersByBlock = []
        
        eta = r**2*self.betap**2
        xi = z*self.betaz

        self.nr = len(r)
        self.nz = len(z)
        
        """
        -----------------------------------------------------------------------
        Wavefunctions and their derivatives are defined assuming that the spin
        and isospin components have been handled analytically. Similarly, the
        angular term $ e^{i\phi \Lambda}/\sqrt{2\pi} $ has been factored out,
        and should be handled analytically.
        
        Also: for $ d\Psi/d\phi $, the factor of $i$ is removed analytically,
        and any minus signs that result *must* be handled analytically.
        -----------------------------------------------------------------------
        """        
        #Basis states and derivatives on the coordinate mesh
        psi = []
        #First derivatives - [dr, dphi, dz]
        dpsi = []
        #Second derivatives - [d2r, drdphi, drdz, d2phi, dphidz, d2z]
        d2psi = []
        
        #Array for the angular/spin components when considering div.J
        MbyBlock = []
        
        for k in np.unique(self.quantumNumbers['k']):
            subDf = self.quantumNumbers[self.quantumNumbers['k'] == k]
            quantNumbersByBlock.append(subDf)
            
            psiArr = np.zeros((len(subDf),len(eta),len(xi)))
            dpsiArr = np.zeros((3,len(subDf),len(eta),len(xi)))
            d2psiArr = np.zeros((6,len(subDf),len(eta),len(xi)))
            
            Marr = np.zeros((3,len(subDf),len(subDf)))
            
            for i in range(len(subDf)):
                #Pandas converts some datatypes if you don't index it this way,
                #see e.g. https://stackoverflow.com/questions/41662881/pandas-dataframe-iloc-spoils-the-data-type
                lmd = subDf['m_l'].iloc[i]
                
                psir = self._psir(eta,subDf['nr'].iloc[i],lmd)
                dpsir = self._dpsir(eta,subDf['nr'].iloc[i],lmd)
                d2psir = self._d2psir(eta,subDf['nr'].iloc[i],lmd)
                
                psiz = self._psiz(xi,subDf['nz'].iloc[i])
                dpsiz = self._dpsiz(xi,subDf['nz'].iloc[i])
                d2psiz = self._d2psiz(xi,subDf['nz'].iloc[i])
                
                psiArr[i] = np.outer(psir,psiz)
                
                dpsiArr[0,i] = np.outer(dpsir,psiz)
                dpsiArr[1,i] = lmd*psiArr[i]
                dpsiArr[2,i] = np.outer(psir,dpsiz)
                
                d2psiArr[0,i] = np.outer(d2psir,psiz)
                d2psiArr[1,i] = lmd*dpsiArr[0,i]
                d2psiArr[2,i] = np.outer(dpsir,dpsiz)
                d2psiArr[3,i] = -lmd**2 * psiArr[i]
                d2psiArr[4,i] = lmd*dpsiArr[2,i]
                d2psiArr[5,i] = np.outer(psir,d2psiz)
                
                
                idxEq = np.where(subDf['m_l']==lmd)[0]
                idxPlus = np.where(subDf['m_l']+1==lmd)[0]
                idxMinus = np.where(subDf['m_l']-1==lmd)[0]
                
                if len(idxPlus) > 0:
                    Marr[0,i,idxPlus] = 1
                    Marr[1,i,idxPlus] = 1
                if len(idxMinus) > 0:
                    Marr[0,i,idxMinus] = 1
                    Marr[1,i,idxMinus] = -1
                    
                Marr[2,i,idxEq] = subDf.iloc[i]['2m_s']
                
            psi.append(psiArr)
            dpsi.append(dpsiArr)
            d2psi.append(d2psiArr)
            
            MbyBlock.append(Marr)
            
        self.psi = psi
        self.dpsi = dpsi
        self.d2psi = d2psi
        self.M = MbyBlock
        self.quantNumbersByBlock = quantNumbersByBlock
        self.nBlocks = len(self.quantNumbersByBlock)
        
        return None
            
    def _psir(self,eta,nr,lambd):
        N = np.sqrt(math.factorial(nr)/math.factorial(nr+lambd))
        
        ret = eta**(lambd/2) * np.exp(-eta/2)
        ret *= special.eval_genlaguerre(nr,lambd,eta)
        
        return N * self.betap * np.sqrt(2) * ret
    
    def _dpsir(self,eta,nr,lambd):
        N = np.sqrt(math.factorial(nr)/math.factorial(nr+lambd))
        
        lagEval1 = special.eval_genlaguerre(nr,lambd,eta)
        lagEval1 *= (lambd - eta)
        
        lagEval2 = special.eval_genlaguerre(nr-1,lambd+1,eta)
        lagEval2 *= -2*eta
        
        coeff = self.betap/np.sqrt(2)*N*np.exp(-eta/2)*eta**(lambd/2-1)
        
        r = np.sqrt(eta)/self.betap
        return coeff*(lagEval1 + lagEval2)*2*r*self.betap**2
    
    def _d2psir(self,eta,nr,lambd):
        N = np.sqrt(math.factorial(nr)/math.factorial(nr+lambd))
        
        r = np.sqrt(eta)/self.betap
        dEtaDr = 2*r*self.betap**2
        d2EtaDr2 = 2*self.betap**2
        
        lagEval1 = special.eval_genlaguerre(nr,lambd,eta)
        poly1 = ( (lambd-2)*lambd-2*lambd*eta+eta**2 )*dEtaDr**2
        poly1 += -2*eta*(-lambd+eta)*d2EtaDr2
        lagEval1 *= poly1
        
        lagEval2 = special.eval_genlaguerre(nr-1,lambd+1,eta)
        poly2 = -4*lambd*eta*dEtaDr**2 + 4*eta**2*dEtaDr**2 - 4*eta**2 * d2EtaDr2
        lagEval2 *= poly2
        
        lagEval3 = special.eval_genlaguerre(nr-2,lambd+2,eta)
        poly3 = 4*eta**2*dEtaDr**2
        lagEval3 *= poly3
        
        coeff = 1/(2*np.sqrt(2)) * np.exp(-eta/2)*self.betap * N*eta**(lambd/2-2)
        return coeff * (lagEval1 + lagEval2 + lagEval3)
    
    def _psiz(self,xi,nz):
        N = np.sqrt(1/(np.sqrt(np.pi) * 2.**nz * math.factorial(nz)))
        
        ret = np.exp(-xi**2/2) * special.eval_hermite(nz,xi)
        
        return N * np.sqrt(self.betaz) * ret
    
    def _dpsiz(self,xi,nz):
        N = np.sqrt(1/(np.sqrt(np.pi) * 2.**nz * math.factorial(nz)))
        
        if nz == 0:
            hermEval1 = np.zeros(xi.shape)
        else:
            hermEval1 = special.eval_hermite(nz-1,xi)
        hermEval1 *= 2*nz
        
        hermEval2 = special.eval_hermite(nz,xi)
        hermEval2 *= -xi
        
        coeff = np.exp(-xi**2/2) * np.sqrt(self.betaz) * N
        dXiDz = self.betaz
        return  coeff * (hermEval1 + hermEval2) * dXiDz
    
    def _d2psiz(self,xi,nz):
        N = np.sqrt(1/(np.sqrt(np.pi) * 2.**nz * math.factorial(nz)))
        
        dXiDz = self.betaz
        
        hermEval1 = special.eval_hermite(nz,xi)
        poly1 = (-1+xi**2) * dXiDz**2
        hermEval1 *= poly1
        
        if nz-1 < 0:
            hermEval2 = np.zeros(xi.shape)
        else:
            hermEval2 = special.eval_hermite(nz-1,xi)
        poly2 = -4*nz * xi *dXiDz**2
        hermEval2 *= poly2
        
        if nz-2 < 0:
            hermEval3 = np.zeros(xi.shape)
        else:
            hermEval3 = special.eval_hermite(nz-2,xi)
        poly3 = 4*(nz-1)*nz * dXiDz**2
        hermEval3 *= poly3
        
        coeff = np.exp(-xi**2/2) * np.sqrt(self.betaz) * N
        return coeff*(hermEval1 + hermEval2 + hermEval3)
    
class HFBMatrix(CylindricalIntegral):
    #TODO: include variations that UNEDF1 doesn't use, such as w.r.t. $\nabla_r \rho_q$
    def __init__(self,basis,*args):
        super().__init__(*args)
        
        self.basis = basis
        
        self.quantNums = self.basis.quantNumbersByBlock
        self.arrInds = []
        
        #Many matrix elements are diagonal in spin space (e.g. var $\rho$).
        #Precomputing those indices appears to be helpful
        for k in range(self.basis.nBlocks):
            q = self.basis.quantNumbersByBlock[k]
            mlNums = np.unique(q['m_l'])
            
            blockArrInds = []
            for ml in mlNums:
                idx = q[q['m_l']==ml].index.to_numpy()
                blockArrInds.append(idx - q.index[0])
            self.arrInds.append(blockArrInds)
        
    def rho(self,varRho,bz,bp):
        blockMats = []
        
        if varRho.ndim == 2:
            nEls = 1
        else:
            nEls = varRho.shape[0]
        
        for k in range(self.basis.nBlocks):
            shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
            mat = np.zeros((nEls,)+shp)
            
            #Reduces runtime by about 0.1 s
            # psiVarRho = self.basis.psi[k]*varRho
            # for mlInds in self.arrInds[k]:
            #     for (i1Iter,i1) in enumerate(mlInds):
            #         for i2Iter in range(i1Iter+1):
            #             i2 = mlInds[i2Iter]
                        
            #             toIntegrate = psiVarRho[i1]*self.basis.psi[k][i2]
            #             mat[:,i1,i2] = self.integrate(toIntegrate,bz,bp)/(2*np.pi)
            for mlInds in self.arrInds[k]:
                for (i1Iter,i1) in enumerate(mlInds):
                    for i2Iter in range(i1Iter+1):
                        i2 = mlInds[i2Iter]
                        
                        toIntegrate = self.basis.psi[k][i1]*varRho*self.basis.psi[k][i2]
                        mat[:,i1,i2] = self.integrate(toIntegrate,bz,bp)/(2*np.pi)
                    
            mat = utils.symmetrize_array(mat)
            if nEls == 1:
                blockMats.append(mat[0])
            else:
                blockMats.append(mat)
            
        return blockMats
    
    def rho_tilde(self,varRhoTilde,bz,bp):
        return self.rho(varRhoTilde,bz,bp)
    
    # # @profile
    # #The profiler thinks this version is faster, but actually running it,
    # #it turns out to be slower
    # def rho(self,varRho,bz,bp):
    #     blockMats = []
        
    #     for k in range(self.basis.nBlocks):
    #         shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
    #         mat = np.zeros(shp)
            
    #         for mlInds in self.arrInds[k]:
    #             for (i1Iter,i1) in enumerate(mlInds):
    #                 toIntegrate = self.basis.psi[k][i1] * self.basis.psi[k][mlInds] * varRho
    #                 mat[i1,mlInds] = self.integrate(toIntegrate,bz,bp)/(2*np.pi)
                    
    #         blockMats.append(symmetrize_array(mat))
            
    #     return blockMats
    
    def del_rho(self,varLaplRho,bz,bp):
        blockMats = []
        
        r = bp*np.sqrt(self.eta)[:,None]
        
        if varLaplRho.ndim == 2:
            nEls = 1
        else:
            nEls = varLaplRho.shape[0]
        
        for k in range(self.basis.nBlocks):
            shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
            mat = np.zeros((nEls,)+shp)
            
            psi = self.basis.psi[k]
            dpsi = self.basis.dpsi[k]
            d2psi = self.basis.d2psi[k]
            
            #Saves some fraction of the runtime
            dpsiOverR = dpsi[0]/r
            
            #Reduces runtime by about 0.1 s
            # psiVarLaplRho = psi*varLaplRho
            # dpsiVarLaplRho = dpsi*varLaplRho
            # for mlInds in self.arrInds[k]:
            #     for (i1Iter,i1) in enumerate(mlInds):
            #         for i2Iter in range(i1Iter+1):
            #             i2 = mlInds[i2Iter]
                        
            #             #The radial derivative
            #             term1 = psiVarLaplRho[i1]*d2psi[0,i2] 
            #             term1 += 2*dpsiVarLaplRho[0,i1]*dpsi[0,i2] 
            #             term1 += d2psi[0,i1]*psiVarLaplRho[i2]
            #             term1 += psiVarLaplRho[i1]*dpsiOverR[i2] + dpsiOverR[i1]*psiVarLaplRho[i2]
                        
            #             #The z derivative
            #             term2 = psiVarLaplRho[i1]*d2psi[5,i2] + 2*dpsiVarLaplRho[2,i1]*dpsi[2,i2] + d2psi[5,i1]*psiVarLaplRho[i2]
                        
            #             mat[:,i1,i2] = self.integrate(term1+term2,bz,bp)/(2*np.pi)
            for mlInds in self.arrInds[k]:
                for (i1Iter,i1) in enumerate(mlInds):
                    for i2Iter in range(i1Iter+1):
                        i2 = mlInds[i2Iter]
                        
                        #The radial derivative
                        term1 = psi[i1]*d2psi[0,i2] 
                        term1 += 2*dpsi[0,i1]*dpsi[0,i2] 
                        term1 += d2psi[0,i1]*psi[i2]
                        term1 += psi[i1]*dpsiOverR[i2] + dpsiOverR[i1]*psi[i2]
                        
                        #The z derivative
                        term2 = psi[i1]*d2psi[5,i2] + 2*dpsi[2,i1]*dpsi[2,i2] + d2psi[5,i1]*psi[i2]
                        
                        mat[:,i1,i2] = self.integrate(varLaplRho*(term1+term2),bz,bp)/(2*np.pi)
            
            mat = utils.symmetrize_array(mat)
            if nEls == 1:
                blockMats.append(mat[0])
            else:
                blockMats.append(mat)
            
        return blockMats
    
    # This is actually even slower than what we already have
    # def lapl_rho(self,varLaplRho,bz,bp):
    #     blockMats = []
        
    #     r = bp*np.sqrt(self.eta)
        
    #     for k in range(self.basis.nBlocks):
    #         shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
    #         mat = np.zeros(shp)
            
    #         psi = self.basis.psi[k]
    #         dpsi = self.basis.dpsi[k]
    #         d2psi = self.basis.d2psi[k]
            
    #         for mlInds in self.arrInds[k]:
    #             #The r derivative
    #             term1 = np.expand_dims(psi[mlInds],0) * np.expand_dims(d2psi[0,mlInds],1) \
    #                 + 2*np.expand_dims(dpsi[0,mlInds],0)*np.expand_dims(dpsi[0,mlInds],1)\
    #                     + np.expand_dims(psi[mlInds],1) * np.expand_dims(d2psi[0,mlInds],0)
    #             term1 += (np.expand_dims(psi[mlInds],0)*np.expand_dims(dpsi[0,mlInds],1) \
    #                 + np.expand_dims(psi[mlInds],1)*np.expand_dims(dpsi[0,mlInds],0))/r[:,None]
                
    #             #The z derivative
    #             term2 = np.expand_dims(psi[mlInds],0)*np.expand_dims(d2psi[5,mlInds],1)\
    #                 + 2*np.expand_dims(dpsi[2,mlInds],0)*np.expand_dims(dpsi[2,mlInds],1)\
    #                     + np.expand_dims(d2psi[5,mlInds],0)*np.expand_dims(psi[mlInds],1)
                
    #             mat[mlInds[:,None],mlInds[None,:]] = self.integrate(varLaplRho*(term1+term2),bz,bp)/(2*np.pi)
                
    #             # del term1
    #             # del term2
    #         blockMats.append(symmetrize_array(mat))
    #     # sys.exit()
    #     del term1
    #     del term2
            
    #     return blockMats
    
    
    # ## The method below is actually half as fast as the normally-used version
    # def lapl_rho(self,varLaplRho,bz,bp):
    #     blockMats = []
        
    #     r = bp*np.sqrt(self.eta)
        
    #     for k in range(self.basis.nBlocks):
    #         shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
    #         mat = np.zeros(shp)
            
    #         psi = self.basis.psi[k]
    #         dpsi = self.basis.dpsi[k]
    #         d2psi = self.basis.d2psi[k]
            
    #         for mlInds in self.arrInds[k]:
    #             # print(mlInds)
    #             for (i1Iter,i1) in enumerate(mlInds):
    #                 # if i1Iter == 0:
    #                 #     arrToIntegrate = np.zeros((len(mlInds),) + psi[0].shape)
    #                 # else:
    #                 #     arrToIntegrate[:,:,:] = 0
                        
    #                 # print(arrToIntegrate.shape)
    #                 # print(i1Iter)
    #                 # print(i1)
    #                 # print(i2Inds)
    #                 # for i2Iter in range(i1Iter+1):
    #                 #     i2 = mlInds[i2Iter]
    #                 #The radial derivative
    #                 # arrToIntegrate += psi[i1]*d2psi[0,mlInds] + 2*dpsi[0,i1]*dpsi[0,mlInds] + d2psi[0,i1]*psi[mlInds]
    #                 # arrToIntegrate += (psi[i1]*dpsi[0,mlInds] + dpsi[0,i1]*psi[mlInds])/r[:,None]
    #                 term1 = psi[i1]*d2psi[0,mlInds] + 2*dpsi[0,i1]*dpsi[0,mlInds] + d2psi[0,i1]*psi[mlInds]
    #                 # term1 += (psi[i1]*dpsi[0,mlInds] + dpsi[0,i1]*psi[mlInds])/r[:,None]
    #                 # term1 = d2psi[0,mlInds]*psi[i1] + 2*dpsi[0,mlInds]*dpsi[0,i1] + psi[mlInds]*d2psi[0,i1]
    #                 term1 += (psi[i1]*dpsi[0,mlInds] + dpsi[0,i1]*psi[mlInds])/r[:,None]
    #                 # print(term1.shape)
                    
                    
    #                 #The z derivative
    #                 # arrToIntegrate += psi[i1]*d2psi[5,mlInds] + 2*dpsi[2,i1]*dpsi[2,mlInds] + d2psi[5,i1]*psi[mlInds]
    #                 term2 = psi[i1]*d2psi[5,mlInds] + 2*dpsi[2,i1]*dpsi[2,mlInds] + d2psi[5,i1]*psi[mlInds]
                    
    #                 # mat[i1,mlInds] = self.broadcast_integrate(varLaplRho*(arrToIntegrate),bz,bp)/(2*np.pi)
    #                 mat[i1,mlInds] = self.integrate(varLaplRho*(term1+term2),bz,bp)/(2*np.pi)
    #             # sys.exit()
    #         # sys.exit()
    #         blockMats.append(symmetrize_array(mat))
    #     # sys.exit()
            
    #     return blockMats
    
    def tau(self,varTau,bz,bp):
        blockMats = []
        
        r = bp*np.sqrt(self.eta)
        
        if varTau.ndim == 2:
            nEls = 1
        else:
            nEls = varTau.shape[0]
        
        for k in range(self.basis.nBlocks):
            shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
            mat = np.zeros((nEls,)+shp)
            
            psi = self.basis.psi[k]
            dpsi = self.basis.dpsi[k]
            
            lmdArr = self.quantNums[k]['m_l'].to_numpy()
            
            for (mlIter,mlInds) in enumerate(self.arrInds[k]):
                for (i1Iter,i1) in enumerate(mlInds):
                    for i2Iter in range(i1Iter+1):
                        i2 = mlInds[i2Iter]
                        
                        #The radial derivative
                        term1 = dpsi[0,i1]*dpsi[0,i2]
                        
                        #The z derivative
                        term2 = dpsi[2,i1]*dpsi[2,i2]
                        
                        #The $\phi$ derivative
                        lambd = lmdArr[i1]
                        term3 = lambd**2/r[:,None]**2 * psi[i1]*psi[i2]
                        
                        mat[:,i1,i2] = self.integrate(varTau*(term1+term2+term3),bz,bp)/(2*np.pi)
            mat = utils.symmetrize_array(mat)
            if nEls == 1:
                blockMats.append(mat[0])
            else:
                blockMats.append(mat)
            
        return blockMats
    
    def divJ(self,varDivJ,bz,bp):
        blockMats = []
        
        if varDivJ.ndim == 2:
            nEls = 1
        else:
            nEls = varDivJ.shape[0]
        
        r = bp*np.sqrt(self.eta)
        
        for k in range(self.basis.nBlocks):
            Mr, Mphi, Mz = self.basis.M[k]
            
            dr, dphiIn, dz = self.basis.dpsi[k]
            #This copy statement is necessary, else results change call-to-call.
            #It doesn't have any impact at all on the code execution time
            dphi = dphiIn.copy()/r[None,:,None]
            
            mat = np.zeros((nEls,)+Mr.shape)
            for i1 in range(Mr.shape[0]):
                for i2 in range(i1+1):                    
                    term1 = (-dphi[i1])*dz[i2] - dz[i1]*dphi[i2]
                    term2 = dz[i1]*dr[i2] - dr[i1]*dz[i2]
                    term3 = dr[i1]*dphi[i2] - (-dphi[i1])*dr[i2]
                                            
                    arr = Mr[i1,i2]*term1 + Mphi[i1,i2] * term2 + Mz[i1,i2]*term3
                    
                    mat[:,i1,i2] = self.integrate(varDivJ*arr,bz,bp)/(2*np.pi)
            mat = utils.symmetrize_array(mat)
            if nEls == 1:
                blockMats.append(mat[0])
            else:
                blockMats.append(mat)
            
        return blockMats
    
    """
    Leaving the following snippet as an example. JAX natively uses multiple
    CPU cores, and I don't think there's a way to restrict to one core. So,
    while this does save time (about 5 s of the 11 s runtime), it uses multiple
    CPU cores, so the comparison is faulty
    """
    # def divJ(self,varDivJ,bz,bp):
    #     blockMats = []
        
    #     r = bp*np.sqrt(self.eta)
        
    #     for k in range(self.basis.nBlocks):
    #         Mr, Mphi, Mz = self.basis.M[k]
            
    #         dr, dphi, dz = self.basis.dpsi[k]
    #         dphi /= r[None,:,None]
            
    #         @jit
    #         def _func_to_vmap(i1,i2):
    #             term1 = (-dphi[i1])*dz[i2] - dz[i1]*dphi[i2]
    #             term2 = dz[i1]*dr[i2] - dr[i1]*dz[i2]
    #             term3 = dr[i1]*dphi[i2] - (-dphi[i1])*dr[i2]
                                        
    #             arr = Mr[i1,i2]*term1 + Mphi[i1,i2] * term2 + Mz[i1,i2]*term3
                
    #             return self.integrate(varDivJ*arr,bz,bp)/(2*jnp.pi)
    #         vmapped = jax.vmap(jax.vmap(_func_to_vmap,in_axes=(None,0)),in_axes=(0,None))
            
    #         i1 = jnp.arange(Mr.shape[0])
    #         i2 = jnp.arange(Mr.shape[0])
    #         mat = vmapped(i1,i2)
            
    #         blockMats.append(mat)
            
    #     return blockMats
    
    def make_hfb_matrix(self,varRho,varRhoTilde,bz,bp,chemPot):
        hfbArr = []
        for k in range(self.basis.nBlocks):
            hfbArr.append(np.zeros((2*self.basis.psi[k].shape[0],2*self.basis.psi[k].shape[0])))

        for (key,arr) in varRho.items():
            blocks = getattr(self,key)(arr,bz,bp)
            for k in range(self.basis.nBlocks):
                l = self.basis.psi[k].shape[0]
                hfbArr[k][:l,:l] += blocks[k]
                hfbArr[k][l:,l:] -= blocks[k]
                
        for (key,arr) in varRhoTilde.items():
            blocks = getattr(self,key)(arr,bz,bp)
            for k in range(self.basis.nBlocks):
                l = self.basis.psi[k].shape[0]
                hfbArr[k][l:,:l] += blocks[k]
                hfbArr[k][:l,l:] += blocks[k].T
                
        #Chemical potential
        for k in range(self.basis.nBlocks):
            l = self.basis.psi[k].shape[0]
            hfbArr[k][:l,:l] -= chemPot*np.identity(l)
            hfbArr[k][l:,l:] += chemPot*np.identity(l)
        
        return hfbArr
    
def diagonalize_hfb_matrix(hfbArr):
    """
    Returns only U, V, and Eqp for positive-energy states
    """
    nBlocks = len(hfbArr)
    U = []
    V = []
    eqp = []
    
    for k in range(nBlocks):
        t0 = time.time()
        halfDim = hfbArr[k].shape[0]//2
        
        #About half the speed of np.linalg.eigh
        # vals, vecs = linalg.eigh(hfbArr[k],subset_by_index=[halfDim,hfbArr[k].shape[0]-1],
        #                          check_finite=False)
        # eqp.append(vals)
        # U.append(vecs[:halfDim])
        # V.append(vecs[halfDim:])
        
        # t0 = time.time()
        # vals, vecs = np.linalg.eigh(hfbArr[k].astype('float32'))
        vals, vecs = np.linalg.eigh(hfbArr[k])
        # print(50*'=')
        # print(vals[halfDim:][:10])
        
        # #Positive-energy only
        eqp.append(vals[halfDim:])
        # #For indexing, vecs[:,0] = (U[0],V[0]). Positive-energy states are the
        # #upper half of the index
        U.append(vecs[:halfDim,halfDim:])
        V.append(vecs[halfDim:,halfDim:])
        # t1 = time.time()
        # print('Previous time: %.3e s'%(t1-t0))
        
        
        # t0 = time.time()
        # h = hfbArr[k][:halfDim,:halfDim]
        # hTilde = hfbArr[k][:halfDim,halfDim:]
        
        # M = (h + 1j*hTilde) @ (h - 1j*hTilde)
        # vals, vecs = np.linalg.eigh(M)
        
        # print(np.sqrt(vals[:10]))
        # t2 = time.time()
        # print('eigenval time: %.3e s'%(t2-t0))
        
        
        # Q1 = np.real(vecs)
        # Q2 = np.imag(vecs)
        
        # # hQ1 = h @ Q1
        # # hQ2 = h @ Q2
        # # htQ1 = hTilde @ Q1
        # # htQ2 = hTilde @ Q2
        # hQ = h @ vecs
        # hQ1 = np.real(hQ)
        # hQ2 = np.imag(hQ)
        
        # htQ = hTilde @ vecs
        # htQ1 = np.real(htQ)
        # htQ2 = np.imag(htQ)
        
        # e = np.einsum('ij,ij->i',Q1,hQ1)
        # # e = np.array([Q1[:,i] @ hQ1[:,i] for i in range(Q1.shape[0])])
        # # e = np.diag(Q1.T @ hQ1 - Q2.T @ hQ2 + Q1.T @ htQ2 + Q2.T @ htQ1)
        # t1 = time.time()
        # print('evec time: %.3e s'%(t1-t0))
        
        
        
        
        # sys.exit()
        t1 = time.time()
        # print('Block %d diag time: %.3e s'%(k,t1-t0))
    return U, V, eqp

# def diagonalize_hfb_matrix_v2(hfbArr):
#     """
#     Returns only U, V, and Eqp for positive-energy states
#     """
#     nBlocks = len(hfbArr['h'])
#     U = []
#     V = []
#     eqp = []
    
#     for k in range(nBlocks):
#         # t0 = time.time()
#         # vals, vecs = np.linalg.eigh(hfbArr[k])
#         # print(50*'=')
#         # print(vals[halfDim:][:10])
#         # t1 = time.time()
#         # print('Previous time: %.3e s'%(t1-t0))
#         # #Positive-energy only
#         # eqp.append(vals[halfDim:])
#         # #For indexing, vecs[:,0] = (U[0],V[0]). Positive-energy states are the
#         # #upper half of the index
#         # U.append(vecs[:halfDim,halfDim:])
#         # V.append(vecs[halfDim:,halfDim:])
        
#         # t0 = time.time()
#         h = hfbArr['h'][k]
#         hTilde = hfbArr['hTilde'][k]
#         # h = hfbArr[k][:halfDim,:halfDim]
#         # # hTilde = hfbArr[k][halfDim:,:halfDim]
#         # hTilde = hfbArr[k][:halfDim,halfDim:]
        
#         M = (h + 1j*hTilde) @ (h - 1j*hTilde)
#         vals, vecs = np.linalg.eigh(M)
        
#         eqp.append(np.sqrt(vals))
#         U.append(np.real(vecs))
#         V.append(np.imag(vecs))
#         # print(vecs.shape)
#         # print(vecs.dtype)
#         # t1 = time.time()
#         # print(np.sqrt(vals[:10]))
#         # print('New time: %.3e s'%(t1-t0))
        
        
#         # sys.exit()
#     return U, V, eqp

class PairingRegularization:
    """
    Organized as a class for readability reasons - the methods in this class
    depend on each other, but do different things
    """
    def __init__(self,basis,ebarMax=60,cutoffTol=10.**(-6)):
        self.basis = basis
        self.ebarMax = ebarMax
        self.cutoffTol = cutoffTol
        self.asDict = [{key:df[key].to_numpy().copy() for key in df.columns} for df in self.basis.quantNumbersByBlock]

    @np.errstate(under='ignore')
    #Some warning occurs in np.sqrt, but it doesn't always happen, and
    #the result is still correct
    # @profile
    def get_active_states(self,Vin,eqpIn,chemPot):
        #About 4x faster than Pandas operations - from 0.03 s to 0.007 s.
        #Profiler makes it look worse than it actually is, but it's still not
        #great
        
        ret = []
        
        for k in range(self.basis.nBlocks):
            self.asDict[k]['eqp'] = eqpIn[k]
            V = Vin[k]
            # asDict[k]['occ'] = np.einsum('ij,ij->j',V,V).clip(0,1)
            self.asDict[k]['occ'] = np.sum(V**2,axis=0).clip(0,1)
            
            self.asDict[k]['ebar'] = (1-2*self.asDict[k]['occ'])*self.asDict[k]['eqp'] + chemPot
            self.asDict[k]['del'] = 2*self.asDict[k]['eqp']*np.sqrt(self.asDict[k]['occ']*(1-self.asDict[k]['occ']))
            
            self.asDict[k]['isActive'] = (self.asDict[k]['ebar'] <= self.ebarMax)
            
            #Basically, checking if we're close to the energy cutoff (I don't understand
            #where this comes from, though)
            
            #Want to truncate the exponent, and we do so using sys.float_info.max
            #https://stackoverflow.com/a/3477332 which is equivalent to Fortran's HUGE
            maxVal = sys.float_info.max
            
            exponent = maxVal * np.ones(len(self.asDict[k]['eqp']))
            borderInds = np.where(100*np.abs(self.asDict[k]['ebar'] - self.ebarMax) < np.log(maxVal))[0]
            exponent[borderInds] = np.exp(100*(self.asDict[k]['ebar'][borderInds] - self.ebarMax))
            self.asDict[k]['exponent'] = exponent
            goodInds = (1/(1+self.asDict[k]['exponent']) > self.cutoffTol)
            self.asDict[k]['isActive'][goodInds] = True
            
            ret.append(self.asDict[k])
            
        return ret
    
    # @profile
    def adjust_fermi_energy(self,activeStates,N,chemPot):
        #This version is 30-40 times faster than the Pandas version, dropping
        #the runtime for the pairing regularization from 0.16 s to 0.02 s
        ebarList = [activeStates[k]['ebar'][activeStates[k]['isActive']] for k in range(self.basis.nBlocks)]
        delList = [activeStates[k]['del'][activeStates[k]['isActive']] for k in range(self.basis.nBlocks)]
        
        def bcs_occ(lambd):
            occ = 0
            for k in range(self.basis.nBlocks):
                diff = ebarList[k] - lambd
                bcsOcc = 0.5*(1 - diff/np.sqrt(diff**2 + delList[k]**2))
                
                occ += bcsOcc.sum()
            
            return 2*occ - N
        
        def bcs_occ_derivative(lambd):
            ret = 0
            for k in range(self.basis.nBlocks):
                ek = np.sqrt((ebarList[k]-lambd)**2 + delList[k]**2)
                d = delList[k]**2/(2*ek**3)
                
                ret += d.sum()
            return 2*ret
        
        sol = optimize.root_scalar(bcs_occ,x0=chemPot,fprime=bcs_occ_derivative,method='newton')
        if sol.converged:
            ret = sol.root
        else:
            ret = chemPot

        return ret
    
class PSDerivatives:
    def __init__(self,nEta=40,nXi=80):
        self.nEta = nEta
        self.nXi = nXi

        self.eta, self.wr = special.roots_laguerre(nEta)
        self.xi, self.wz = special.roots_hermite(nXi)

        self.Deta = self.make_D_eta()
        self.Dxi = self.make_D_xi()

    def psi(self,xi,nz):
        N = np.sqrt(1/(np.sqrt(np.pi) * 2.**nz * math.factorial(nz)))
            
        ret = np.exp(-xi**2/2) * special.eval_hermite(nz,xi)
            
        return N * ret

    def phi(self,eta,nr):
        ret = np.exp(-eta/2)
        ret = ret*special.eval_genlaguerre(nr,0,eta)
        return ret
    
    def make_D_eta(self):
        M = np.zeros((self.nEta,self.nEta))
        for l in range(self.nEta):
            M[l,l] = -0.5
            for n in range(self.nEta):
                if l <= n-1:
                    M[l,n] -= 1

        #Indexed as phiEvals[j,n]
        phiEvals = self.phi(self.eta[:,None],np.arange(self.nEta,dtype=int)[None,:])
        weights = self.wr*np.exp(self.eta/2)
        #Indexed as lagEval[m,i]
        lagEval = special.eval_genlaguerre(np.arange(self.nEta,dtype=int)[:,None],0,self.eta[None,:])

        D = weights*(phiEvals @ M @ lagEval)
        return D

    def make_D_xi(self):
        #psiEvals is indexed as psiEvals[j,n], i.e. the row (first)
        #index is the evaluation location, and the column (second)
        #index is the order of the polynomial
        psiEvals = np.zeros((self.nXi,self.nXi+1))
        psiEvals[:,:-1] = np.array([self.psi(self.xi,n) for n in range(self.nXi)]).T

        C = np.sum(psiEvals**2,axis=1)

        M = np.zeros((self.nXi,self.nXi))

        for n in range(self.nXi):
            M[n] = 1/np.sqrt(2)*(-np.sqrt(n)*psiEvals[:,n-1]\
                                +np.sqrt(n+1)*psiEvals[:,n+1])
            M[n] /= C

        D = (M.T @ psiEvals[:,:-1].T).T
        return D

    def laplacian(self,arr,bz,bp):
        ret = self.Deta @ arr + self.eta[:,None] * (self.Deta @ self.Deta @ arr)
        ret = 4/bp**2 * ret

        ret += 1/bz**2 * (self.Dxi @ self.Dxi @ arr.T).T
        return ret
    
class Reconstruction:
    def __init__(self,basis,rGrid,laplaceMode='exact',laplaceOpts={}):
        self.basis = basis
        self.rGrid = rGrid
        
        self.Vgrid = None
        self.dVgrid = 3*[None,]
        self.d2Vgrid = 6*[None,]
        self.Ugrid = None

        assert laplaceMode in ['exact','pseudospectral']
        self.laplaceMode = laplaceMode
        
        self.activeList, self.upList, self.downList = None, None, None

        self.densities = {}
        if laplaceMode == 'pseudospectral':
            self.derivativeObj = laplaceOpts['derivativeObj']
                
    def compute_V(self,activeStates,U,V,coordInds=None):
        self._get_spin_ud_inds(activeStates)
        
        if self.Vgrid is None:
            self.Vgrid = {ud:[] for ud in ['up','down']}
            
            for k in range(len(self.upList)):
                activeInds = self.activeList[k]
                udInds = {'up':self.upList[k],'down':self.downList[k]}
                
                #Column Varr[:,i] corresponds to eigenvalue[i]. Since some states aren't active,
                #we want Varr[:,activeInds]. But, for the spin up/down component, we only want
                #Varr[spinUpInds,:]. So, we index as Varr[spin,activeInds], with appropriate
                #broadcasting b/c numpy
                
                for ud in ['up','down']:
                    idx = udInds[ud][:,None],activeInds[None,:]
                    if coordInds is None:
                        wfInds = (udInds[ud],)
                    else:
                        wfInds = (udInds[ud][:,None,None], *coordInds)
                    
                    # Vgrid[ud].append(np.einsum('ij,ikl->jkl',V[k][idx],
                    #                     self.basis.psi[k][udInds[ud]]))
                    #Faster than equivalent np.einsum call
                    # a = V[k][idx].T
                    # b = np.swapaxes(self.basis.psi[k][*wfInds],0,1)
                    # c = a @ b
                    # d = np.swapaxes(c,0,1)
                    # Vgrid[ud].append(d)
                    # print(V[k][idx].shape)
                    # print(self.basis.psi[k][*wfInds].shape)
                    self.Vgrid[ud].append(
                        np.swapaxes(
                            V[k][idx].T @ np.swapaxes(self.basis.psi[k][*wfInds],0,1),0,1))
        
        return
    
    def compute_dV(self,activeStates,U,V,idxToCompute=[0,1,2],coordInds=None):
        self._get_spin_ud_inds(activeStates)
        
        for derivativeIdx in idxToCompute:
            if self.dVgrid[derivativeIdx] is None:
                self.dVgrid[derivativeIdx] = {ud:[] for ud in ['up','down']}
                
                for k in range(len(self.upList)):
                    activeInds = self.activeList[k]
                    udInds = {'up':self.upList[k],'down':self.downList[k]}
                    
                    #Column Varr[:,i] corresponds to eigenvalue[i]. Since some states aren't active,
                    #we want Varr[:,activeInds]. But, for the spin up/down component, we only want
                    #Varr[spinUpInds,:]. So, we index as Varr[spin,activeInds], with appropriate
                    #broadcasting b/c numpy
                    
                    for ud in ['up','down']:
                        idx = udInds[ud][:,None],activeInds[None,:]
                        if coordInds is None:
                            wfInds = (udInds[ud],)
                        else:
                            wfInds = (udInds[ud][:,None,None], *coordInds)
                        
                        self.dVgrid[derivativeIdx][ud].append(
                            np.swapaxes(V[k][idx].T @ np.swapaxes(
                                self.basis.dpsi[k][derivativeIdx][*wfInds],0,1),0,1))
        return
    
    # @profile
    def compute_d2V(self,activeStates,U,V,idxToCompute=[0,5],coordInds=None):
        self._get_spin_ud_inds(activeStates)
        
        for derivativeIdx in idxToCompute:
            if self.d2Vgrid[derivativeIdx] is None:
                self.d2Vgrid[derivativeIdx] = {ud:[] for ud in ['up','down']}
                
                for k in range(len(self.upList)):
                    activeInds = self.activeList[k]
                    udInds = {'up':self.upList[k],'down':self.downList[k]}
                    
                    #Column Varr[:,i] corresponds to eigenvalue[i]. Since some states aren't active,
                    #we want Varr[:,activeInds]. But, for the spin up/down component, we only want
                    #Varr[spinUpInds,:]. So, we index as Varr[spin,activeInds], with appropriate
                    #broadcasting b/c numpy
                    
                    for ud in ['up','down']:
                        idx = udInds[ud][:,None],activeInds[None,:]
                        if coordInds is None:
                            wfInds = (udInds[ud],)
                        else:
                            wfInds = (udInds[ud][:,None,None], *coordInds)
                        
                        self.d2Vgrid[derivativeIdx][ud].append(
                            np.swapaxes(V[k][idx].T @ np.swapaxes(
                                self.basis.d2psi[k][derivativeIdx][*wfInds],0,1),0,1))
        return
    
    def compute_U(self,activeStates,U,V,coordInds=None):
        self._get_spin_ud_inds(activeStates)
        
        if self.Ugrid is None:
            self.Ugrid = {ud:[] for ud in ['up','down']}
            
            for k in range(len(self.upList)):
                activeInds = self.activeList[k]
                udInds = {'up':self.upList[k],'down':self.downList[k]}
                
                #Column Varr[:,i] corresponds to eigenvalue[i]. Since some states aren't active,
                #we want Varr[:,activeInds]. But, for the spin up/down component, we only want
                #Varr[spinUpInds,:]. So, we index as Varr[spin,activeInds], with appropriate
                #broadcasting b/c numpy
                
                for ud in ['up','down']:
                    idx = udInds[ud][:,None],activeInds[None,:]
                    if coordInds is None:
                        wfInds = (udInds[ud],)
                    else:
                        wfInds = (udInds[ud][:,None,None], *coordInds)
                    
                    self.Ugrid[ud].append(
                        np.swapaxes(
                            U[k][idx].T @ np.swapaxes(self.basis.psi[k][*wfInds],0,1),0,1))
        
        return
        
    def _get_spin_ud_inds(self,activeStates):
        if self.activeList is None:
            #TODO: maybe belongs elsewhere, such as in HOBasis?
            self.upList = []
            self.downList = []
            self.activeList = []
            for k in range(self.basis.nBlocks):
                activeInds = np.arange(len(activeStates[k]))
                activeInds = np.where(activeStates[k]['isActive'])[0]
                if activeInds.size == 0:
                    continue
                
                self.activeList.append(activeInds)
                
                # subDf = activeStates[k]
                subDf = self.basis.quantNumbersByBlock[k]
                self.upList.append(np.where(subDf['2m_s']==1)[0])
                self.downList.append(np.where(subDf['2m_s']==-1)[0])
        return 
    # @timer
    def rho(self,activeStates,U,V,coordInds=None):
        self.compute_V(activeStates,U,V,coordInds=coordInds)
        
        shp = self.Vgrid['up'][0].shape[1:]
                
        rho = np.zeros(shp)
        
        for (ud,Varr) in self.Vgrid.items():
            for V in Varr:
                rho += np.sum(V**2,axis=0)
            
        #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
        #integral
        rho = 2 * rho / (2*np.pi)
        
        self.densities['rho'] = rho
        return rho
    
    def dr_rho(self,activeStates,U,V,coordInds=None):
        self.compute_V(activeStates,U,V,coordInds=coordInds)
        self.compute_dV(activeStates,U,V,idxToCompute=[0,],coordInds=coordInds)

        shp = self.Vgrid['up'][0].shape[1:]
            
        dr_rho = np.zeros(shp)
        
        for ud in ['up','down']:
            for k in range(len(self.Vgrid[ud])):
                dr_rho += 2*np.sum(self.Vgrid[ud][k]*self.dVgrid[0][ud][k],
                                   axis=0)
        
        #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
        #integral
        dr_rho = 2 * dr_rho / (2*np.pi)

        return dr_rho
    
    def dz_rho(self,activeStates,U,V,coordInds=None):
        self.compute_V(activeStates,U,V,coordInds=coordInds)
        self.compute_dV(activeStates,U,V,idxToCompute=[2,],coordInds=coordInds)

        shp = self.Vgrid['up'][0].shape[1:]
            
        dz_rho = np.zeros(shp)
        
        for ud in ['up','down']:
            for k in range(len(self.Vgrid[ud])):
                dz_rho += 2*np.sum(self.Vgrid[ud][k]*self.dVgrid[2][ud][k],
                                   axis=0)
        
        #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
        #integral
        dz_rho = 2 * dz_rho / (2*np.pi)

        return dz_rho

    # @timer
    def del_rho(self,activeStates,U,V,coordInds=None):
        if self.laplaceMode == 'exact':
            self.compute_V(activeStates,U,V,coordInds=coordInds)
            self.compute_dV(activeStates,U,V,idxToCompute=[0,1,2],coordInds=coordInds)
            self.compute_d2V(activeStates,U,V,idxToCompute=[0,5],coordInds=coordInds)
            
            shp = self.Vgrid['up'][0].shape[1:]
            
            delRho = np.zeros(shp)
            
            for ud in ['up','down']:
                for k in range(len(self.Vgrid[ud])):
                    delRho += 2*np.sum(self.Vgrid[ud][k]*(self.d2Vgrid[0][ud][k] + self.d2Vgrid[5][ud][k]) \
                                    + self.dVgrid[0][ud][k]**2 + self.dVgrid[2][ud][k]**2 \
                                    + self.Vgrid[ud][k]*self.dVgrid[0][ud][k]/self.rGrid,
                                    axis=0)
            
            #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
            #integral
            delRho = 2 * delRho / (2*np.pi)
        elif self.laplaceMode == 'pseudospectral':
            if coordInds is not None:
                raise NotImplementedError
            delRho = self.derivativeObj.laplacian(self.densities['rho'],self.basis.bz,self.basis.bp)

        return delRho
    # @timer
    def tau(self,activeStates,U,V,coordInds=None):
        self.compute_V(activeStates,U,V,coordInds=coordInds)
        self.compute_dV(activeStates,U,V,coordInds=coordInds)
        
        shp = self.Vgrid['up'][0].shape[1:]
        
        tau = np.zeros(shp)
        
        for ud in ['up','down']:
            for k in range(len(self.Vgrid[ud])):
                tau += np.sum(self.dVgrid[0][ud][k]**2 \
                              + (self.dVgrid[1][ud][k]/self.rGrid)**2 \
                              + self.dVgrid[2][ud][k]**2,
                              axis=0)
            
        #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
        #integral
        tau = 2 * tau / (2*np.pi)
        
        return tau
    # @timer
    def divJ(self,activeStates,U,V,coordInds=None):
        self.compute_dV(activeStates,U,V,coordInds=coordInds)
        
        shp = self.dVgrid[0]['up'][0].shape[1:]
        
        divJ = np.zeros(shp)
        
        for k in range(len(self.Vgrid['up'])):
            dVp = [self.dVgrid[i]['up'][k] for i in range(len(self.dVgrid))]
            dVm = [self.dVgrid[i]['down'][k] for i in range(len(self.dVgrid))]
            
            arrToAdd = (dVm[1]*(dVp[2]+dVm[0]) + dVp[1]*(dVm[2]-dVp[0]))/self.rGrid
            arrToAdd += dVp[2]*dVm[0] - dVm[2]*dVp[0]
            
            divJ += -2*np.sum(arrToAdd,axis=0)
            
        #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
        #integral
        divJ = 2 * divJ / (2*np.pi)
        
        return divJ
    
    # @timer
    def rho_tilde(self,activeStates,U,V,coordInds=None):
        self.compute_V(activeStates,U,V,coordInds=coordInds)
        self.compute_U(activeStates,U,V,coordInds=coordInds)
        
        shp = self.Vgrid['up'][0].shape[1:]
        
        kappa = np.zeros(shp)
        
        for ud in ['up','down']:
            for k in range(len(self.Vgrid[ud])):
                kappa += np.sum(self.Vgrid[ud][k]*self.Ugrid[ud][k],
                                axis=0)
                
        #Multiply by 2 for time-reversed states, divide by 2 pi for $\phi$ coordinate
        #integral
        # rhoT = -2 * rhoT / (2*np.pi)
        kappa = -kappa / (2*np.pi)
        #TODO: I don't know where the factor of 2 from above goes. I have
        #to remove it to agree with HFBTHO, but I think it ought to still be there
            
        return kappa
    
class AuxiliaryFieldConstraint:
    def __init__(self,constraintObjs,basis,basis2=None):
        self.constraintObjs = constraintObjs
        self.basis = basis
        self.nConstrain = len(constraintObjs)
        
        self.moment = None
        
        if basis2 is not None:
            raise NotImplementedError
        
    def get_constraints_in_ph_basis(self,bz,bp):
        if self.moment is None:
            #TODO: shouldn't be too hard to make different for different proton/neutron
            #bases
            arrInds = []
            
            #Many matrix elements are diagonal in spin space (e.g. var $\rho$).
            #Precomputing those indices is helpful
            for k in range(self.basis.nBlocks):
                q = self.basis.quantNumbersByBlock[k]
                mlNums = np.unique(q['m_l'])
                
                blockArrInds = []
                for ml in mlNums:
                    idx = q[q['m_l']==ml].index.to_numpy()
                    blockArrInds.append(idx - q.index[0])
                arrInds.append(blockArrInds)
                
            moment = []
            for k in range(self.basis.nBlocks):
                arr = np.zeros((self.nConstrain,)+2*(len(self.basis.quantNumbersByBlock[k]),))
                
                for idx in arrInds[k]:
                    for (n1Iter,n1) in enumerate(idx):
                        for n2Iter in range(n1Iter):
                            n2 = idx[n2Iter]
                            for (lIter,obj) in enumerate(self.constraintObjs):
                                arr[lIter,n1,n2] = getattr(obj,'matrix_element')(self.basis.psi[k][n1]*self.basis.psi[k][n2],bz,bp)/(2*np.pi)
                                arr[lIter,n2,n1] = arr[lIter,n1,n2]
                moment.append(arr)
            self.moment = moment
        else:
            moment = self.moment
            
        return moment
    
    def get_uv_quasiparticle(self,Uin,Vin,activeStates,hfbthoCompatibility=True):
        if hfbthoCompatibility:
            Uactive = []
            Vactive = []
            
            Uall = Uin[0].flatten(order='F')
            
            for arr in Uin[1:]:
                Uall = np.append(Uall,arr.flatten(order='F'))
                
            Vall = Vin[0].flatten(order='F')
            for arr in Vin[1:]:
                Vall = np.append(Vall,arr.flatten(order='F'))
                
            runningSum = 0
            for df in activeStates:
                nActive = len(df[df['isActive']])
                if nActive == 0:
                    continue
                
                nBasis = len(df)
                UtoAppend = Uall[runningSum:runningSum + nBasis*nActive].reshape((nBasis,nActive),order='F')
                VtoAppend = Vall[runningSum:runningSum + nBasis*nActive].reshape((nBasis,nActive),order='F')
                Uactive.append(UtoAppend)
                Vactive.append(VtoAppend)
                runningSum += nBasis*nActive
        else:
            Uactive = []
            Vactive = []
            
            for k in range(self.basis.nBlocks):
                activeInds = np.where(activeStates[k]['isActive'])[0]
                if len(activeInds) == 0:
                    continue
                
                Uactive.append(Uin[k][:,activeInds])
                Vactive.append(Vin[k][:,activeInds])
                            
        return Uactive, Vactive
    
    def get_qrpa_mat(self,activeStates):
        qrpaMat = []
        for k in range(self.basis.nBlocks):
            activeInds = np.where(activeStates[k]['isActive'])[0]
            if len(activeInds) == 0:
                continue
            eqp = activeStates[k].iloc[activeInds]['eqp'].to_numpy()
            
            qrpaMat.append(1/(eqp + eqp[:,None]))
            
        return qrpaMat
    
class LipkinNogami:
    """
    All told, the two routines get_Geff and get_lmd2 take
    about 0.09 s to run. Could probably be faster,
    but totally negligible compared to existing bottlenecks
    """
    def __init__(self,basis):
        self.basis = basis
    
    # @timer
    def get_Geff(self,activeStates,U,V,hTilde):
        #My Geff
        Epair = 0
        DeltaBar = 0
        trRho = 0

        rhoQP = []
        for k in range(self.basis.nBlocks):
            activeInds = np.where(activeStates[k]['isActive'])[0]
            if len(activeInds) == 0:
                rhoQP.append(np.zeros(2*(len(activeStates[k]['isActive']),)))
                continue
            
            rho = V[k][:,activeInds] @ V[k][:,activeInds].T
            kappa = V[k][:,activeInds] @ U[k][:,activeInds].T

            rhoQP.append(rho)
            
            Epair += -0.5*np.trace(hTilde[k] @ kappa)
            DeltaBar += np.trace(hTilde[k] @ rho)
            trRho += np.trace(rho)
        Epair = 2*Epair #Including time-reversed states
        #I think time-reversed factors of 2 cancel in DeltaBar
        DeltaBar /= trRho

        #Signs float about
        # DeltaBar = -DeltaBar
        return DeltaBar**2/Epair, rhoQP
    
    # @timer
    def get_lmd2(self,Geff,rhoQP):
        # Assumes BCS occupations
        sum1 = 0
        sum2 = 0
        sum3 = 0
        sum4 = 0
        for k in range(len(rhoQP)):
            rho = rhoQP[k]
            # print(np.diag(rho))
            # sys.exit()
            
            v2, _ = np.linalg.eigh(rho)
            # print(v2 - np.diag(rho)[::-1])
            # fig, ax = plt.subplots()
            # ax.plot(v2)
            # ax.plot(np.diag(rho[::-1]))
            # plt.show()
            # sys.exit()
            v2 = v2.clip(0) #Floating point error drops this below 0 sometimes
            v = np.sqrt(v2)
            u = np.sqrt(1-v2)

            sum1 += np.sum(u * v**3)
            sum2 += np.sum(u**3 * v)
            sum3 += np.sum((u*v)**4)
            sum4 += np.sum((u*v)**2)
        return -Geff/4*(sum1*sum2 - sum3)/(sum4**2 - sum3)
        