import numpy as np
from scipy import special
import sys
import math

from jax import jit
import jax
jax.config.update("jax_enable_x64", True)
import jax.numpy as jnp
from functools import partial

import itertools
import pandas as pd
#See https://pandas.pydata.org/pandas-docs/stable/user_guide/indexing.html#returning-a-view-versus-a-copy
pd.options.mode.copy_on_write = True

import time
from scipy import optimize

import matplotlib.pyplot as plt

def symmetrize_array(arr):
    #From https://stackoverflow.com/a/54277518
    return np.tril(arr) + np.triu(arr.T, 1)

def plot_field(field,eta,xi,vmin=None,vmax=None,mesh=False):
    fig, ax = plt.subplots()
    if mesh:
        cf = ax.pcolormesh(np.sqrt(eta),xi,field.T,cmap='Spectral_r',vmin=vmin,vmax=vmax,
                           extend='both')
    else:
        if vmin is None or vmax is None:
            levels = 30
        else:
            levels = np.linspace(vmin,vmax,num=30)
        cf = ax.contourf(np.sqrt(eta),xi,field.T,cmap='Spectral_r',vmin=vmin,vmax=vmax,
                         extend='both',levels=levels)
    plt.colorbar(cf,ax=ax)
    ax.set(xlabel=r'$r/b_\perp$',ylabel=r'$z/b_z$')
    
    return fig, ax

def side_by_side_density(arr1,arr2,eta,xi,titles=None,mesh=False,cmap='Spectral_r'):
    textboxProps = {"boxstyle":'round', "facecolor":'white', "alpha":1,'pad':0.2}
    
    fig, ax = plt.subplots()
    
    rHere = np.sqrt(eta)
    arr = np.vstack([arr1[::-1],arr2])
    rVals = np.hstack([-rHere[::-1],rHere])
    if mesh:
        cf = ax.pcolormesh(rVals,xi,arr.T,cmap=cmap,)
    else:
        cf = ax.contourf(rVals,xi,arr.T,cmap=cmap,
                         extend='both',
                         levels=30
                         )
    plt.colorbar(cf,ax=ax)
    ax.axvline(0,color='black')
    
    ax.set(xlabel=r'$r/b_\perp$',ylabel=r'$z/b_z$')
    
    if titles is not None:
        ax.text(0.02,0.97,titles[0],transform=ax.transAxes,
                verticalalignment='top',horizontalalignment="left",
                bbox=textboxProps,fontsize=8)
        ax.text(0.97,0.97,titles[1],transform=ax.transAxes,
                verticalalignment='top',horizontalalignment="right",
                bbox=textboxProps,fontsize=8)
    return fig, ax

class CylindricalIntegral:
    def __init__(self,xi,eta,wz,wp):
        self.xi = xi
        self.eta = eta
        self.wz = wz
        self.wp = wp
        
    def integrate(self,arr,bz,bp):
        #Compiling this with jit slows down computations considerably, maybe
        #from converting data types
        return np.pi*bz*bp**2 * (self.wp @ arr @ self.wz)

class Normalization(CylindricalIntegral):
    def __call__(self,rho,bz,bp):
        return self.integrate(rho,bz,bp)
    
class AxialMultipoleMoment(CylindricalIntegral):
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
    
    def __call__(self,l,rho,bz,bp):
        coeff = self._coeff(l)
        
        ee, xx = np.meshgrid(self.eta,self.xi)
        
        s = np.sqrt(bp**2*ee + bz**2 * xx**2)
        
        arr = s**l * special.eval_legendre(l,bz*xx/s)
        
        arr = rho*arr.T * coeff * np.sqrt((2*l+1)/(4*np.pi))
        
        return self.integrate(arr,bz,bp)
    
class BaseSkyrme(CylindricalIntegral):
    def __init__(self,*args):
        super().__init__(*args)
        
    def get_energy(self,listOfFields,bz,bp):
        raise NotImplementedError
        
    def var_rho(self,listOfFields,bz,bp,**kwargs):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_dr_rho(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_dz_rho(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_del_rho(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_tau(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_div_J(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_J_fz(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_J_zf(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_J_fr(self,listOfFields,bz,bp):
        shp = listOfFields[0].shape
        return 2*(np.zeros(shp),)
    
    def var_J_rf(self,listOfFields,bz,bp):
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
    
    def var_rho(self,listOfFields,bz,bp,l=None,lagrangeMultiplier=None):
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
                 coeff=2*[20.73553,]):
        """
        The normal kinetic energy density, $\hbar^2/2m \tau$. By default,
        assumes equal proton and neutron mass, with value
        $\hbar^2/2m = 20.73553$ MeV [CITE]
        """
        super().__init__(*args)
        self.coeff = coeff
        
    def get_eneg(self,listOfFields,bz,bp):
        taup, taun = listOfFields
        return self.coeff[0] * self.integrate(taup,bz,bp) + self.coeff[1] * self.integrate(taun,bz,bp)
    
    def var_tau(self,listOfFields,bz,bp):
        taup, taun = listOfFields
        return self.coeff[0]*np.ones(taup.shape), self.coeff[1]*np.ones(taun.shape)
    
class Skyrme_rho_rho(BaseSkyrme):
    def __init__(self,B1,B2,*args):
        super().__init__(*args)
        self.B1 = B1
        self.B2 = B2
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rhoTot = rhop + rhon
        
        return self.B1*self.integrate(rhoTot**2,bz,bp) + self.B2 * \
            self.integrate(rhop**2+rhon**2,bz,bp)
            
    def var_rho(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rhoTot = rhop+rhon
        
        return 2*self.B1*rhoTot + 2*self.B2*rhop, 2*self.B1*rhoTot + 2*self.B2*rhon
            
class Skyrme_rho_tau(BaseSkyrme):
    def __init__(self,B3,B4,*args):
        super().__init__(*args)
        self.B3 = B3
        self.B4 = B4
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,taup,taun = listOfFields
        rho = rhop + rhon
        tau = taup + taun
        
        return self.B3 * self.integrate(rho*tau,bz,bp) + self.B4 * \
            self.integrate(rhop*taup + rhon*taun,bz,bp)
            
    def var_rho(self,listOfFields,bz,bp):
        rhop,rhon,taup,taun = listOfFields
        tau = taup+taun
        
        return self.B3*tau + self.B4*taup, self.B3*tau + self.B4*taun
    
    def var_tau(self,listOfFields,bz,bp):
        rhop,rhon,taup,taun = listOfFields
        
        rho = rhop + rhon
        return self.B3*rho + self.B4*rhop, self.B3*rho + self.B4*rhon
            
class Skyrme_rho_dRho(BaseSkyrme):
    def __init__(self,B5,B6,*args):
        super().__init__(*args)
        self.B5 = B5
        self.B6 = B6
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,drhop,drhon = listOfFields
        rho = rhop + rhon
        drho = drhop + drhon
        
        return self.B5 * self.integrate(rho*drho,bz,bp) + self.B6*\
            self.integrate(rhop*drhop + rhon*drhon,bz,bp)
            
    def var_rho(self,listOfFields,bz,bp):
        rhop,rhon,drhop,drhon = listOfFields
        drho = drhop + drhon
        
        return self.B5*drho + self.B6*drhop, self.B5*drho + self.B6*drhon
    
    def var_del_rho(self,listOfFields,bz,bp):
        rhop,rhon,drhop,drhon = listOfFields
        rho = rhop + rhon
        
        return self.B5*rho + self.B6*rhop, self.B5*rho + self.B6*rhon

class Skyrme_rho_alpha(BaseSkyrme):
    def __init__(self,B7,B8,alpha,*args):
        super().__init__(*args)
        self.B7 = B7
        self.B8 = B8
        self.alpha = alpha
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon = listOfFields
        rho = rhop + rhon
        
        return self.B7 * self.integrate(rho**(2+self.alpha),bz,bp) + self.B8*\
            self.integrate(rho**self.alpha*(rhop**2 + rhon**2),bz,bp)
            
    def var_rho(self,listOfFields,bz,bp):
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
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,divJp,divJn = listOfFields
        return self.B9*self.integrate((rhop+rhon)*(divJp+divJn),bz,bp) + \
            self.B9p*self.integrate(rhop*divJp+rhon*divJn,bz,bp)
            
    def var_rho(self,listOfFields,bz,bp):
        rhop,rhon,divJp,divJn = listOfFields
        divJ = divJp + divJn
        
        return self.B9*divJ + self.B9p*divJp, self.B9*divJ + self.B9p*divJn
    
    def var_div_J(self,listOfFields,bz,bp):
        rhop,rhon,divJp,divJn = listOfFields
        rho = rhop + rhon
        return self.B9*rho + self.B9p*rhop, self.B9*rho + self.B9p*rhon
        
class CoulombDirect_Gaussian(BaseSkyrme):
    def __init__(self,*args,nLeg=160,b=50):
        super().__init__(*args)
        
        #TODO: move to some global params file
        self.e2 = 1.43997840859651305 #electron charge squared
        
        self.nLeg = nLeg
        #Legendre nodes are $\zeta'$ from my notes
        legNodes, legWeights = np.polynomial.legendre.leggauss(self.nLeg)
        #Want only nodes and weights in interval [0,1], to match HFBTHO
        self.legNodes = legNodes[nLeg//2:]
        self.legWeights = legWeights[nLeg//2:]
        self.b = b
        
        self.zeta = self.legNodes
        self.a = 1/self.b * self.zeta/np.sqrt(1-self.zeta**2)
        
    def _get_field(self,rhop,bz,bp):
        Iarr = np.zeros((self.nLeg//2,)+rhop.shape)
        
        rVals = np.sqrt(self.eta)*bp
        zVals = self.xi*bz
        
        #TODO: this loop is slow, and could be sped up considerably with some
        #clever broadcasting
        for (i,a) in enumerate(self.a):
            for (j,r) in enumerate(rVals):
                for (k,z) in enumerate(zVals):
                    dist = (r-rVals[:,None])**2 + (z-zVals[None,:])**2
                    arrToIntegrate = np.exp(-dist * a**2)
                    arrToIntegrate *= special.ive(0,2*r*rVals*a**2)[:,None] * rhop
                    Iarr[i,j,k] = self.integrate(arrToIntegrate,bz,bp)
        
        arr = Iarr/(1-self.zeta[:,None,None]**2)**(3/2) * self.legWeights[:,None,None]
        arr = arr.sum(axis=0)
        
        return self.e2/(self.b*np.sqrt(np.pi))*arr
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.integrate(rhop*self._get_field(rhop,bz,bp),bz,bp)
    
    def var_rho(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return 2*self._get_field(rhop,bz,bp), np.zeros(rhop.shape)
    
class CoulombExchange(BaseSkyrme):
    def __init__(self,*args):
        super().__init__(*args)
        
        #TODO: move to some global params file
        e2 = 1.43997840859651305 #electron charge squared
        self.coeff = -3/4*e2*(3/np.pi)**(1/3)
    
    def get_eneg(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.coeff * self.integrate(rhop**(4/3),bz,bp)
    
    def var_rho(self,listOfFields,bz,bp):
        rhop, = listOfFields
        return self.coeff*4/3*rhop**(1/3), np.zeros(rhop.shape)

class Pairing_delta(BaseSkyrme):
    def __init__(self,V0,V1,g,rhoc,*args):
        super().__init__(*args)
        self.V0 = V0
        self.V1 = V1
        self.g = g
        self.rhoc = rhoc
        
        if g != 1:
            raise NotImplementedError
        
    def get_eneg(self,listOfFields,bz,bp):
        rhop,rhon,rhopPair,rhonPair = listOfFields
        arrp = self.V0[0]*(1-self.V1[0] * (rhop+rhon)/self.rhoc) * rhopPair**2
        arrn = self.V0[1]*(1-self.V1[1] * (rhop+rhon)/self.rhoc) * rhonPair**2
        
        return self.integrate(arrp+arrn,bz,bp)
    
    def var_rho(self,listOfFields,bz,bp):
        rhop,rhon,rhopPair,rhonPair = listOfFields
        return -self.V0[0]*self.V1[0]/self.rhoc*rhopPair**2, \
            -self.V0[1]*self.V1[1]/self.rhoc*rhonPair**2
            
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
        
    def var_rho(self,varRho,bz,bp):
        blockMats = []
        
        for k in range(self.basis.nBlocks):
            shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
            mat = np.zeros(shp)
            
            for mlInds in self.arrInds[k]:
                for (i1Iter,i1) in enumerate(mlInds):
                    for i2Iter in range(i1Iter+1):
                        i2 = mlInds[i2Iter]
                        #Simple things, like pre-allocating wf1 and wf2, don't
                        #speed this step up. Precomputing them will help,
                        #but I can handle that later, once I know exactly what
                        #needs to be precomputed
                        wf1 = self.basis.psi[k][i1]
                        wf2 = self.basis.psi[k][i2]
                        mat[i1,i2] = self.integrate(wf1*wf2*varRho,bz,bp)/(2*np.pi)
                    
            blockMats.append(symmetrize_array(mat))
            
        return blockMats
    
    def var_lapl_rho(self,varLaplRho,bz,bp):
        blockMats = []
        
        r = bp*np.sqrt(self.eta)
        
        for k in range(self.basis.nBlocks):
            shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
            mat = np.zeros(shp)
            
            psi = self.basis.psi[k]
            dpsi = self.basis.dpsi[k]
            d2psi = self.basis.d2psi[k]
            
            for mlInds in self.arrInds[k]:
                for (i1Iter,i1) in enumerate(mlInds):
                    for i2Iter in range(i1Iter+1):
                        i2 = mlInds[i2Iter]
                        
                        #The radial derivative
                        term1 = psi[i1]*d2psi[0,i2] + 2*dpsi[0,i1]*dpsi[0,i2] + d2psi[0,i1]*psi[i2]
                        term1 += (psi[i1]*dpsi[0,i2] + dpsi[0,i1]*psi[i2])/r[:,None]
                        
                        #The z derivative
                        term2 = psi[i1]*d2psi[5,i2] + 2*dpsi[2,i1]*dpsi[2,i2] + d2psi[5,i1]*psi[i2]
                        
                        mat[i1,i2] = self.integrate(varLaplRho*(term1+term2),bz,bp)/(2*np.pi)
            blockMats.append(symmetrize_array(mat))
            
        return blockMats
    
    def var_tau(self,varTau,bz,bp):
        blockMats = []
        
        r = bp*np.sqrt(self.eta)
        
        for k in range(self.basis.nBlocks):
            shp = 2*(len(self.basis.quantNumbersByBlock[k]),)
            mat = np.zeros(shp)
            
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
                        
                        mat[i1,i2] = self.integrate(varTau*(term1+term2+term3),bz,bp)/(2*np.pi)
            blockMats.append(symmetrize_array(mat))
        return blockMats
    
    def var_divJ(self,varDivJ,bz,bp):
        blockMats = []
        
        r = bp*np.sqrt(self.eta)
        
        for k in range(self.basis.nBlocks):
            Mr, Mphi, Mz = self.basis.M[k]
            
            dr, dphiIn, dz = self.basis.dpsi[k]
            #This copy statement is necessary, else results change call-to-call.
            #It doesn't have any impact at all on the code execution time
            dphi = dphiIn.copy()/r[None,:,None]
            
            mat = np.zeros(Mr.shape)
            for i1 in range(Mr.shape[0]):
                for i2 in range(i1+1):                    
                    term1 = (-dphi[i1])*dz[i2] - dz[i1]*dphi[i2]
                    term2 = dz[i1]*dr[i2] - dr[i1]*dz[i2]
                    term3 = dr[i1]*dphi[i2] - (-dphi[i1])*dr[i2]
                                            
                    arr = Mr[i1,i2]*term1 + Mphi[i1,i2] * term2 + Mz[i1,i2]*term3
                    
                    mat[i1,i2] = self.integrate(varDivJ*arr,bz,bp)/(2*np.pi)
            blockMats.append(symmetrize_array(mat))
            
        return blockMats
    """
    Leaving the following snippet as an example. JAX natively uses multiple
    CPU cores, and I don't think there's a way to restrict to one core. So,
    while this does save time (about 5 s of the 11 s runtime), it uses multiple
    CPU cores, so the comparison is faulty
    """
    # def var_divJ(self,varDivJ,bz,bp):
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
        vals, vecs = np.linalg.eigh(hfbArr[k])
        halfDim = vals.shape[0]//2
        
        #Positive-energy only
        eqp.append(vals[halfDim:])
        #For indexing, vecs[:,0] = (U[0],V[0]). Positive-energy states are the
        #upper half of the index
        U.append(vecs[:halfDim,halfDim:])
        V.append(vecs[halfDim:,halfDim:])
    return U, V, eqp

class PairingRegularization:
    """
    Organized as a class for readability reasons - the methods in this class
    depend on each other, but do different things
    """
    def __init__(self,basis,ebarMax=60,cutoffTol=10.**(-6)):
        self.basis = basis
        self.ebarMax = ebarMax
        self.cutoffTol = cutoffTol

    def get_active_states(self,Vin,eqpIn,chemPot):
        activeStates = [df.copy() for df in self.basis.quantNumbersByBlock]
    
        for k in range(self.basis.nBlocks):
            activeStates[k]['eqp'] = eqpIn[k]
            V = Vin[k]
            activeStates[k]['occ'] = np.einsum('ij,ij->j',V,V)
            
            activeStates[k]['ebar'] = (1-2*activeStates[k]['occ'])*activeStates[k]['eqp'] + chemPot
            activeStates[k]['del'] = 2*activeStates[k]['eqp']*np.sqrt(activeStates[k]['occ']*(1-activeStates[k]['occ']))
            
            activeStates[k]['isActive'] = (activeStates[k]['ebar'] <= self.ebarMax)
            
            #Basically, checking if we're close to the energy cutoff (I don't understand
            #where this comes from, though)
            
            #Want to truncate the exponent, and we do so using sys.float_info.max
            #https://stackoverflow.com/a/3477332 which is equivalent to Fortran's HUGE
            maxVal = sys.float_info.max
            
            exponent = maxVal * np.ones(len(activeStates[k]))
            borderInds = np.where(100*np.abs(activeStates[k]['ebar'] - self.ebarMax) < np.log(maxVal))
            exponent[borderInds] = np.exp(100*(activeStates[k]['ebar'].iloc[borderInds] - self.ebarMax))
            activeStates[k]['exponent'] = exponent
            
            #Pandas can be annoying, but this works
            goodInds = (1/(1+activeStates[k]['exponent']) > self.cutoffTol)
            activeStates[k].loc[goodInds,'isActive'] = True
            
        return activeStates
    
    def adjust_fermi_energy(self,activeStates,N,chemPot):
        def bcs_occ(lambd):
            occ = 0
            for k in range(self.basis.nBlocks):
                df = activeStates[k][activeStates[k]['isActive']]
                bcsOcc = 0.5*(1 - (df['ebar']-lambd)/np.sqrt((df['ebar']-lambd)**2 + df['del']**2))
                
                occ += bcsOcc.sum()
            return 2*occ - N
        
        def bcs_occ_derivative(lambd):
            ret = 0
            for k in range(self.basis.nBlocks):
                df = activeStates[k][activeStates[k]['isActive']]
                ek = np.sqrt((df['ebar']-lambd)**2 + df['del']**2)
                d = df['del']**2/(2*ek**3)
                
                ret += d.sum()
            return 2*ret

        sol = optimize.root_scalar(bcs_occ,x0=chemPot,fprime=bcs_occ_derivative,method='newton')
        return sol.root
    
def eigenvec_to_coordinate_space(U,V,basis,activeStates):
    Vup = []
    Vdown = []
    
    Uup = []
    Udown = []
    for k in range(basis.nBlocks):
        activeInds = np.arange(len(activeStates[k]))
        activeInds = np.where(activeStates[k]['isActive'])[0]
        if activeInds.size == 0:
            continue
        
        #Column Varr[:,i] corresponds to eigenvalue[i]. Since some states aren't active,
        #we want Varr[:,activeInds]. But, for the spin up/down component, we only want
        #Varr[spinUpInds,:]. So, we index as Varr[spin,activeInds], with appropriate
        #broadcasting b/c numpy
        subDf = activeStates[k]
        spinUp = np.where(subDf['2m_s']==1)[0]
        spinDown = np.where(subDf['2m_s']==-1)[0]
        
        Vup.append(np.einsum('ij,ikl->jkl',V[k][spinUp[:,None],activeInds[None,:]],
                             basis.psi[k][spinUp]))
        Vdown.append(np.einsum('ij,ikl->jkl',V[k][spinDown[:,None],activeInds[None,:]],
                               basis.psi[k][spinDown]))
        
        Uup.append(np.einsum('ij,ikl->jkl',U[k][spinUp[:,None],activeInds[None,:]],
                             basis.psi[k][spinUp]))
        Udown.append(np.einsum('ij,ikl->jkl',U[k][spinDown[:,None],activeInds[None,:]],
                               basis.psi[k][spinDown]))
        
    return Uup, Udown, Vup, Vdown
