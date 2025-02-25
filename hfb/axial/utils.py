import sys, os
import time
import traceback
import numpy as np
import h5py
import matplotlib.pyplot as plt

"""============================ Config Variables ==========================="""
class VariableNames:
    #Generic labels/keys
    phKeys = ['rho','del_rho','tau','divJ']
    ppKeys = ['rho_tilde',]
    
    titlesDict = {pn:{'rho':r'$\rho_'+pn+'$',
                      'del_rho':r'$\Delta\rho_'+pn+'$',
                      'tau':r'$\tau_'+pn+'$',
                      'divJ':r'div $J_'+pn+'$',
                      'rho_tilde':r'$\tilde{\rho}_'+pn+'$'}
                 for pn in ['n','p']}
    
    helRenamingDict = {'rho':'v','tau':'vhb','del_rho':'vd','divJ':'vs','rho_tilde':'dv'}
    defaultReconDensities = ['rho','tau','del_rho','divJ','rho_tilde']

class GlobalVariables:
    e2 = 1.43997840859651305 #electron charge squared
    h2m = 20.7355300000 #$\hbar^2/2m$
    printTimings = False#True

    defaultNCores = 4
    meshShape = (40,80)

"""============================ Boring Utilities ==========================="""
    
def timer(func):
    if GlobalVariables.printTimings:
        def inner(*args,**kwargs):
            t0 = time.time()
            ret = func(*args,**kwargs)
            t1 = time.time()
            
            print(func.__name__+' time: %.3e s'%(t1-t0))
            
            return ret
    else:
        def inner(*args,**kwargs):
            ret = func(*args,**kwargs)
            return ret
    return inner

#TODO
# def pn_wrapper(func):
#     dictOut = {pn:{} for pn in ['n','p']}

#     def inner(*args,**kwargs):

class Logging:
    def __init__(self,fName,paramsDicts):
        self.fName = fName
        if os.path.isfile(self.fName):
            os.remove(self.fName)
        
        self.write_nested_dict_as_arr('params',paramsDicts)

    def write_dict_as_attrs(self,nm,dictIn):
        with h5py.File(self.fName,'a') as h5File:
            h5File.create_group(nm)
            for (key,val) in dictIn.items():
                h5File[nm].attrs.create(key,val)
        return
    
    def _recurse_dict(self,dat,keyIn):
        if isinstance(dat,dict):
            for (key,subDat) in dat.items():
                fullKey = '/'.join((keyIn,key))
                self._recurse_dict(subDat,fullKey)    
        else:
            with h5py.File(self.fName,'a') as h5File:
                if dat is None: dat = 'None'
                h5File.create_dataset(keyIn,data=dat)
        return

    def write_nested_dict_as_arr(self,nm,datIn):
        with h5py.File(self.fName,'a') as h5File:
            h5File.create_group(nm)
            self._recurse_dict(datIn,nm)
        return

# Context manager that copies stdout and any exceptions to a log file
class OutputManager(object):
    #Taken from https://stackoverflow.com/a/57008707
    def __init__(self, filename):
        self.file = open(filename, 'w')
        self.stdout = sys.stdout
        self.stderr = sys.stderr

    def __enter__(self):
        sys.stdout = self
        sys.stderr = self

    def __exit__(self, exc_type, exc_value, tb):
        sys.stdout = self.stdout
        sys.stderr = self.stderr
        if exc_type is not None:
            self.file.write(traceback.format_exc())
        self.file.close()

    def write(self, data):
        self.file.write(data)

        self.flush()

    def flush(self):
        self.file.flush()
    
def check_if_run_needs_restart(logFile,eThresh=np.inf):
    """
    If run finished but energy difference is greater than eThresh,
    restarts the run
    """
    doRun = True

    if os.path.isfile(logFile):
        with open(logFile,'r') as fOpen:
            lns = fOpen.readlines()
        for ln in lns:
            if 'Energy difference' in ln:
                eDiff = float(ln.split()[2])
                if eDiff < eThresh:
                    doRun = False

    return doRun
    
def recursive_trim_dict(dat,lastIter):
    if isinstance(dat,dict):
        for (key,subDat) in dat.items():
            dat[key] = recursive_trim_dict(subDat,lastIter)
        return dat
    else:
        return dat[:lastIter]

"""============================ Array Utilities ==========================="""

def symmetrize_array(arr):
    #From https://stackoverflow.com/a/54277518
    # return np.tril(arr) + np.triu(arr.T, 1)
    return np.tril(arr) + np.triu(np.swapaxes(arr,-2,-1), 1)

def symmetrize_array_u(arr):
    #From https://stackoverflow.com/a/54277518
    # return np.tril(arr) + np.triu(arr.T, 1)
    return np.triu(arr) + np.tril(np.swapaxes(arr,-2,-1), -1)

def fill_arr(arr,inds,shp=(40,80),val=0.):
    paddedArr = val*np.ones(shp)
    paddedArr[*inds] = arr
    return paddedArr

class WavefunctionSpin:
    @staticmethod
    def split(U,V,quantNumDf):
        Vup = []
        Vdown = []
        
        Uup = []
        Udown = []

        for k in range(len(quantNumDf)):
            #Column Varr[:,i] corresponds to eigenvalue[i]. Since some states aren't active,
            #we want Varr[:,activeInds]. But, for the spin up/down component, we only want
            #Varr[spinUpInds,:]. So, we index as Varr[spin,activeInds], with appropriate
            #broadcasting b/c numpy
            subDf = quantNumDf[k]
            spinUp = np.where(subDf['2m_s']==1)[0]
            spinDown = np.where(subDf['2m_s']==-1)[0]
            
            Vup.append(V[k][spinUp])
            Vdown.append(V[k][spinDown])
            
            Uup.append(U[k][spinUp])
            Udown.append(U[k][spinDown])
            
        return Uup, Udown, Vup, Vdown
    
    @staticmethod
    def join(Uup,Udown,Vup,Vdown,quantNumDf):
        #TODO: elegant way of handling a single block at a time
        flatWfs = []
        for k in range(len(quantNumDf)):
            nBasisStates = len(quantNumDf[k])
            stackedArr = np.zeros((2*nBasisStates,nBasisStates))

            subDf = quantNumDf[k]
            spinUp = np.where(subDf['2m_s']==1)[0]
            spinDown = np.where(subDf['2m_s']==-1)[0]

            stackedArr[spinUp] = Uup[k]
            stackedArr[spinDown] = Udown[k]

            stackedArr[nBasisStates+spinUp] = Vup[k]
            stackedArr[nBasisStates+spinDown] = Vdown[k]

            flatWfs.append(stackedArr)
        return flatWfs
    
    @staticmethod
    def join_uv(Uup,Udown,Vup,Vdown,quantNumDf):
        #TODO: elegant way of handling a single block at a time
        flatU = []
        flatV = []
        for k in range(len(quantNumDf)):
            nBasisStates = len(quantNumDf[k])
            subDf = quantNumDf[k]
            spinUp = np.where(subDf['2m_s']==1)[0]
            spinDown = np.where(subDf['2m_s']==-1)[0]

            #U
            stackedArr = np.zeros((nBasisStates,nBasisStates))
            stackedArr[spinUp] = Uup[k]
            stackedArr[spinDown] = Udown[k]
            flatU.append(stackedArr)

            #V
            stackedArr = np.zeros((nBasisStates,nBasisStates))
            stackedArr[spinUp] = Vup[k]
            stackedArr[spinDown] = Vdown[k]
            flatV.append(stackedArr)
        return flatU, flatV

"""========================== Plotting Utilities ========================="""

def plot_arr(field,eta,xi,vmin=None,vmax=None,mesh=False,cmap='Spectral_r'):
    fig, ax = plt.subplots()
    if mesh:
        cf = ax.pcolormesh(np.sqrt(eta),xi,field.T,cmap=cmap,vmin=vmin,vmax=vmax)
    else:
        if vmin is None or vmax is None:
            levels = 30
        else:
            levels = np.linspace(vmin,vmax,num=30)
        cf = ax.contourf(np.sqrt(eta),xi,field.T,cmap=cmap,vmin=vmin,vmax=vmax,
                         extend='both',levels=levels)
    plt.colorbar(cf,ax=ax)
    ax.set(xlabel=r'$r/b_\perp$',ylabel=r'$z/b_z$')
    ax.set_aspect('equal')
    
    return fig, ax

def plot_side_by_side_arrays(arr1,arr2,eta,xi,titles=None,mesh=False,cmap='Spectral_r',
                             vmin=None,vmax=None,nLevels=30):
    textboxProps = {"boxstyle":'round', "facecolor":'white', "alpha":1,'pad':0.2}
    
    fig, ax = plt.subplots()
    
    rHere = np.sqrt(eta)
    arr = np.vstack([arr1[::-1],arr2])
    rVals = np.hstack([-rHere[::-1],rHere])
    if mesh:
        cf = ax.pcolormesh(rVals,xi,arr.T,cmap=cmap,)
    else:
        if vmin is None or vmax is None:
            levels = nLevels
        else:
            levels = np.linspace(vmin,vmax,num=nLevels)
        cf = ax.contourf(rVals,xi,arr.T,cmap=cmap,
                         extend='both',
                         levels=levels
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
