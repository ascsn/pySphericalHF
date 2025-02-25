"""
TODO: construct reduced basis for $\rho_p$, then use collocation + maxvol
to reconstruct $\rho_p$ from minimal dataset. This then goes into the integration
$$ \int dz' dr' r' \rho_p(r,r',z,z') $$

Common utilities for generating (especially) reduced datasets
from full HFB calculations

Data pipeline:
    -Complete HFB calculations
        -Initialize runs
        -Manage incomplete/incorrect runs
    -Extract data to folder for ease-of-access
#       -HFB energy
#       -Multipole moments
#       -Chemical potentials
#       -Fields
        -Lipkin-Nogami parameters
#       -Config file with the number of oscillator shells, oscillator widths
        -Runtime data
    -Single iteration to extract wavefunctions
#       -Optionally store HFB matrix in HO basis
#       -Read wavefunctions
#   -Generate training/testing indices
    -Compute SVD of fields
#       -Optionally normalize fields input (default to True)
#       -Save basis, real coefficients, basis in WF basis
        -Auto-save plot of singular values
#   -Generate densities
    -Compute SVD of wavefunctions
#       -Build and save
        -Auto-plot singular values
    -Compute collocation points w/ maxvol
#       -Make reconstruction everywhere
#       -Make reconstruction at tailored location
        -Clever thing w/ Coulomb
"""

import os
import numpy as np
import h5py
import pandas as pd

import sys, os
import glob
import matplotlib.pyplot as plt
import tqdm

hfbPath = '.'
sys.path.insert(0,hfbPath)
import solvers, utils

hfbScriptsPath = os.path.expanduser('~/Research/hfbtho_scripts/src/')
sys.path.insert(0,hfbScriptsPath)
from hfbtho_scripts import ReadHelFile, Thoout, CoordinateStrings

"""=================================== HFB management ==================================="""
class ManageHFB:
    def __init__(self):
        raise NotImplementedError
    
"""================================== Data Collection ==================================="""
def collect_fields(inDir,outDir,fName='fields.h5',
                   fieldsToGet=['rho','tau','del_rho','divJ','rho_tilde']):
    helFiles = sorted(glob.glob(os.path.join(inDir,'*/*hel'),recursive=True))

    dummyStrings = {'p':'FieldsP.','n':'FieldsN.'}
    
    os.makedirs(outDir,exist_ok=True)
    fName = os.path.join(outDir,fName)

    #Checking if file has expected amount of data. Not robust - basically
    #for ease-of-use in debugging my scripts as I write them
    if os.path.isfile(fName):
        with h5py.File(fName,'r') as h5File:
            if len(h5File.keys()) == len(helFiles):
                return

    #If not, writes the data to file
    with h5py.File(fName,'w') as h5File:
        for (fIter,f) in tqdm.tqdm(enumerate(helFiles)):
            groupStr = str(fIter).zfill(6)
            h5File.create_group(groupStr)
            
            obj = ReadHelFile(f,warn=False)
            for pn in ['n','p']:
                h5File[groupStr].create_group(pn)
                for field in fieldsToGet:
                    helStr = utils.VariableNames.helRenamingDict[field]+pn
                    h5File[groupStr][pn].\
                        create_dataset(field,data=obj.allDatDicts[dummyStrings[pn]][helStr].reshape(utils.GlobalVariables.meshShape))
    
    return None

def get_oscillator_config(inDir,outDir,fName='oscillator-config.dat'):
    #Reading if it exists
    os.makedirs(outDir,exist_ok=True)
    fName = os.path.join(outDir,fName)
    if os.path.isfile(fName):
        with open(fName,'r') as fOpen:
            lns = fOpen.readlines()
        bz = float(lns[0].split()[1])
        bp = float(lns[1].split()[1])
        N = int(lns[2].split()[1])
    else:
        #Making if it doesn't
        helFiles = sorted(glob.glob(os.path.join(inDir,'*/*hel'),recursive=True))
        
        bzVals = np.zeros(len(helFiles))
        bpVals = np.zeros(len(helFiles))
        nShellsVals = np.zeros(len(helFiles),dtype=int)
        
        for (fIter,f) in tqdm.tqdm(enumerate(helFiles)):
            obj = ReadHelFile(f,warn=False)
            bzVals[fIter] = obj.allDatDicts['HO-Basis']['bz']
            bpVals[fIter] = obj.allDatDicts['HO-Basis']['bp']
            nShellsVals[fIter] = obj.allDatDicts['HO-Basis']['n00']
        
        assert np.all(bzVals == bzVals[0])
        assert np.all(bpVals == bpVals[0])
        assert np.all(nShellsVals == nShellsVals[0])

        bz = bzVals[0]
        bp = bpVals[0]
        N = nShellsVals[0]

        lnsOut = ['bz %.12f\n'%bzVals[0],
                  'bp %.12f\n'%bpVals[0],
                  'N %d\n'%nShellsVals[0]]
        with open(fName,'w') as fOpen:
            fOpen.writelines(lnsOut)

    return bz, bp, N

def collect_moments_and_hfb_energy(inDir,outDir,fName='moments.dat'):
    thooutFiles = sorted(glob.glob(os.path.join(inDir,'*/thoout*'),recursive=True))

    df = pd.DataFrame(columns=['isConverged',]+CoordinateStrings.possibleCoords+['EHFB',],
                      index=range(len(thooutFiles)))
    for (fIter,f) in enumerate(thooutFiles):
        dat = Thoout.read(f,[])
        df.loc[fIter,'isConverged'] = dat[0]
        df.loc[fIter,CoordinateStrings.possibleCoords] = dat[1]
        df.loc[fIter,'EHFB'] = dat[3]
    
    df['isConverged'] = df['isConverged'].astype(bool)
    for col in df.columns[1:]:
        df[col] = df[col].astype(float)

    os.makedirs(outDir,exist_ok=True)
    df.to_csv(os.path.join(outDir,fName),sep='\t',
              index=False)
    
    return

def collect_chempots(inDir,outDir,fName='chempot.dat'):
    thooutFiles = sorted(glob.glob(os.path.join(inDir,'*/thoout*'),recursive=True))

    df = pd.DataFrame(columns=['lmdn','lmdp'],
                      index=range(len(thooutFiles)),
                      dtype=float)
    for (fIter,f) in enumerate(thooutFiles):
        df.loc[fIter] = Thoout.read_chempot(f)
    
    os.makedirs(outDir,exist_ok=True)
    df.to_csv(os.path.join(outDir,fName),sep='\t',
              index=False)
    
    return

def collect_ln_params(inDir,outDir,fName='ln-params.dat'):
    thooutFiles = sorted(glob.glob(os.path.join(inDir,'*/thoout*'),recursive=True))

    df = pd.DataFrame(columns=['lmd2n','lmd2p','eLNn','eLNp'],
                      index=range(len(thooutFiles)),
                      dtype=float)
    for (fIter,f) in enumerate(thooutFiles):
        dat = Thoout.read_ln_params(f)
        df.loc[fIter] = [dat['lmd2']['n'],dat['lmd2']['p'],
                         dat['eLN']['n'],dat['eLN']['p']]
    
    os.makedirs(outDir,exist_ok=True)
    df.to_csv(os.path.join(outDir,fName),sep='\t',
              index=False)
    
    return

"""============================== Wavefunction Generation ==============================="""
class GetWavefunctions:
    def __init__(self,runObj):
        self.runObj = runObj

    def main(self,fields,chemPot,otherArgsDict={'make':[],'diag':[]}):
        hfbArrDict = self.runObj.make_hfb_matrix(fields,chemPot,*otherArgsDict['make'])
        U, V, eqp = self.runObj.diagonalize_hfb_matrix(hfbArrDict,*otherArgsDict['diag'])
        activeStates, newChemPot = self.runObj.pair_reg(V,eqp,chemPot)

        return hfbArrDict, U, V, eqp, activeStates, newChemPot
    
    #This sort of thing feels template-able, but I can't think through it right now
    def split_by_spin(self,U,V,basis):
        Uup = {}
        Udown = {}
        Vup = {}
        Vdown = {}
        for pn in ['n','p']:
            Uup[pn], Udown[pn], Vup[pn], Vdown[pn] = utils.WavefunctionSpin.split(U[pn],V[pn],
                                                                                  basis.quantNumbersByBlock)
        
        return Uup, Udown, Vup, Vdown
    
    def write(self,fName,Uup,Udown,Vup,Vdown,activeStates,eqp=None,hfbArr=None):
        nBlocks = len(Uup['p'])
        with h5py.File(fName,'w') as h5File:
            for pn in ['n','p']:
                h5File.create_group(pn)
                
                wfGroup = '/'.join((pn,'wavefunction'))
                h5File.create_group(wfGroup)
                
                h5File[wfGroup].create_group('Uup')
                h5File[wfGroup].create_group('Udown')
                h5File[wfGroup].create_group('Vup')
                h5File[wfGroup].create_group('Vdown')

                h5File[pn].create_group('isActive')

                if eqp is not None:
                    h5File[pn].create_group('eqp')

                if hfbArr is not None:
                    h5File[pn].create_group('hfbArr')

                for k in range(nBlocks):
                    kStr = str(k).zfill(2)
                    h5File[wfGroup]['Uup'].create_dataset(kStr,data=Uup[pn][k])
                    h5File[wfGroup]['Udown'].create_dataset(kStr,data=Udown[pn][k])
                    h5File[wfGroup]['Vup'].create_dataset(kStr,data=Vup[pn][k])
                    h5File[wfGroup]['Vdown'].create_dataset(kStr,data=Vdown[pn][k])

                    h5File[pn]['isActive'].create_dataset(kStr,data=activeStates[pn][k]['isActive'])

                    if eqp is not None:
                        h5File[pn]['eqp'].create_dataset(kStr,data=eqp[pn][k])

                    if hfbArr is not None:
                        h5File[pn]['hfbArr'].create_dataset(kStr,data=hfbArr[pn][k])
        return
    
class ReadWavefunctions:
    def __init__(self,fName,pn=None,blockIter=None):
        self.fName = fName

        wfKeys = ['Uup','Udown','Vup','Vdown']
        for key in wfKeys:
            setattr(self,key,{pn:[] for pn in ['n','p']})
        
        for key in ['isActive','eqp','hfbArr']:
            setattr(self,key,{pn:[] for pn in ['n','p']})

        self.pn = pn
        self.blockIter = blockIter

        self.read()

    def read(self):
        with h5py.File(self.fName,'r') as h5File:
            for pn in ['n','p']:
                if self.pn is not None:
                    if self.pn != pn:
                        continue
                wfGroup = '/'.join((pn,'wavefunction'))
                nBlocks = len(h5File[wfGroup]['Uup'].keys())
                
                for k in range(nBlocks):
                    if self.blockIter is not None:
                        if self.blockIter != k:
                            continue
                    kStr = str(k).zfill(2)
                    self.Uup[pn].append(np.array(h5File[wfGroup]['Uup'][kStr]))
                    self.Udown[pn].append(np.array(h5File[wfGroup]['Udown'][kStr]))
                    self.Vup[pn].append(np.array(h5File[wfGroup]['Vup'][kStr]))
                    self.Vdown[pn].append(np.array(h5File[wfGroup]['Vdown'][kStr]))

                    self.isActive[pn].append(np.array(h5File[pn]['isActive'][kStr]))

                    if 'eqp' in h5File[pn].keys():
                        self.eqp[pn].append(np.array(h5File[pn]['isActive'][kStr]))
                    else:
                        self.eqp = None
                    if 'hfbArr' in h5File[pn].keys():
                        self.hfbArr[pn].append(np.array(h5File[pn]['hfbArr'][kStr]))
                    else:
                        self.hfbArr = None
        return
    
"""============================== Density Generation ==============================="""
def get_densities(fIn,fOut,r,basis,chempotDict):
    wfObj = ReadWavefunctions(fIn)

    densities = {pn:{} for pn in ['n','p']}

    for pn in ['n','p']:
        U, V = utils.WavefunctionSpin.join_uv(wfObj.Uup[pn],
                                              wfObj.Udown[pn],
                                              wfObj.Vup[pn],
                                              wfObj.Vdown[pn],
                                              basis.quantNumbersByBlock)
        
        pairReg = solvers.PairingRegularization(basis)
        activeStates = pairReg.get_active_states(V,wfObj.eqp[pn],chempotDict[pn])

        reconstruction = solvers.Reconstruction(basis,np.repeat(r[None,:,None],basis.psi[0].shape[-1],axis=-1))
        for key in utils.VariableNames.defaultReconDensities:
            densities[pn][key] = getattr(reconstruction,key)(activeStates,U,V)
        
    #Writing to file
    with h5py.File(fOut,'w') as h5File:
        for pn in ['n','p']:
            h5File.create_group(pn)
            for (key, arr) in densities[pn].items():
                h5File[pn].create_dataset(key,data=arr)

    return

def read_densities(fName):
    densities = {pn:{} for pn in ['n','p']}
    
    with h5py.File(fName,'r') as h5File:
        for pn in ['n','p']:
            for key in h5File[pn].keys():
                densities[pn][key] = np.array(h5File[pn][key])
    return densities

def read_all_densities(di,includeInds=None):
    if includeInds is None:
        includeInds = range(len(os.listdir(di)))

    densitiesInit = read_densities(os.path.join(di,str(includeInds[0]).zfill(6)+'.h5'))
    allDensities = {pn:{key:np.zeros((len(includeInds),)+utils.GlobalVariables.meshShape) 
                        for key in densitiesInit[pn].keys()}
                        for pn in ['n','p']}
    
    for (iIter,i) in enumerate(includeInds):
        densities = read_densities(os.path.join(di,str(i).zfill(6)+'.h5'))
        for pn in ['n','p']:
            for (key,arr) in densities[pn].items():
                allDensities[pn][key][iIter] = arr
    
    return allDensities

"""================================= Train-test indices ================================="""
#Future options include excluding certain indices based on convergence criteria

class TrainTestInds:
    def __init__(self,saveDir,nTotal=501,nSamples=100,fName='train-test-inds.h5'):
        fName = os.path.join(saveDir,fName)

        if os.path.isfile(fName):
            with h5py.File(fName,'r') as h5File:
                self.trainInds = np.array(h5File['trainInds'])
                self.valInds = np.array(h5File['valInds'])
        else:
            self.trainInds = np.random.choice(range(nTotal),size=nSamples,replace=False)
            self.valInds = list(set(range(nTotal)) - set(self.trainInds))

            os.makedirs(saveDir,exist_ok=True)
            with h5py.File(fName,'w') as h5File:
                h5File.create_dataset('trainInds',data=self.trainInds)
                h5File.create_dataset('valInds',data=self.valInds)

"""======================================= Fields ======================================="""
def read_fields(di,includeInds=None,fName='fields.h5'):
    fName = os.path.join(di,fName)
    
    with h5py.File(fName,'r') as h5File:
        #Initializing arrays
        if includeInds is None:
            includeInds = range(len(h5File.keys()))

        keys = list(h5File.keys())

        fieldsDict = {pn:{} for pn in ['n','p']}
        for pn in ['n','p']:
            for key in h5File[keys[0]][pn].keys():
                fieldsDict[pn][key] = np.zeros((len(includeInds),)+utils.GlobalVariables.meshShape)
        
        #Getting data
        for (iIter,i) in enumerate(includeInds):
            groupStr = str(i).zfill(6)
            for pn in ['n','p']:
                for (key,arr) in h5File[groupStr][pn].items():
                    fieldsDict[pn][key][iIter] = np.array(arr)
    return fieldsDict

class FieldsSVD:
    #TODO: auto-save plots of principal components (maybe?)
    #TODO: make it so I can normalize individual fields. Only really
    #interested when pairing collapsed, so feed in as nested dictionary

    #TODO: examine how normalizing changes things. I guess, since the
    #magnitude of the fields is important, normalizing it may not be
    #appropriate. Certainly, in general it works without normalizing
    #the fields prior to the SVD.

    #When normalizing, drastically more principal components are left,
    #in many cases double what's left with unnormalized data. This
    #relative inefficiency is why the default option is False
    def __init__(self,fieldsDict,normalize=False):
        self.fieldsDict = fieldsDict
        self.uFull = None
        self.sFull = None
        self.vtFull = None

        self.normalize = normalize
    
    def truncated_svd(self,tol):
        if self.uFull is None:
            self.uFull = {pn:{} for pn in ['n','p']}
            self.sFull = {pn:{} for pn in ['n','p']}
            self.vtFull = {pn:{} for pn in ['n','p']}

            for pn in ['n','p']:
                for (key,arr) in self.fieldsDict[pn].items():
                    arrIn = arr.reshape((arr.shape[0],-1))
                    if self.normalize:
                        arrIn /= np.linalg.norm(arrIn,axis=0,keepdims=True)
                    self.uFull[pn][key], self.sFull[pn][key], self.vtFull[pn][key] =\
                        np.linalg.svd(arrIn)
        
        minInds = {pn:{} for pn in ['n','p']}
        for pn in ['n','p']:
            for (key,s) in self.sFull[pn].items():
                extraInds = np.where(s/s[0] <= tol)[0]
                if len(extraInds) > 0:
                    minInds[pn][key] = extraInds[0]
                else:
                    minInds[pn][key] = len(s)
        return {pn:{key:arr[:minInds[pn][key]].reshape((-1,)+utils.GlobalVariables.meshShape)
                    for (key,arr) in self.vtFull[pn].items()}
                for pn in ['n','p']}, minInds

    def get_real_coeffs(self,basisDict,arrDict):
        #Doesn't *need* to be a class method, but b/c data structures are
        #funky, just have one function for fields and another for wavefunctions
        realCoeffsDict = {pn:{} for pn in ['n','p']}
        for pn in ['n','p']:
            for (key,arr) in arrDict[pn].items():
                flatBasis = basisDict[pn][key].reshape((basisDict[pn][key].shape[0],-1))
                flatArr = arr.reshape((arr.shape[0],-1))
                realCoeffsDict[pn][key] = flatArr @ flatBasis.T
        return realCoeffsDict
    
    def write(self,di,basisDict,minIndsDict,realCoeffsDict,fName='field-svd.h5'):
        os.makedirs(di,exist_ok=True)
        fName = os.path.join(di,fName)

        if not os.path.isfile(fName):
            with h5py.File(fName,'w') as h5File:
                for pn in ['n','p']:
                    h5File.create_group(pn)
                    for key in basisDict[pn].keys():
                        h5File[pn].create_group(key)

                        h5File[pn][key].attrs.create('minInds',minIndsDict[pn][key])
                        h5File[pn][key].create_dataset('basis',data=basisDict[pn][key])
                        h5File[pn][key].create_dataset('realCoeffs',data=realCoeffsDict[pn][key])
        return
    
class FieldsBasisConfigSpace:
    def __init__(self,csBasis,hfbMatObj):
        self.csBasis = csBasis
        self.hfbMatObj = hfbMatObj
        self.fieldsConfigSpace = None

    def get_basis_config_space(self,fieldsBasis,minIndsDict):
        #Caches based on different tolerances being precomputed. Since this will
        #run rarely, I won't bother making it robust to mistakes
        if self.fieldsConfigSpace is None:
            self.fieldsConfigSpace = {pn:{} for pn in ['n','p']}
            for (keyIter,key) in enumerate(fieldsBasis['n'].keys()):
                print(key)
                #For vectorization purposes, we stack p and n at once, then split after
                splitInd = minIndsDict['n'][key]

                inputArr = np.vstack([fieldsBasis['n'][key],fieldsBasis['p'][key]])
                integs = getattr(self.hfbMatObj,key)(inputArr,
                                                     self.csBasis.bz,
                                                     self.csBasis.bp)
                self.fieldsConfigSpace['n'][key] = [arr[:splitInd] for arr in integs]
                self.fieldsConfigSpace['p'][key] = [arr[splitInd:] for arr in integs]
            ret = self.fieldsConfigSpace
        else:
            #Assumes that fewer data points are needed than what's already cached
            ret = {pn:{} for pn in ['n','p']}
            for pn in ['n','p']:
                for (key,lst) in self.fieldsConfigSpace[pn].items():
                    ret[pn][key] = [arr[:minIndsDict[pn][key]] for arr in lst]
        
        return ret
    
    def write(self,di,fieldsCsDict,fName='fields-ho-basis.h5'):
        os.makedirs(di,exist_ok=True)
        fName = os.path.join(di,fName)

        if not os.path.isfile(fName):
            with h5py.File(fName,'w') as h5File:
                for pn in ['n','p']:
                    h5File.create_group(pn)
                    for (key, lst) in fieldsCsDict[pn].items():
                        h5File[pn].create_group(key)
                        for (arrIter,arr) in enumerate(lst):
                            h5File[pn][key].create_dataset(str(arrIter).zfill(2),data=arr)

        return
    
def read_fields_svd(fName):
    svdBasisDict = {pn:{} for pn in ['n','p']}
    realCoeffsDict = {pn:{} for pn in ['n','p']}
    minInds = {pn:{} for pn in ['n','p']}

    with h5py.File(fName,'r') as h5File:
        for pn in ['n','p']:
            for key in h5File[pn].keys():
                svdBasisDict[pn][key] = np.array(h5File[pn][key]['basis'])
                realCoeffsDict[pn][key] = np.array(h5File[pn][key]['realCoeffs'])
                
                minInds[pn][key] = h5File[pn][key].attrs['minInds']
    return svdBasisDict, minInds, realCoeffsDict

def read_fields_config_space(fName):
    fieldsDict = {pn:{} for pn in ['n','p']}
    with h5py.File(fName,'r') as h5File:
        for pn in ['n','p']:
            for key in h5File[pn].keys():
                fieldsDict[pn][key] = []
                for iterKey in sorted(h5File[pn][key].keys()):
                    fieldsDict[pn][key].append(np.array(h5File[pn][key][iterKey]))
    return fieldsDict

"""=================================== Wavefunctions ===================================="""
class WavefunctionSVD:
    def __init__(self,wfsArr,isActiveArr,nPad):
        self.wfsArr = wfsArr
        self.isActiveArr = isActiveArr
        self.nToInclude = self.get_nstates(nPad)
        
        self.sFull = None
        self.vtFull = None

    def get_nstates(self,nPad):
        isActiveSum = np.any(self.isActiveArr,axis=0)
    
        nToInclude = len(np.where(isActiveSum)[0])
        nToInclude = min(nToInclude+nPad,len(isActiveSum))
        nToInclude = max(nToInclude,1)
        return nToInclude
    
    def truncated_svd(self,tol):
        if self.sFull is None:
            sz = self.wfsArr.shape[1]

            stackedArr = self.wfsArr[:,:,:self.nToInclude]
            flatArr = np.swapaxes(stackedArr,1,2).reshape((-1,sz))
            
            s = np.linalg.svd(flatArr,compute_uv=False)

            arr = flatArr.T @ flatArr
            _, vecs = np.linalg.eigh(arr)
            vecs = vecs[:,::-1]

            self.sFull = s
            self.vtFull = vecs
        
        if len(np.where(self.sFull/self.sFull[0] < tol)[0]) > 0:
            nNeeded = np.where(self.sFull/self.sFull[0] < tol)[0][0]
        else:
            nNeeded = len(self.sFull)

        basisArr = self.vtFull[:,:nNeeded].T

        return basisArr, nNeeded, self.sFull[:nNeeded]
    
    def write(self,fName,basisArr,nNeeded,singularVals):
        if not os.path.isfile(fName):
            with h5py.File(fName,'w') as h5File:
                h5File.create_dataset('basis',data=basisArr)
                h5File.attrs.create('nStatesToGet',nNeeded)
                h5File.create_dataset('singularVals',data=singularVals)
        return
    
    def plot_svd(self,fName,nNeeded):
        fig, ax = plt.subplots()
        ax.plot(self.sFull/self.sFull[0])
        ax.axvline(nNeeded,color='black')
        ax.set(yscale='log',xlabel='Component',ylabel=r'$\sigma/\sigma_0$')
        fig.savefig(fName,bbox_inches='tight')
        return
    
def read_wf_basis(di,nBlocks):
    wfBasisDict = {pn:[] for pn in ['n','p']}
    wfStatesToGet = {pn:np.zeros(nBlocks,dtype=int) for pn in ['n','p']}
    singularValues = {pn:[] for pn in ['n','p']}

    for pn in ['n','p']:
        for blockIter in range(nBlocks):
            blockStr = str(blockIter).zfill(2)
            fName = os.path.join(di,pn,'wfBasis_'+blockStr+'.h5')

            with h5py.File(fName,'r') as h5File:
                basisArr = np.array(h5File['basis'])
                s = np.array(h5File['singularVals'])
                nStatesToGet = h5File.attrs['nStatesToGet']
            
            wfBasisDict[pn].append(basisArr)
            wfStatesToGet[pn][blockIter] = nStatesToGet
            singularValues[pn].append(s)

    return wfBasisDict, wfStatesToGet, singularValues
    
"""================================== Reconstruction Matrices ==================================="""
def reconstruction_matrix(arr):
    M = arr.reshape((arr.shape[0],-1)).T
    return np.linalg.inv((M.T @ M)) @ M.T

def get_global_reconstruction_matrices(inFile,outFile):
    svdBasisDict, _, _ = read_fields_svd(inFile)

    with h5py.File(outFile,'w') as h5File:
        for pn in ['n','p']:
            h5File.create_group(pn)
            for (key,arr) in svdBasisDict[pn].items():
                h5File[pn].create_group(key)
                psInv = reconstruction_matrix(arr)
                h5File[pn][key].create_dataset('psInv',data=psInv)
    return 

def maxvol(basis, indxGuess,maxIters=100, eps=10**(-10)):
    r"""basis looks like a long matrix, the columns are the "pillars" V_i(x):
    [   V_1(x)
        V_2(x)
        .
        .
        .
    ]
    indxGuess is a first guess of where we should "measure", or ask the questions

    Shamelessly taken from the ROSE package

    """
    nbases = basis.shape[1]
    interpBasis = np.copy(basis)

    for ij in range(len(indxGuess)):
        interpBasis[[ij, indxGuess[ij]], :] = interpBasis[[indxGuess[ij], ij], :]
    indexing = np.array(range(len(interpBasis)))

    for ij in range(len(indxGuess)):
        indexing[[ij, indxGuess[ij]]] = indexing[[indxGuess[ij], ij]]

    bHist = np.zeros(maxIters)

    for iIn in range(maxIters):
        B = np.dot(interpBasis, np.linalg.inv(interpBasis[:nbases]))
        b = np.max(B)
        bHist[iIn] = b
        if b - eps > 1:
            p1, p2 = np.where(B == b)[0][0], np.where(B == b)[1][0]
            interpBasis[[p1, p2], :] = interpBasis[[p2, p1], :]
            indexing[[p1, p2]] = indexing[[p2, p1]]
        else:
            break
        # this thing returns the indices of where we should measure
    # return np.sort(indexing[:nbases]), iIn, bHist
    return indexing, iIn, bHist

class MaxvolReconstruction:
    """
    We know which fields need which wavefunctions/derivatives to recompute.
    This class makes sure that we construct everything at the required locations
    so that we don't e.g. recompute $V_k$ multiple times.

    It does so by constructing a superset of indices at which to rebuild all fields.
    If for instance $\rho_n$ needs to be reconstructed at $(a,b)$ and $\del_rho_n$
    needs reconstruction at $(b,d,g)$, both will be reconstructed at $(a,b,d,g)$.

    The exception is $\rho_p$. For the direct Coulomb term, $\rho_p$ is reconstructed
    everywhere.
    """
    def __init__(self,inFile,outFile):
        self.main(inFile,outFile)
    
    def get_supset_inds(self,basisFields):
        """
        Essentially a wrapper around maxvol
        """
        indsDict = {pn:{} for pn in ['n','p']}
        supsetInds = set()
        for pn in ['n','p']:
            for (key,basis) in basisFields[pn].items():
                indexing, _, _ = maxvol(basis.reshape((basis.shape[0],-1)).T,range(100))
                indsDict[pn][key] = indexing[:basis.shape[0]]

                supsetInds = supsetInds | set(indsDict[pn][key]) #Set union
        supsetInds = list(supsetInds)

        return indsDict, supsetInds
    
    def main(self,inFile,outFile):
        
        basisFields, _, _ = read_fields_svd(inFile)
        indsDict, supsetInds = self.get_supset_inds(basisFields)

        reshapedInds = np.unravel_index(supsetInds,utils.GlobalVariables.meshShape)

        with h5py.File(outFile,'w') as h5File:
            for pn in ['n','p']:
                h5File.create_group(pn)
                for (key,arr) in basisFields[pn].items():
                    h5File[pn].create_group(key)
                    psInv = reconstruction_matrix(arr[:,*reshapedInds])
                    
                    h5File[pn][key].create_dataset('psInv',data=psInv)
                    if not (pn == 'p' and key == 'rho'):
                        h5File[pn][key].create_dataset('idx',
                                                       data=reshapedInds)
        return

class MaxvolReconstruction_rhop:
    def __init__(self,fieldsFile,densitiesFile,outFile):
        self.main(fieldsFile,densitiesFile,outFile)
    
    def get_supset_inds(self,basisFields,rhop):
        """
        Essentially a wrapper around maxvol
        """
        indsDict = {pn:{} for pn in ['n','p']}
        supsetInds = set()
        for pn in ['n','p']:
            for (key,basis) in basisFields[pn].items():
                indexing, _, _ = maxvol(basis.reshape((basis.shape[0],-1)).T,range(100))
                indsDict[pn][key] = indexing[:basis.shape[0]]

                supsetInds = supsetInds | set(indsDict[pn][key]) #Set union
        indexing, _, _ = maxvol(rhop.reshape((rhop.shape[0],-1)).T,range(100))
        indsDict['rhop'] = indexing[:rhop.shape[0]]
        supsetInds = supsetInds | set(indsDict['rhop']) #Set union

        supsetInds = list(supsetInds)

        return indsDict, supsetInds
    
    def main(self,fieldsFile,densitiesFile,outFile):
        
        basisFields, _, _ = read_fields_svd(fieldsFile)
        
        basisDensities, _, _ = read_fields_svd(densitiesFile)
        rhop = basisDensities['p']['rho']

        indsDict, supsetInds = self.get_supset_inds(basisFields,rhop)

        reshapedInds = np.unravel_index(supsetInds,utils.GlobalVariables.meshShape)

        with h5py.File(outFile,'w') as h5File:
            for pn in ['n','p']:
                h5File.create_group(pn)
                for (key,arr) in basisFields[pn].items():
                    h5File[pn].create_group(key)
                    psInv = reconstruction_matrix(arr[:,*reshapedInds])
                    
                    h5File[pn][key].create_dataset('psInv',data=psInv)
                    h5File[pn][key].create_dataset('idx',data=reshapedInds)

            h5File.create_group('rhop_density')
            psInv = reconstruction_matrix(rhop[:,*reshapedInds])
                    
            h5File['rhop_density'].create_dataset('psInv',data=psInv)
            h5File['rhop_density'].create_dataset('idx',data=reshapedInds)
        return
    
def read_reconstruction_mats(fIn):
    pseudoInverseDict = {pn:{} for pn in ['n','p']}
    reconInds = {pn:{} for pn in ['n','p']}

    with h5py.File(fIn,'r') as h5File:
        allKeys = set(h5File.keys())
        for pn in ['n','p']:
            for key in h5File[pn].keys():
                pseudoInverseDict[pn][key] = np.array(h5File[pn][key]['psInv'])
                
                if 'idx' in h5File[pn][key].keys():
                    reconInds[pn][key] = np.array(h5File[pn][key]['idx'])
                else:
                    reconInds[pn][key] = None
            allKeys -= set([pn,])
        for key in allKeys:
            pseudoInverseDict[key] = np.array(h5File[key]['psInv'])
                
            if 'idx' in h5File[key].keys():
                reconInds[key] = np.array(h5File[key]['idx'])
            else:
                reconInds[key] = None

    return pseudoInverseDict, reconInds
    
"""================================== Misc File IO ==================================="""
def read_chempot(di,returnAsDict=True):
    chemPot = pd.read_csv(os.path.join(di,'chempot.dat'),sep='\s+')

    if returnAsDict:
        return {'p':chemPot['lmdp'],'n':chemPot['lmdn']}
    else:
        return chemPot
    
def read_ln_params(di,returnAsDict=True):
    df = pd.read_csv(os.path.join(di,'ln-params.dat'),sep='\s+')

    if returnAsDict:
        return {'p':df['lmd2p'],'n':df['lmd2n']}, {'p':df['eLNp'],'n':df['eLNn']}
    else:
        return df