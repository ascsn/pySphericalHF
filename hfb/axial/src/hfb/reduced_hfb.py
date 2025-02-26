import sys, os

import numpy as np
import time

hfbPath = '.'
sys.path.insert(0,hfbPath)
import solvers, utils, runfiles

def get_hfb_in_reduced_basis(hfbMatrixBasis,wfBasisDict):
    """
    Now treats only upper triangular part of reduced HFB matrix
    """
    hfbTerms = {pn:{key:[] for key in ['UU_VV','UhU_VhV','VhtU_UhtV']} for pn in ['n','p']}
    hfbArrShapes = {pn:[] for pn in ['n','p']}

    for pn in ['n','p']:
        for blockIter in range(len(wfBasisDict[pn])):
            reducedWfBasis = wfBasisDict[pn][blockIter]
            uvCtr = reducedWfBasis.shape[1]//2 #For splitting into U and V
            
            U = reducedWfBasis[:,:uvCtr]
            V = reducedWfBasis[:,uvCtr:]

            uOuter = U @ U.T
            vOuter = V @ V.T

            hfbArrShapes[pn].append(uOuter.shape[0])
            idx = np.triu_indices_from(uOuter)

            hfbTerms[pn]['UU_VV'].append((uOuter - vOuter)[idx])

            #Particle channel
            hUhVDict = {}
            
            for key in utils.VariableNames.phKeys:
                UhU = U @ hfbMatrixBasis[pn][key][blockIter] @ U.T
                VhV = -V @ hfbMatrixBasis[pn][key][blockIter] @ V.T
                
                idx = np.triu_indices_from(UhU[0])
                hUhVDict[key] = (UhU + VhV)[:,*idx]

            hfbTerms[pn]['UhU_VhV'].append(hUhVDict)

            #Pairing channel
            hTildeDict = {}
            for key in utils.VariableNames.ppKeys:
                arr = V @ hfbMatrixBasis[pn][key][blockIter] @ U.T

                idx = np.triu_indices_from(arr[0])
                hTildeDict[key] = (arr + np.swapaxes(arr,1,2))[:,*idx]

            hfbTerms[pn]['VhtU_UhtV'].append(hTildeDict)
    return hfbTerms, hfbArrShapes

"""========================= Single Iteration Code ========================="""
class ReducedHFB(runfiles.Main):
    def __init__(self,edf,basis,basisMatsDict,nParticles,
                 r,bz,bp,laplaceMode='exact',earlyStop=False,
                 densitiesToConstruct=utils.VariableNames.defaultReconDensities):
        self.edf = edf
        self.basis = basis
        self.needsReconstruction = {'n':True,'p':True}
        self.basisMatsDict = basisMatsDict
        self.pairRegObjs = {pn:solvers.PairingRegularization(basis) for pn in ['n','p']}
        self.nParticles = nParticles

        self.laplaceMode = laplaceMode
        self.psObj = solvers.PSDerivatives(basis.nr,basis.nz)

        self.earlyStop = earlyStop
        
        self.bz = bz
        self.bp = bp
        
        self.rGridRepeated = np.repeat(r[None,:,None],basis.psi[0].shape[-1],axis=-1)
        self.needsReconstruction = {pn:{key:True for key in basisMatsDict[pn].keys()} for pn in ['n','p']}
        self.densities = {pn:{} for pn in ['n','p']}

        self.densitiesToConstruct = densitiesToConstruct
    
    @utils.timer
    def make_reduced_hfb_matrix(self,coeffsIn,chemPot):
        hfbArrDict = {'p':[],'n':[]}
        for pn in ['n','p']:
            for k in range(self.basis.nBlocks):
                blockSize = self.basis.psi[k].shape[0]
                hfbArr = np.zeros(2*(2*blockSize,))
                
                for key in utils.VariableNames.phKeys:
                    hfbArr[:blockSize,:blockSize] += np.einsum('i,ijk->jk',coeffsIn[pn][key],
                                                               self.basisMatsDict[pn][key][k])
                for key in utils.VariableNames.ppKeys:
                    hfbArr[blockSize:,:blockSize] += np.einsum('i,ijk->jk',coeffsIn[pn][key],
                                                               self.basisMatsDict[pn][key][k])
                hfbArr[blockSize:,blockSize:] = -hfbArr[:blockSize,:blockSize]
                hfbArr[:blockSize,blockSize:] = hfbArr[blockSize:,:blockSize].T
                    
                hfbArr[:blockSize,:blockSize] -= chemPot[pn] * np.identity(blockSize)
                hfbArr[blockSize:,blockSize:] += chemPot[pn] * np.identity(blockSize)
                
                hfbArrDict[pn].append(hfbArr)
        return hfbArrDict

    # @utils.timer
    # def diagonalize_hfb_matrix(self,hfbArrDict):
    #     Udict = {}
    #     Vdict = {}
    #     eqpDict = {}
    
    #     for pn in ['n','p']:
    #         Udict[pn], Vdict[pn], eqpDict[pn] = solvers.diagonalize_hfb_matrix(hfbArrDict[pn])
        
    #     return Udict, Vdict, eqpDict
    
    # @utils.timer
    # # @profile
    # def pair_reg(self,V,eqp,chemPot):
    #     activeStates = {pn:self.pairRegObjs[pn].get_active_states(V[pn],eqp[pn],chemPot[pn])
    #                        for pn in ['n','p']}
    #     newChemPot = {pn:self.pairRegObjs[pn].adjust_fermi_energy(activeStates[pn],
    #                                                               self.nParticles[pn],chemPot[pn])
    #                                                               for pn in ['n','p']}
        
    #     return activeStates, newChemPot

    @utils.timer
    # @profile
    def reconstruct_densities(self,activeStates,U,V,reconstructionCoordInds,returnReconObj=False):
        densities = {pn:{} for pn in ['n','p']}

        if returnReconObj:
            objs = {}

        for pn in ['n','p']:
            if reconstructionCoordInds is None:
                reconstruction = solvers.Reconstruction(self.basis,self.rGridRepeated,
                                                        laplaceMode=self.laplaceMode,laplaceOpts={'derivativeObj':self.psObj})
            else:
                reconstruction = solvers.Reconstruction(self.basis,
                                                        self.rGridRepeated[:,*reconstructionCoordInds])
            for key in self.densitiesToConstruct:
                if self.needsReconstruction[pn][key]:
                    densities[pn][key] = getattr(reconstruction,key)(activeStates[pn],U[pn],V[pn],
                                                                     coordInds=reconstructionCoordInds)
                    self.densities[pn][key] = densities[pn][key]
                else:
                    densities[pn][key] = self.densities[pn][key]
                
            if returnReconObj:
                objs[pn] = reconstruction

        if returnReconObj:
            return densities, objs
        else:
            return densities

    @utils.timer
    def reconstruct_fields(self,densities,reconstructionCoordInds):
        #TODO: always reconstructs fields regardless of necessity
        newFields = {pn:{key:np.zeros(utils.GlobalVariables.meshShape) for key in self.densitiesToConstruct}
                      for pn in ['n','p']}
        
        for var in self.densitiesToConstruct:
            for (key,obj) in self.edf.terms.items():
                integArgs = [densities[tup[0]][tup[1]] for tup in self.edf.integArgs[key]]
                pField, nField = getattr(obj,var)(integArgs,self.bz,self.bp)
                
                if pField.shape == utils.GlobalVariables.meshShape:
                    newFields['p'][var] += pField
                else:
                    paddedArr = np.zeros(utils.GlobalVariables.meshShape)
                    paddedArr[*reconstructionCoordInds] = pField
                    newFields['p'][var] += paddedArr
                
                if nField.shape == utils.GlobalVariables.meshShape:
                    newFields['n'][var] += nField
                else:
                    paddedArr = np.zeros(utils.GlobalVariables.meshShape)
                    paddedArr[*reconstructionCoordInds] = nField
                    newFields['n'][var] += paddedArr
        
        #The kinetic variation has constant $\hbar^2/2m$ in it
        if reconstructionCoordInds is not None:
            for pn in ['n','p']:
                newFields[pn]['tau'][*reconstructionCoordInds] -= utils.GlobalVariables.h2m
                newFields[pn]['tau'] += utils.GlobalVariables.h2m
        
        return newFields

    @utils.timer
    def get_new_coeffs(self,fields,pseudoInverseDict):
        newCoeffsDict = {pn:{} for pn in ['n','p']}

        for pn in ['n','p']:
            for (key,field) in fields[pn].items():
                newCoeffsDict[pn][key] = pseudoInverseDict[pn][key] @ field.flatten()
                
        return newCoeffsDict
    
    @utils.timer
    # @profile
    def single_iter(self,coeffs,chemPot,oldDensities,
                    reconstructionCoordInds,pseudoInverseDict,tol):
        timeDict = {}
        
        #The last 2 arguments define the coordinates we reconstruct the fields
        #on. I'm not sure if it makes sense to let you change them while the
        #code is running, or if the class should be remade with those options
        #fixed
        t0 = time.time()
        hfbArrDict = self.make_reduced_hfb_matrix(coeffs,chemPot)
        t1 = time.time()
        timeDict['make_reduced_hfb_matrix'] = t1 - t0
        
        t0 = time.time()
        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        t1 = time.time()
        timeDict['diagonalize_hfb_matrix'] = t1 - t0

        t0 = time.time()
        activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)
        t1 = time.time()
        timeDict['pair_reg'] = t1 - t0

        t0 = time.time()
        densities = self.reconstruct_densities(activeStates,Udict,Vdict,
                                               reconstructionCoordInds)
        t1 = time.time()
        timeDict['reconstruct_densities'] = t1 - t0
        
        t0 = time.time()
        fields = self.reconstruct_fields(densities,reconstructionCoordInds)
        t1 = time.time()
        timeDict['reconstruct_fields'] = t1 - t0

        t0 = time.time()
        newCoeffs = self.get_new_coeffs(fields,pseudoInverseDict)
        t1 = time.time()
        timeDict['get_new_coeffs'] = t1 - t0
        
        #Seems to lower precision with negligible runtime improvement
        if self.earlyStop:
            for pn in ['n','p']:
                for (key,arr) in newCoeffs[pn].items():
                    if np.abs(arr - coeffs[pn][key]).max() <= tol:
                        self.needsReconstruction[pn][key] = False
        return newCoeffs, newChemPot, densities, timeDict

class ReducedHFB_FancyDiag(ReducedHFB):
    def __init__(self, edf, basis, basisMatsDict, hfbMatShapes, nParticles, r, bz, bp, 
                 nStatesToGet, wfBasisDict,
                 laplaceMode='exact', earlyStop=False):
        super().__init__(edf, basis, basisMatsDict, nParticles, r, bz, bp, laplaceMode, earlyStop)
        
        self.nStatesToGet = nStatesToGet
        self.wfBasisDict = wfBasisDict
        self.hfbMatShapes = hfbMatShapes
        self.needsReconstruction = {pn:{key:True for key in ['rho','del_rho','tau','divJ','rho_tilde']}
                                     for pn in ['n','p']}

    def make_reduced_hfb_matrix(self, coeffsIn, chemPot):
        hfbArrDict = {pn:[] for pn in ['n','p']}
        for pn in ['n','p']:
            for k in range(self.basis.nBlocks):
                uvOuter = self.basisMatsDict[pn]['UU_VV'][k]

                hUhVDict = self.basisMatsDict[pn]['UhU_VhV'][k]
                hTildeDict = self.basisMatsDict[pn]['VhtU_UhtV'][k]

                #Particle channel
                hfbMat = np.zeros(uvOuter.shape)
                for key in utils.VariableNames.phKeys:
                    hfbMat += coeffsIn[pn][key] @ hUhVDict[key]
                
                #Pairing channel
                for key in utils.VariableNames.ppKeys:
                    hfbMat += coeffsIn[pn][key] @ hTildeDict[key]
                    
                #Chemical potential
                hfbMat -= chemPot[pn]*uvOuter

                #Reshaping
                sz = self.hfbMatShapes[pn][k]
                hfbArrFull = np.zeros(2*(sz,))
                hfbArrFull[np.triu_indices(sz)] = hfbMat

                hfbArrDict[pn].append(hfbArrFull)
                
        return hfbArrDict

    @utils.timer
    # @profile
    def diagonalize_hfb_matrix(self, hfbArrDict):
        Udict = {pn:[] for pn in ['n','p']}
        Vdict = {pn:[] for pn in ['n','p']}
        eqpDict = {pn:[] for pn in ['n','p']}
    
        for pn in ['n','p']:
            for k in range(self.basis.nBlocks):
                vals, vecs = np.linalg.eigh(hfbArrDict[pn][k],UPLO='U')
                stateInds = np.where(vals>0)[0][:self.nStatesToGet[pn][k]]
                # stateInds = np.where(vals>0)[0]

                vals = vals[stateInds]
                vecs = vecs[:,stateInds]

                eqpDict[pn].append(vals)
                eigenvecsHO = vecs.T @ self.wfBasisDict[pn][k]
                Udict[pn].append(eigenvecsHO[:,:self.basis.psi[k].shape[0]].T)
                Vdict[pn].append(eigenvecsHO[:,self.basis.psi[k].shape[0]:].T)
        
        return Udict, Vdict, eqpDict
        
class Reconstruction_Collocation(solvers.Reconstruction):
    """
    This class does the same as solvers.Reconstruction, except it evaluates the wavefunctions
    on a minimal set of required indices. The sole exception is the proton $V_k$'s - since
    the Coulomb potential depends on the entire grid, we construct $V_{k,p}$ (and thereby $\rho_p$)
    everywhere. The compute_V subroutine then takes a subset of that indices once we're done with
    $\rho_p$.
    """
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
                    
                    self.Vgrid[ud].append(
                        np.swapaxes(
                            V[k][idx].T @ np.swapaxes(self.basis.psi[k][*wfInds],0,1),0,1))
        
        #Handles subset indices for collocation method
        if coordInds is not None:
            #Funky way of checking whether I'm dealing with $\rho_p$ or something else
            if len(coordInds[0]) != self.Vgrid['up'][0][0].size:
                for ud in ['up','down']:
                    for k in range(len(self.upList)):
                        self.Vgrid[ud][k] = self.Vgrid[ud][k][:,None,*coordInds]

        return
    
class ReducedHFB_Collocation(ReducedHFB):
    """
    Kind of a sloppy implementation for a lot of this. Assumes that all densities except $\rho_p$ are
    computed on the same mesh, although that restriction could (perhaps) be somewhat lifted with some care
    """
    def __init__(self, edf, basis, basisMatsDict, nParticles, r, bz, bp, laplaceMode='exact', earlyStop=False):
        super().__init__(edf, basis, basisMatsDict, nParticles, r, bz, bp, laplaceMode, earlyStop,
                         densitiesToConstruct=utils.VariableNames.defaultReconDensities)

    @utils.timer
    def reconstruct_densities(self, activeStates, U, V, reconstructionCoordInds, returnReconObj=False,
                              useFullInds=False,):
        densities = {pn:{} for pn in ['n','p']}
        reshapedInds = reconstructionCoordInds['n']['rho']

        if returnReconObj:
            objs = {}

        for pn in ['n','p']:
            if useFullInds:
                reconstruction = Reconstruction_Collocation(self.basis,self.rGridRepeated)
            else:
                #rGrid is only fed in to be used in expressions in tau/divJ/Drho, not in rho itself. So,
                #it doesn't matter that we recompute $\rho_p$ everywhere and rGrid doesn't match up, since
                #rGrid is irrelevant for that density
                reconstruction = Reconstruction_Collocation(self.basis,self.rGridRepeated[:,*reshapedInds])
            for key in self.densitiesToConstruct:
                densities[pn][key] = getattr(reconstruction,key)(activeStates[pn],U[pn],V[pn],
                                                                    coordInds=reconstructionCoordInds[pn][key])
                self.densities[pn][key] = densities[pn][key]
            if returnReconObj:
                objs[pn] = reconstruction
        
        if returnReconObj:
            return densities, objs
        else:
            return densities
    
    @utils.timer
    def reconstruct_fields(self,densities,reconstructionCoordInds):
        rhopFull = densities['p']['rho'].copy()
        densities['p']['rho'] = densities['p']['rho'][*reconstructionCoordInds['n']['rho']]
        
        newFields = {pn:{key:np.zeros(len(reconstructionCoordInds['n']['rho'][0])) 
                         for key in self.densitiesToConstruct} for pn in ['n','p']}

        for var in self.densitiesToConstruct:
            for (key,obj) in self.edf.terms.items():
                if key == 'coulomb_direct':
                    integArgs = [rhopFull,]
                else:
                    integArgs = [densities[tup[0]][tup[1]] for tup in self.edf.integArgs[key]]
                
                pField, nField = getattr(obj,var)(integArgs,self.bz,self.bp)

                if key == 'coulomb_direct':
                    pField = pField[*reconstructionCoordInds['n']['rho']]
                    nField = nField[*reconstructionCoordInds['n']['rho']]

                newFields['p'][var] += pField.flatten()
                newFields['n'][var] += nField.flatten()
        
        return newFields
    
    def single_iter_final(self,coeffs,chemPot,returnReconObjs=False):
        hfbArrDict = self.make_reduced_hfb_matrix(coeffs,chemPot)
        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)
        ret = self.reconstruct_densities(activeStates,Udict,Vdict,
                                               {pn:{key:None for key in coeffs[pn].keys()}
                                                    for pn in ['n','p']},useFullInds=True,
                                                    returnReconObj=returnReconObjs)
        return ret
    
class ReducedHFB_FancyDiag_Collocation(ReducedHFB_FancyDiag,ReducedHFB_Collocation):
    def __init__(self, edf, basis, basisMatsDict, hfbMatShapes, nParticles, r, bz, bp, nStatesToGet, wfBasisDict, laplaceMode='exact', earlyStop=False):
        super().__init__(edf, basis, basisMatsDict, hfbMatShapes, nParticles, r, bz, bp, nStatesToGet, wfBasisDict, laplaceMode, earlyStop)

class ReducedHFB_ReducedProtonDensity(ReducedHFB):
    def __init__(self, edf, basis, basisMatsDict, nParticles, r, bz, bp, 
                 densityBasisDict,
                 laplaceMode='exact', earlyStop=False,
                 densitiesToConstruct=utils.VariableNames.defaultReconDensities):
        super().__init__(edf, basis, basisMatsDict, nParticles, r, bz, bp, laplaceMode, earlyStop,
                         densitiesToConstruct)
        self.densityBasisDict = densityBasisDict

    @utils.timer
    def reconstruct_densities(self, activeStates, U, V, reconstructionCoordInds,
                              returnReconObj=False,useFullInds=False,):
        densities = {pn:{} for pn in ['n','p']}
        reshapedInds = reconstructionCoordInds['n']['rho']

        if returnReconObj:
            objs = {}

        for pn in ['n','p']:
            if useFullInds:
                reconstruction = Reconstruction_Collocation(self.basis,self.rGridRepeated)
            else:
                #rGrid is only fed in to be used in expressions in tau/divJ/Drho, not in rho itself. So,
                #it doesn't matter that we recompute $\rho_p$ everywhere and rGrid doesn't match up, since
                #rGrid is irrelevant for that density
                reconstruction = Reconstruction_Collocation(self.basis,self.rGridRepeated[:,*reshapedInds])
            for key in self.densitiesToConstruct:
                densities[pn][key] = getattr(reconstruction,key)(activeStates[pn],U[pn],V[pn],
                                                                 coordInds=reconstructionCoordInds[pn][key])
                self.densities[pn][key] = densities[pn][key]
            if returnReconObj:
                objs[pn] = reconstruction
        
        if returnReconObj:
            return densities, objs
        else:
            return densities
    
    @utils.timer
    def reconstruct_fields(self,densities,reconstructionCoordInds):
        # rhopFull = densities['p']['rho'].copy()
        # densities['p']['rho'] = densities['p']['rho'][*reconstructionCoordInds['n']['rho']]
        
        newFields = {pn:{key:np.zeros(len(reconstructionCoordInds['n']['rho'][0])) 
                         for key in self.densitiesToConstruct} for pn in ['n','p']}

        for var in self.densitiesToConstruct:
            for (key,obj) in self.edf.terms.items():
                integArgs = [densities[tup[0]][tup[1]] for tup in self.edf.integArgs[key]]
                pField, nField = getattr(obj,var)(integArgs,self.bz,self.bp)

                if key == 'coulomb_direct' and var == 'rho':
                    pField = pField[*reconstructionCoordInds['p']['rho']]
                    nField = nField[*reconstructionCoordInds['n']['rho']]

                newFields['p'][var] += pField.flatten()
                newFields['n'][var] += nField.flatten()
        
        return newFields
    
    @utils.timer
    # @profile
    def single_iter(self,coeffs,chemPot,oldDensities,
                    reconstructionCoordInds,pseudoInverseDict,tol):
        timeDict = {}
        
        #The last 2 arguments define the coordinates we reconstruct the fields
        #on. I'm not sure if it makes sense to let you change them while the
        #code is running, or if the class should be remade with those options
        #fixed
        t0 = time.time()
        hfbArrDict = self.make_reduced_hfb_matrix(coeffs,chemPot)
        t1 = time.time()
        timeDict['make_reduced_hfb_matrix'] = t1 - t0
        
        t0 = time.time()
        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        t1 = time.time()
        timeDict['diagonalize_hfb_matrix'] = t1 - t0

        t0 = time.time()
        activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)
        t1 = time.time()
        timeDict['pair_reg'] = t1 - t0

        t0 = time.time()
        densities = self.reconstruct_densities(activeStates,Udict,Vdict,
                                               reconstructionCoordInds)
        t1 = time.time()
        timeDict['reconstruct_densities'] = t1 - t0
        
        t0 = time.time()
        fields = self.reconstruct_fields(densities,reconstructionCoordInds)
        t1 = time.time()
        timeDict['reconstruct_fields'] = t1 - t0

        t0 = time.time()
        newCoeffs = self.get_new_coeffs(fields,pseudoInverseDict)
        t1 = time.time()
        timeDict['get_new_coeffs'] = t1 - t0
        
        #Seems to lower precision with negligible runtime improvement
        if self.earlyStop:
            for pn in ['n','p']:
                for (key,arr) in newCoeffs[pn].items():
                    if np.abs(arr - coeffs[pn][key]).max() <= tol:
                        self.needsReconstruction[pn][key] = False
        return newCoeffs, newChemPot, densities, timeDict
    
    def single_iter_final(self,coeffs,chemPot,returnReconObjs=False):
        hfbArrDict = self.make_reduced_hfb_matrix(coeffs,chemPot)
        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)
        ret = self.reconstruct_densities(activeStates,Udict,Vdict,
                                         {pn:{key:None for key in coeffs[pn].keys()}
                                              for pn in ['n','p']},
                                              useFullInds=True,
                                              returnReconObj=returnReconObjs)
        return ret

""" ===================== Lipkin-Nogami ===================== """
class ReducedLN(runfiles.LipkinNogami):
    def __init__(self,edf,basis,basisMatsDict,nParticles,
                 r,bz,bp,
                 densitiesToConstruct=utils.VariableNames.defaultReconDensities):
        self.edf = edf
        self.basis = basis
        self.needsReconstruction = {'n':True,'p':True}
        self.basisMatsDict = basisMatsDict
        self.pairRegObjs = {pn:solvers.PairingRegularization(basis) for pn in ['n','p']}
        self.nParticles = nParticles
        
        self.bz = bz
        self.bp = bp
        
        self.rGridRepeated = np.repeat(r[None,:,None],basis.psi[0].shape[-1],axis=-1)
        self.needsReconstruction = {pn:{key:True for key in basisMatsDict[pn].keys()} for pn in ['n','p']}
        self.densities = {pn:{} for pn in ['n','p']}

        self.densitiesToConstruct = densitiesToConstruct
    
    @utils.timer
    def make_reduced_hfb_matrix(self,coeffsIn,chemPot):
        hfbArrDict = {'p':[],'n':[]}
        for pn in ['n','p']:
            for k in range(self.basis.nBlocks):
                blockSize = self.basis.psi[k].shape[0]
                hfbArr = np.zeros(2*(2*blockSize,))
                
                for key in utils.VariableNames.phKeys:
                    hfbArr[:blockSize,:blockSize] += np.einsum('i,ijk->jk',coeffsIn[pn][key],
                                                               self.basisMatsDict[pn][key][k])
                for key in utils.VariableNames.ppKeys:
                    hfbArr[blockSize:,:blockSize] += np.einsum('i,ijk->jk',coeffsIn[pn][key],
                                                               self.basisMatsDict[pn][key][k])
                hfbArr[blockSize:,blockSize:] = -hfbArr[:blockSize,:blockSize]
                hfbArr[:blockSize,blockSize:] = hfbArr[blockSize:,:blockSize].T
                    
                hfbArr[:blockSize,:blockSize] -= chemPot[pn] * np.identity(blockSize)
                hfbArr[blockSize:,blockSize:] += chemPot[pn] * np.identity(blockSize)
                
                hfbArrDict[pn].append(hfbArr)
        return hfbArrDict

    @utils.timer
    # @profile
    def reconstruct_densities(self,activeStates,U,V,reconstructionCoordInds,returnReconObj=False):
        densities = {pn:{} for pn in ['n','p']}

        if returnReconObj:
            objs = {}

        for pn in ['n','p']:
            if reconstructionCoordInds is None:
                reconstruction = solvers.Reconstruction(self.basis,self.rGridRepeated)
            else:
                reconstruction = solvers.Reconstruction(self.basis,
                                                        self.rGridRepeated[:,*reconstructionCoordInds])
            for key in self.densitiesToConstruct:
                if self.needsReconstruction[pn][key]:
                    densities[pn][key] = getattr(reconstruction,key)(activeStates[pn],U[pn],V[pn],
                                                                     coordInds=reconstructionCoordInds)
                    self.densities[pn][key] = densities[pn][key]
                else:
                    densities[pn][key] = self.densities[pn][key]
                
            if returnReconObj:
                objs[pn] = reconstruction

        if returnReconObj:
            return densities, objs
        else:
            return densities

    @utils.timer
    def reconstruct_fields(self,densities,reconstructionCoordInds):
        #TODO: always reconstructs fields regardless of necessity
        newFields = {pn:{key:np.zeros(utils.GlobalVariables.meshShape) for key in self.densitiesToConstruct}
                      for pn in ['n','p']}
        
        for var in self.densitiesToConstruct:
            for (key,obj) in self.edf.terms.items():
                integArgs = [densities[tup[0]][tup[1]] for tup in self.edf.integArgs[key]]
                pField, nField = getattr(obj,var)(integArgs,self.bz,self.bp)
                
                if pField.shape == utils.GlobalVariables.meshShape:
                    newFields['p'][var] += pField
                else:
                    paddedArr = np.zeros(utils.GlobalVariables.meshShape)
                    paddedArr[*reconstructionCoordInds] = pField
                    newFields['p'][var] += paddedArr
                
                if nField.shape == utils.GlobalVariables.meshShape:
                    newFields['n'][var] += nField
                else:
                    paddedArr = np.zeros(utils.GlobalVariables.meshShape)
                    paddedArr[*reconstructionCoordInds] = nField
                    newFields['n'][var] += paddedArr
        
        #The kinetic variation has constant $\hbar^2/2m$ in it
        if reconstructionCoordInds is not None:
            for pn in ['n','p']:
                newFields[pn]['tau'][*reconstructionCoordInds] -= utils.GlobalVariables.h2m
                newFields[pn]['tau'] += utils.GlobalVariables.h2m
        
        return newFields

    @utils.timer
    def get_new_coeffs(self,fields,pseudoInverseDict):
        newCoeffsDict = {pn:{} for pn in ['n','p']}

        for pn in ['n','p']:
            for (key,field) in fields[pn].items():
                newCoeffsDict[pn][key] = pseudoInverseDict[pn][key] @ field.flatten()
                
        return newCoeffsDict

    @utils.timer
    def single_iter(self,coeffs,chemPot,oldDensities,
                    reconstructionCoordInds,pseudoInverseDict,
                    lmd2,rhoQP):
        #The last 2 arguments define the coordinates we reconstruct the fields
        #on. I'm not sure if it makes sense to let you change them while the
        #code is running, or if the class should be remade with those options
        #fixed

        timeDict = {}
        
        t0 = time.time()
        hfbArrDict = self.make_reduced_hfb_matrix(coeffs,chemPot)
        hfbArrDict = self.add_LN_to_hfb_matrix(hfbArrDict,rhoQP,lmd2)
        t1 = time.time()
        timeDict['make_reduced_hfb_matrix'] = t1 - t0
        
        t0 = time.time()
        Udict, Vdict, eqpDict = self.diagonalize_hfb_matrix(hfbArrDict)
        t1 = time.time()
        timeDict['diagonalize_hfb_matrix'] = t1 - t0

        t0 = time.time()
        activeStates, newChemPot = self.pair_reg(Vdict,eqpDict,chemPot)
        t1 = time.time()
        timeDict['pair_reg'] = t1 - t0

        t0 = time.time()
        lmd2, rhoQP = self.adjust_lipkin_nogami(activeStates,Udict,Vdict,hfbArrDict)
        t1 = time.time()
        timeDict['ln_adjust'] = t1 - t0

        t0 = time.time()
        densities = self.reconstruct_densities(activeStates,Udict,Vdict,
                                               reconstructionCoordInds)
        t1 = time.time()
        timeDict['reconstruct_densities'] = t1 - t0
        
        t0 = time.time()
        fields = self.reconstruct_fields(densities,reconstructionCoordInds)
        t1 = time.time()
        timeDict['reconstruct_fields'] = t1 - t0

        t0 = time.time()
        newCoeffs = self.get_new_coeffs(fields,pseudoInverseDict)
        t1 = time.time()
        timeDict['get_new_coeffs'] = t1 - t0
        
        return newCoeffs, newChemPot, densities, lmd2, rhoQP, timeDict