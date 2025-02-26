import numpy as np
import utils

class LinearMixing:
    __name__ = "LinearMixing"
    
    def __init__(self,Vin,maxIter,alpha):
        self.alpha = alpha
        
        self.Vold = Vin.copy()
        self.resMax = np.zeros(maxIter)
        self.ctr = 0
        
    def __call__(self,Vout):
        self.resMax[self.ctr] = np.abs(Vout-self.Vold).max()

        Vnew = self.alpha*Vout + (1-self.alpha)*self.Vold
        
        self.Vold = Vnew
        self.ctr += 1
        return Vnew
    
class AcceleratedLinearMixing:
    __name__ = "AcceleratedLinearMixing"
    
    def __init__(self,Vin,maxIter,alpha,alphaMax=1.0,alphaAccel=1.13):
        self.alpha = alpha
        
        self.alphaInit = alpha
        self.alphaMax = alphaMax
        self.alphaAccel = alphaAccel
        
        self.Vold = Vin.copy()
        
        self.resMax = np.zeros(maxIter)
        self.alphaArr = np.zeros(maxIter)

        self.ctr = 0

        self.paramsDict = {'name':AcceleratedLinearMixing.__name__,
                           'alphaInit':self.alphaInit,
                           'alphaMax':alphaMax,
                           'alphaAccel':alphaAccel,
                           'maxIter':maxIter
                           }

    def __call__(self,Vout):
        self.resMax[self.ctr] = np.abs(Vout-self.Vold).max()
        self.alphaArr[self.ctr] = self.alpha
        
        if self.ctr > 0:
            if self.resMax[self.ctr] <= self.resMax[self.ctr-1]:
                self.alpha = min(self.alphaMax,self.alphaAccel*self.alpha)
            else:
                self.alpha = self.alphaInit

        Vmix = self.alpha*Vout + (1-self.alpha)*self.Vold

        self.Vold = Vmix
        self.ctr += 1
        return Vmix

class Broyden:
    __name__ = "Broyden"
    
    def __init__(self,Vin,maxIter,alpha,w0=0.01,wn=1,M=7):
        self.alpha = alpha
        
        self.nVars = len(Vin)
        self.w0 = w0
        self.wn = wn
        self.M = M
        
        self.ctr = 0
        
        self.F = np.zeros((maxIter,self.nVars))
        self.Vin = np.zeros((maxIter,self.nVars))
        self.resMax = np.zeros(maxIter)
        self.alphaArr = alpha*np.ones(maxIter)

        self.paramsDict = {'name':Broyden.__name__,
                           'alpha':self.alpha,
                           'w0':w0,
                           'wn':wn,
                           'M':M,
                           'maxIter':maxIter
                           }
        
        self.Vin[0] = Vin

    def __call__(self,Vin,Vout):
        self.Vin[self.ctr] = Vin
        self.F[self.ctr] = Vout - Vin
        self.resMax[self.ctr] = np.max(np.abs(self.F[self.ctr]))

        if self.ctr == 0:
            self.ctr += 1
            mixType = 'L'
            return Vin + self.alpha*self.F[0], mixType
        
        #Furthest back iteration that will be used. Is $\tilde{m}$ in my notes
        sumStartIter = max(0,self.ctr - self.M)
        rng = range(sumStartIter,self.ctr)
        nStep = len(rng)
        
        dF = np.zeros((nStep,self.nVars))
        dV = np.zeros((nStep,self.nVars))
        u = np.zeros((nStep,self.nVars))
        
        for (nIter,n) in enumerate(rng):
            dF[nIter] = (self.F[n+1] - self.F[n])/np.linalg.norm(self.F[n+1] - self.F[n])
            dV[nIter] = (self.Vin[n+1] - self.Vin[n])/np.linalg.norm(self.F[n+1] - self.F[n])
            u[nIter] = self.alpha * dF[nIter] + dV[nIter]
            
        a = np.zeros((nStep,nStep))
        for (nIter,n) in enumerate(rng):
            for (kIter,k) in enumerate(rng):
                a[kIter,nIter] = self.wn**2 * dF[kIter] @ dF[nIter]
                
        #What I think is correct, vs what HFBTHO does
        # beta = np.linalg.inv(self.w0**2*np.identity(nStep)+a)
        beta = np.linalg.inv((1+self.w0**2)*np.identity(nStep)+a)
        
        #Seems to match HFBTHO
        c = np.zeros(nStep)
        for (kIter,k) in enumerate(rng):
            c[kIter] = self.wn * dF[kIter] @ self.F[self.ctr]
            
        gamma = np.zeros(nStep)
        for (nIter,n) in enumerate(rng):
            for (kIter,k) in enumerate(rng):
                gamma[nIter] += c[kIter]*beta[kIter,nIter]
        
        Vbroyden = self.alpha*self.F[self.ctr]
        for (nIter,n) in enumerate(rng):
            Vbroyden -= self.wn*gamma[nIter]*u[nIter]
            
        curvature = Vbroyden @ self.F[self.ctr]
        if curvature > -1:
            mixType = 'B'
            ret = Vin + Vbroyden
        else:
            mixType = 'L'
            ret = Vin + 0.5*self.alpha*self.F[self.ctr]
        
        self.ctr += 1
        
        #Since we increased self.ctr, get self.ctr rather than self.ctr+1
        return ret, mixType
    
class AcceleratedBroyden:
    def __init__(self,Vin,maxIter,alpha,w0=0.01,wn=1,M=7,alphaMax=1.0,alphaAccel=1.13):
        raise ValueError
        self.alpha = alpha
        
        self.alphaInit = alpha
        self.alphaMax = alphaMax
        self.alphaAccel = alphaAccel
        
        self.nVars = len(Vin)
        self.w0 = 0.01
        self.wn = wn
        self.M = M
        
        self.ctr = 0
        
        self.F = np.zeros((maxIter,self.nVars))
        self.Vin = np.zeros((maxIter,self.nVars))
        self.resMax = np.zeros(maxIter)
        
        self.Vin[0] = Vin
        
    def __call__(self,Vout):
        mTilde = max(0,self.ctr - self.M)
        
        Vin = self.Vin[self.ctr]
        self.F[self.ctr] = Vout - Vin
        
        rng = range(mTilde,self.ctr-1)
        
        nStep = len(rng)
        
        dF = np.zeros((nStep,self.nVars))
        dV = np.zeros((nStep,self.nVars))
        u = np.zeros((nStep,self.nVars))
        
        for (nIter,n) in enumerate(rng):
            dF[nIter] = (self.F[n+1] - self.F[n])/np.linalg.norm(self.F[n+1] - self.F[n])
            dV[nIter] = (self.Vin[n+1] - self.Vin[n])/np.linalg.norm(self.F[n+1] - self.F[n])
            u[nIter] = self.alpha * dF[nIter] + dV[nIter]
            
        a = np.zeros((nStep,nStep))
        for (nIter,n) in enumerate(rng):
            for (kIter,k) in enumerate(rng):
                a[kIter,nIter] = self.wn**2 * dF[kIter] @ dF[nIter]
                
        beta = np.linalg.inv(self.w0*np.identity(nStep)+a)
                
        c = np.zeros(nStep)
        for (kIter,k) in enumerate(rng):
            c[kIter] = self.wn * dF[kIter] @ self.F[self.ctr]
            
        gamma = np.zeros(nStep)
        for (nIter,n) in enumerate(rng):
            for (kIter,k) in enumerate(rng):
                gamma[nIter] += c[kIter]*beta[kIter,nIter]
        
        #If residual is worse, falls back to linear mixing
        Vlinear = self.Vin[self.ctr] + self.alpha*self.F[self.ctr]
        
        Vbroyden = Vlinear.copy()
        for (nIter,n) in enumerate(rng):
            Vbroyden -= self.wn*gamma[nIter]*u[nIter]
            
        self.resMax[self.ctr] = np.abs(Vbroyden - self.Vin[self.ctr]).max()
        
        if self.ctr > 0:
            if self.resMax[self.ctr] <= self.resMax[self.ctr-1]:
                self.alpha = min(self.alphaMax,self.alphaAccel*self.alpha)
            else:
                self.alpha = self.alphaInit
        
            if self.resMax[self.ctr] > self.resMax[self.ctr-1]:# and self.ctr > 50:
                self.Vin[self.ctr+1] = Vlinear
                print('Linear')
            else:
                self.Vin[self.ctr+1] = Vbroyden
                print('Broyden')
        # self.Vin[self.ctr+1] = Vbroyden
        
        self.ctr += 1
        
        #Since we increased self.ctr, get self.ctr rather than self.ctr+1
        return self.Vin[self.ctr]
    
class VariableFlattening:
    def __init__(self,varsDict):
        self.rngDict = {pn:{} for pn in ['n','p']}
        self.shps = {pn:{} for pn in ['n','p']}
        self.keysOrder = {pn:[] for pn in ['n','p']}

        ctr = 0
        for pn in ['n','p']:
            for (key,arr) in varsDict[pn].items():
                self.rngDict[pn][key] = slice(ctr,ctr+arr.size)
                self.shps[pn][key] = arr.shape
                ctr += arr.size
                self.keysOrder[pn].append(key)
    
    def flatten(self,varsDict):
        flatVars = []

        for pn in ['n','p']:
            for key in self.keysOrder[pn]:
                flatVars += list(varsDict[pn][key].flatten().copy())
        return np.array(flatVars)
    
    def unflatten(self,flatVars):
        varsDict = {pn:{} for pn in ['n','p']}
        for pn in ['n','p']:
            for (key,slc) in self.rngDict[pn].items():
                varsDict[pn][key] = flatVars[slc].reshape(self.shps[pn][key])
        return varsDict
    
class HFBMatrixFlattening:
    def __init__(self,basis):
        self.basis = basis

        self.inds = [np.triu_indices(len(self.basis.quantNumbersByBlock[k]))
                     for k in range(self.basis.nBlocks)]
        self.rngs = {pn:{'h':[],'hTilde':[]} for pn in ['n','p']}

    def flatten(self,hfbArr):
        flatVars = []

        ctr = 0
        for pn in ['n','p']:
            for k in range(self.basis.nBlocks):
                nStates = len(self.basis.quantNumbersByBlock[k])
                h = hfbArr[pn][k][:nStates,:nStates]
                hTilde = hfbArr[pn][k][:nStates,nStates:]

                hFlat = h[self.inds[k]]
                self.rngs[pn]['h'].append(slice(ctr,ctr+hFlat.size))
                ctr += hFlat.size
                flatVars += list(hFlat)

                hTildeFlat = hTilde[self.inds[k]]
                self.rngs[pn]['hTilde'].append(slice(ctr,ctr+hTildeFlat.size))
                ctr += hTildeFlat.size
                flatVars += list(hTildeFlat)
        return np.array(flatVars)
    
    def unflatten(self,flatVars):
        hfbArrDict = {pn:[] for pn in ['n','p']}
        for pn in ['n','p']:
            for k in range(self.basis.nBlocks):
                nStates = len(self.basis.quantNumbersByBlock[k])
                hfbArr = np.zeros(2*(2*nStates,))

                hPad = np.zeros(2*(nStates,))
                h = flatVars[self.rngs[pn]['h'][k]]
                hPad[self.inds[k]] = h
                hPad = utils.symmetrize_array_u(hPad)
                hfbArr[:nStates,:nStates] = hPad
                hfbArr[nStates:,nStates:] = -hPad

                htPad = np.zeros(2*(nStates,))
                ht = flatVars[self.rngs[pn]['hTilde'][k]]
                htPad[self.inds[k]] = ht
                htPad = utils.symmetrize_array_u(htPad)
                hfbArr[nStates:,:nStates] = htPad
                hfbArr[:nStates,nStates:] = htPad.T

                hfbArrDict[pn].append(hfbArr)
        return hfbArrDict
                

        