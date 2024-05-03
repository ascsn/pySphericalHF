from init import *
import densities 
import fields 
import solvers 
import wf
import functionals
import utilities
import matplotlib.pyplot as plt
print(nbox)
movie_dir = '../movie_files/'
nuc = '16O'
### Initialize the single particle wavefunctions

psi_array,energies = wf.initWfs(name='HO',shift=2)

### Initialize the Densities (\rho)
rhoArr = densities.rho(psi_array)
 

### Initialize the coulomb field
V_c = fields.coulombArr(rhoArr[1])


### Intialize the mean field Hamiltonian h with initial wavefunctions 
hArr = np.zeros(len(grid)) # the hamiltonian
hArr = functionals.h_BKN(rhoArr[0])

#plt.plot(grid,hArr,label='initial h')
#plt.xlabel('r')
#plt.ylabel(r'$h$')
#plt.legend()
#plt.show() 

## Construct initial g_array. Every wavefunction will have one of these functions
## associated with it. 
fArr = np.zeros((2,nmax+1,lmax+1,len(spin),len(grid)))
### 
D = solvers.getNumerov_matrix()
#E = -20
#dE = .5
#Rch_array = []
#rhoArr_iter = np.zeros((nIter,len(grid)))
#rho_diff = []
for nter in range(0,nIter):
    print(nter)
    plt.plot(grid,rhoArr[0],label='total')
    plt.plot(grid,rhoArr[1],label='proton')
    plt.plot(grid,rhoArr[2],label='neutron')
    plt.xlabel('r')
    plt.ylabel(r'$\rho$')
    #plt.xlim([0,8])
    plt.title(f'Iteration: {nter}')
    plt.legend()
    #pad_iter = str(nter).zfill(3)
    #plt.savefig(movie_dir+f'density_frames/{nuc}_bounded/bounded_{pad_iter}.png')
    plt.show()
    for q in range(0,2):
        for n in range(nmax+1):
            for l in range(lmax+1):
                for s in range(len(spin)):
                    j = l + spin[s]
                    print(q,n,l,j)
                    if q == 0:
                        fArr[q][n][l][s] = (hArr + V_c) + fields.centriforceArr(l) 
                    else: 
                        fArr[q][n][l][s] = hArr + fields.centriforceArr(l)
                    
                    V_matrix = np.diag(fArr[q][n][l][s])
                    H = -hb2m0*D + V_matrix
                    #psi = psi_array[q][n][l][s]
                    #energies[q][n][l][s],psi_array[q][n][l][s] = solvers.solve_Numerov(psi,E,dE,fArr[q][n][l][s])
                    
                    energies[q][n][l][s], psi_array[q][n][l][s] = solvers.MatrixNumerovSolve(H)
                    print(energies[q][n][l][s])
                    #plt.plot(grid,fArr[q][n][l][s])
                    #plt.title(f'{nter},{q},{n},{l},{j}')
                    #plt.xlim([0,7])
                    #plt.show()
    rhoArr = densities.rho(psi_array)
    #rhoArr_iter[nter] = rhoArr[0]
    V_c = fields.coulombArr(rhoArr[1])
    hArr = functionals.h_BKN(rhoArr[0]) 
    npro,nneu = utilities.getNZ(rhoArr)
    print(f' Integrated: N = {nneu}, Z = {npro}')
    Rp,Rn,Rch = utilities.getRadi(rhoArr)
    print(f'Neutron Radius: {Rn}')
    print(f'Proton Radius: {Rp}')
    print(f'Charge Radius: {Rch}')
    plt.plot(grid,hArr)
    plt.title('harr')
    plt.xlim([0,7])
    plt.show()
    
                            
    #rho_diff.append(np.linalg.norm(rhoArr_iter[nter -1] - rhoArr_iter[nter]))
                  
    
'''
plt.plot(np.arange(0,nIter),rho_diff,label='total')
plt.xlabel('Iteration')
plt.ylabel(r'$|\rho^{n} - \rho^{(n-1)}|_{2}$')
plt.title(r'$\rho$ Convergence')
plt.legend()
plt.savefig(movie_dir+f'density_frames/{nuc}_bounded_convergence.pdf')
plt.show()    


plt.plot(np.arange(0,nIter),Rch_array,'o')
plt.hlines(Rch_array[-1],0,nIter,label=f'Rch = {Rch_array[-1]}',color='red')
plt.legend()
plt.xlabel('Iteration')
plt.ylabel(r'$R_{ch}$ (fm)')
plt.savefig('Rch_{nuc}-BKN.pdf')
plt.show()

plt.plot(grid[1:],rhoArr[0][1:])
plt.xlabel('r')
plt.ylabel(r'$\rho$')
#plt.xlim([0,8])
plt.title(f'Iteration: Final')
plt.show()             

rho_data = np.stack((grid,rhoArr[0]),axis=-1)
rho_diff_data = np.stack((np.arange(0,nIter),rho_diff),axis=-1)
np.savetxt(f'rho_data_{step_size}_{nuc}.txt',rho_data, delimiter=',')
np.savetxt(f'rho_diff_data_{step_size}_{nuc}.txt',rho_diff_data, delimiter=',')
'''
