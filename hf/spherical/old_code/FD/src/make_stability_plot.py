import numpy as np
import glob
import matplotlib.pyplot as plt

data_files = sorted(glob.glob('rho_data*4He.txt'))[::-1]
print(data_files)

colors = ['red','green','black']
step_sizes = ['1','.1','.02']
for i,case in enumerate(data_files):
    data = np.loadtxt(case,delimiter=',')
    plt.plot(data[:,0],data[:,1],color=colors[i],label='h = '+step_sizes[i])
plt.legend()
plt.xlim([0,5])
plt.xlabel('r (fm)')
plt.ylabel(r'$\rho$ (fm$^{-3}$)')
plt.savefig('4He_stability.pdf')

plt.show()