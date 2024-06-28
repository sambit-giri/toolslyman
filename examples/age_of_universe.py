import numpy as np 
import toolslyman

# Parameters 
param = toolslyman.param()
print('Cosmological parameters')
print(param.cosmo.__dict__)
print('Code parameters')
print(param.code.__dict__)

# Ages
zs = np.linspace(param.code.zmin,param.code.zmax,param.code.Nz)
t0 = toolslyman.age_estimator(param, zs)

# Plot
import matplotlib.pyplot as plt

fig, ax = plt.subplots(1,1,figsize=(5,4))
ax.plot(zs, t0)
ax.set_xlabel('Redshift')
ax.set_ylabel('Age')
plt.show()
