import seaborn as sns
import matplotlib.pyplot as plt
import numpy as np
path = 'exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001_selfcstr0.0001/target.npy'
weight = np.load(path)
#weight = np.diagonal(weight)
#min=weight.min()
#print(min)
#max=weight.max()
#print(max)
#weight = (weight-min)/(max-min)
#print(weight)
sns.heatmap(weight,annot=False,vmin=0,vmax=1)
plt.savefig(path.replace('.npy','.png'))
plt.close()
