import os
import kaldiio
import matplotlib.pyplot as plt
import numpy as np

with open('/home/mayi/project/wespeaker/examples/voxceleb/v2/exp/2pooling_constrainMax_deSmp/scores/vox1_E_cleaned.kaldi.score','r') as file:
    lines = file.readlines()
target_rows, non_target_rows = [],[]
for line in lines:
    utt1,utt2,score,label = line.strip().split(' ')
    if label=='target':
        target_rows.append(float(score))
    else:
        non_target_rows.append(float(score))
target_rows = np.array(sorted(target_rows))
non_target_rows = np.array(sorted(non_target_rows))
plt.hist(target_rows,bins=100,color='blue',alpha=0.5,label='target')
plt.hist(non_target_rows,bins=100,color='orange',alpha=0.5,label='non target')
plt.legend()
plt.xlabel('score')
plt.ylabel('Frequency')
plt.savefig('score.png')

