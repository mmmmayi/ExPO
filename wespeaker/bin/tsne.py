import os
import kaldiio
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire
from sklearn.manifold import TSNE
import random   
def check_pho(exp_dir,
              eval_scp_path,
              num=10):
    name=[]
    np.random.seed(4)
    for i in exp_dir:
        name.append(i.split('/')[-1])
    with open(eval_scp_path[0]) as scp_file:
        lines = scp_file.readlines()
    idx = np.random.choice(len(lines),10,replace=False)

    
    sampled_keys = [lines[i].split(' ')[0].strip() for i in idx]
    print(sampled_keys)
    all_data = np.zeros((num*len(exp_dir),1536,40))
    
    i=0
    for n in range(len(name)):
        dict = kaldiio.load_scp(eval_scp_path[n])
        for k in range(num):
            all_data[k+i*num]=dict[sampled_keys[k]]
        i+=1
    tsne=TSNE(n_components=2,random_state=42)
    for i in range(40):
        ith_pho=all_data[:,:,i].reshape(-1,1536)
        tsne_result = tsne.fit_transform(ith_pho)
        plt.figure(figsize=(10,8))
        for j in range(len(name)):
            
            data = tsne_result[j*num:(j+1)*num,:]
            plt.scatter(data[:,0],data[:,1],label=name[j])
        plt.legend()
        plt.savefig('tsne_pho/'+str(i)+'.png')
        plt.close()

        

def check_utt(exp_dir,
              eval_scp_path,
              num=10):
    name=[]
    for i in exp_dir:
        name.append(i.split('/')[-1])
       
    emb_dict={}
    with open(eval_scp_path[0]) as scp_file:
        lines = scp_file.readlines()
    idx = np.random.choice(len(lines),10,replace=False)
    
    sampled_keys = [lines[i].split(' ')[0].strip() for i in idx]
 
    for i in range(len(exp_dir)):
        dict = kaldiio.load_scp(eval_scp_path[i])
        for utt in sampled_keys:
        
            emb_dict[name[i]+utt] = dict[utt]
    idx=0
    tsne=TSNE(n_components=2,random_state=42)
    for utt in sampled_keys:
        temp = []
        for i in range(len(exp_dir)):
            pho = emb_dict[name[i]+utt]
            temp.append(tsne.fit_transform(pho.T))
        plt.figure(figsize=(10,8))
        for i in range(len(temp)):
            plt.scatter(temp[i][:,0],temp[i][:,1],label=name[i])
        plt.legend()
        plt.savefig('tsne/'+utt.replace('.wav','.png').replace('/','-'))
        plt.close()
        idx+=1
        if idx>num:
            break


def main(exp_dir):
    exp_dir=exp_dir.split('+')
    eval_scp_dir = []
    for i in exp_dir:
        eval_scp_dir.append(os.path.join(i, 'embeddings','vox1_O','phovector_0s.scp'))
    check_utt(exp_dir,eval_scp_dir)

if __name__ == "__main__":
    fire.Fire(main)

