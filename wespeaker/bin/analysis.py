import os
import kaldiio
#import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire
from sklearn.metrics.pairwise import cosine_similarity
def split_sort_score(score_dir):
    with open(score_dir,'r') as file:
        lines = file.readlines()
    target_rows, non_target_rows = [],[]
    for line in lines:
        utt1,utt2,score,label = line.strip().split(' ')
        if label=='target':
            target_rows.append([utt1,utt2,float(score),label])
        else:
            non_target_rows.append([utt1,utt2,float(score),label])
    target_rows = sorted(target_rows,key=lambda x:x[2],reverse=True)
    non_target_rows = sorted(non_target_rows,key=lambda x:x[2],reverse=True)
    return target_rows,non_target_rows
    
def find_index(result_array,dict_raw,top=20,find_largest=True):

    if find_largest:
        indices = np.argpartition(result_array, 0-top, axis=None)[0-top:]
    else:
        indices = np.argpartition(0-result_array, 0-top, axis=None)[0-top:]
    indices = np.unravel_index(indices, result_array.shape)
    max_values = result_array[indices]

    if find_largest:
        sorted_indices = np.argsort(-max_values)
    else:
        sorted_indices = np.argsort(max_values)

    sorted_max_values = max_values[sorted_indices]
    #sorted_max_indices = (indices[0][sorted_indices], indices[1][sorted_indices])
    sorted_max_indices = (indices[0][sorted_indices])

    result = {}
    for i in range(top):
        #pho_1 = dict_raw[sorted_max_indices[0][i]]
        pho_1 = dict_raw[sorted_max_indices[i]]
        pho_2=pho_1
        #pho_2 = dict_raw[sorted_max_indices[0][i]]
        comb_1 = pho_1+'-'+pho_2
        comb_2 = pho_2+'-'+pho_1
        if comb_1 in result or comb_2 in result:
            continue
        result[comb_1]=sorted_max_values[i]
        if len(result)==int(top/2):
            break
    return result

def plot(emb_dict,dict_raw,cond,title,score_dir,top,find_largest,target_num):

    result_array = np.zeros(40)
    non_zero = np.zeros(40)
    for i in cond:
        emb1 = emb_dict[i[0]]
        emb2 = emb_dict[i[1]]
        cosine_similarity = np.matmul(emb1,emb2.T)
        #single = (cosine_similarity+cosine_similarity.T)/2
        single = np.diagonal(cosine_similarity)
        single = np.where(non_zero<target_num,single,0)
        result_array += single
        non_zero += np.where(single!= 0, 1, 0)
    #print(result_array)
    print(non_zero)
    
    result = result_array/(non_zero+1e-6)
    #print(result)
    print('============')
    plot_fig(dict_raw,result,title,score_dir,top,find_largest)
    return result

def plot_fig(dict_raw,result,title,score_dir,top,find_largest):
    result = find_index(result,dict_raw,top=2*top,find_largest=find_largest)
    keys = list(result.keys())
    values = list(result.values())
    plt.bar(keys, values)
    plt.title(title)
    plt.xlabel('Similarity')
    plt.ylabel('Phoneme')
    plt.xticks(rotation=45)
    plt.savefig(os.path.join(score_dir,title+'.png'))
    plt.close()

def seperate(eval_scp_path='',
                        score_dir='',
                        trials=(),
                        ratio=1,
                        top = 20):
    dict_raw = dict()
    line_idx = 0
    with open('pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[line_idx] = phoneme
            line_idx += 1

    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb_dict[utt] = np.transpose(emb)
    
    for trial in trials:
        name = trial.split('/')[-1]
        store_path = os.path.join(score_dir,
                                  os.path.basename(trial) + '.score')
        target_rows, non_target_rows = split_sort_score(store_path)
        #num = int(ratio*len(target_rows))
        
        #true_target = target_rows[:num]
        #false_target = target_rows[-num:]
        #true_nontarget = non_target_rows[-num:]
        #false_nontarget = non_target_rows[:num]
        indices = np.random.permutation(np.arange(len(target_rows)))
        true_target = [target_rows[i] for i in indices]

        indices = np.random.permutation(np.arange(len(non_target_rows)))
        true_nontarget = [non_target_rows[i] for i in indices]

        target_num = 500
        true_target = plot(emb_dict,dict_raw,true_target,name+'true_target',score_dir,top,True,target_num)
        #false_target = plot(emb_dict,dict_raw,false_target,'false_target',score_dir,top,True)
        true_nontarget = plot(emb_dict,dict_raw,true_nontarget,name+'true_nontarget',score_dir,top,False,target_num)
        #false_nontarget = plot(emb_dict,dict_raw,false_nontarget,'false_nontarget',score_dir,top,False)

        true_diff = true_target-true_nontarget
   
   
        plot_fig(dict_raw,true_diff,name+str(ratio)+' true difference',score_dir,top,True) 
        np.save(os.path.join(score_dir,name+str(ratio)+'diff_pho'), true_diff)

        true_ratio = true_target/true_nontarget
        #true_diff = true_target/(true_nontarget+1e-6)
        plot_fig(dict_raw,true_ratio,name+str(ratio)+' true ratio',score_dir,top,True) 
        np.save(os.path.join(score_dir,name+str(ratio)+'ratio_pho'), true_ratio)       

        #false_diff = false_target-false_nontarget
        #plot_fig(dict_raw,false_diff,str(ratio)+' overlap difference',score_dir,top,True) 
     
        return true_target,  true_nontarget

def main(exp_dir,
         eval_scp_path,
         *trials):
    store_score_dir = os.path.join(exp_dir, 'scores')
    true_target, true_nontarget = seperate(eval_scp_path, store_score_dir, trials)

    
if __name__ == "__main__":
    fire.Fire(main)