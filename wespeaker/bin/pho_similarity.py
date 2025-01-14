import os
import kaldiio
#import seaborn as sns
import matplotlib.pyplot as plt
from tqdm import tqdm
import numpy as np
from pathlib import Path
import fire
from sklearn.metrics.pairwise import cosine_similarity
import librosa
import librosa.display
import torch,torchaudio,os
from torch.nn.functional import conv1d, pad
def plot(dur, enrol,test,dict_raw,save_path,sub):
    enrol_wav='/hpctmp/ma_yi/'+sub+'/'+enrol
    enrol_pho=enrol_wav.replace('.wav','.pre.pho')
    spec_e,change_e,value_e=data(dur,enrol_wav,enrol_pho,dict_raw)

    test_wav='/hpctmp/ma_yi/'+sub+'/'+test
    test_pho=test_wav.replace('.wav','.pre.pho')
    spec_t,change_t,value_t=data(dur,test_wav,test_pho,dict_raw)

    fig,ax=plt.subplots(nrows=2,ncols=1,sharex=True)
    librosa.display.specshow(spec_e.cpu().squeeze().numpy(),sr=16000,ax=ax[0])
    for i in range(len(change_e)):
        if i==0:
            location=change_e[i].item()/2
        else:
            location=(change_e[i].item()-change_e[i-1].item())/2+change_e[i-1].item()



        ax[0].axvline(x=change_e[i].item(),color='r',linestyle='--')
        ax[0].text(location,ax[0].get_ylim()[1]*1.01,str(int(value_e[i].item())),fontsize=7,horizontalalignment='center')

    librosa.display.specshow(spec_t.cpu().squeeze().numpy(),sr=16000,ax=ax[1])
    for i in range(len(change_t)):
        if i==0:
            location=change_t[i].item()/2
        else:
            location=(change_t[i].item()-change_t[i-1].item())/2+change_t[i-1].item()



        ax[1].axvline(x=change_t[i].item(),color='r',linestyle='--')
        ax[1].text(location,ax[1].get_ylim()[1]*1.01,str(int(value_t[i].item())),fontsize=7,horizontalalignment='center')

    plt.tight_layout()
    plt.savefig(os.path.join(save_path,enrol.strip('.wav').replace('/','_')+test.strip('.wav').replace('/','_')+'.pho.png'))
    plt.close()

def data(dur,path,pho_path,dict_raw):
    waveform, sample_rate = torchaudio.load(path)
    
    if waveform.shape[-1]>dur*16000:
        waveform=waveform[:,:dur*16000]
    pho = load_file(waveform.shape,pho_path,dict_raw)
    win_eye = torch.eye(400).unsqueeze(1)
    pho = conv1d(pho.float(), win_eye, stride=160).squeeze()
    pho, _ = torch.mode(pho, dim=0)
    change = torch.diff(pho)
    change = (change!=0).nonzero(as_tuple=True)[0]
    value=pho[change-1]

    Spec = torchaudio.transforms.Spectrogram(n_fft=400, win_length=400, hop_length=160, pad=0, window_fn=torch.hamming_window, power=2.0, center=False)
    Mel_scale = torchaudio.transforms.MelScale(80,16000,20,7600,400//2+1)
    spec = Mel_scale((Spec(waveform)+1e-8)+1e-8).log()
    return spec,change,value

def load_file(size,pho_path,dict):
        weight = _phn2vec(size[1], pho_path, dict)
        
        label = torch.from_numpy(weight).clone().unsqueeze(0).unsqueeze(0)

        return label 

def _phn2vec(wav_length, phn_path, phn_dict):
    #phone_time_seq = np.zeros(wav_length)
    if not os.path.exists(phn_path):
        print(phn_path)
        return np.ones(wav_length)
    phone_seq = []
    phone_weight = np.zeros(wav_length)
    with open(str(phn_path)) as f:
        for line in f:
            sample_start, sample_end, label_char = line.split(' ')
            sample_start = int(float(sample_start)*16000)
            sample_end = int(float(sample_end)*16000)
            if sample_start>wav_length:
                break
            sample_end = wav_length if sample_end>wav_length else sample_end
            label_char = label_char.strip()
            if label_char=='[UNK]':
                label_char = '[SIL]'

            phone_weight[sample_start: sample_end] = phn_dict[label_char]
            
    return phone_weight

def split_sort_score(score_dir):
    with open(score_dir,'r') as file:
        lines = file.readlines()
    target_rows, non_target_rows = [],[]
    for line in lines:
        utt1,utt2,score,label = line.strip().split(' ')
        if utt1==utt2:
            continue
        if label=='target':
            target_rows.append([utt1,utt2,float(score),label])
        else:
            non_target_rows.append([utt1,utt2,float(score),label])
    target_rows = sorted(target_rows,key=lambda x:x[2],reverse=True)
    non_target_rows = sorted(non_target_rows,key=lambda x:x[2],reverse=True)
    return target_rows,non_target_rows
    
def trials_cosine_score(eval_scp_path='',
                        score_dir='',
                        trials=(),
                        sub='librispeech',
                        num=100):
    line_idx = 0
    dict_raw = dict()
    with open('pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[phoneme] = line_idx
            line_idx += 1
    emb_dict = {}
    for utt, emb in kaldiio.load_scp_sequential(eval_scp_path):
        emb_dict[utt] = np.transpose(emb)
    
    for trial in trials:
        store_path = os.path.join(score_dir,
                                  os.path.basename(trial) + '.score')
        target_rows, non_target_rows = split_sort_score(store_path)
    
        true_target = target_rows[:num]
        false_target = target_rows[-num:]
        true_nontarget = non_target_rows[-num:]
        false_nontarget = non_target_rows[:num]
        '''
        for i in true_target:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            
            #print(cosine_similarity)
            #sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'true_target')).mkdir(parents=True, exist_ok=True)
            #plt.savefig(os.path.join(score_dir,'true_target',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            #plt.close()
            plot(3,i[0],i[1],dict_raw,os.path.join(score_dir,'true_target'),sub)
        
        for i in false_target:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            
            sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'false_target')).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(score_dir,'false_target',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            plt.close()
        

        for i in true_nontarget:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            #sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'true_nontarget')).mkdir(parents=True, exist_ok=True)
            #plt.savefig(os.path.join(score_dir,'true_nontarget',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            #plt.close()
            plot(3,i[0],i[1],dict_raw,os.path.join(score_dir,'true_nontarget'),sub)
        
        for i in false_nontarget:
            emb1 = emb_dict[i[0]]
            emb2 = emb_dict[i[1]]
            cosine_similarity = np.matmul(emb1,emb2.T)
            sns.heatmap(cosine_similarity,annot=False,vmin=0,vmax=1)
            Path(os.path.join(score_dir,'false_nontarget')).mkdir(parents=True, exist_ok=True)
            plt.savefig(os.path.join(score_dir,'false_nontarget',i[0].strip('.wav').replace('/','_')+i[1].strip('.wav').replace('/','_')+'.png'))
            plt.close()
        '''
    for trial in trials:
        store_path = os.path.join(score_dir, os.path.basename(trial) + '.pho_score')
        with open(trial, 'r') as trial_r, open(store_path, 'w') as w_f:
            lines = trial_r.readlines()
            for line in tqdm(lines,desc='scoring trial {}'.format(os.path.basename(trial))):
                segs = line.strip().split()
                emb1, emb2 = emb_dict[segs[0]], emb_dict[segs[1]]
                cosine_similarity = np.matmul(emb1,emb2.T)
                #cosine_score = cosine_similarity(emb1,emb2)
                #print(cosine_score.shape)
                
                #idx = np.count_nonzero(cosine_score)
                #print(idx)
                #quit()
                #idx_diag = len(np.where(np.diagonal(cosine_similarity)!=0)[0])
                #score = np.sum(np.diagonal(cosine_similarity))/(idx_diag+1e-6)
                idx_full = len(np.where(cosine_similarity!=0)[0])
                score = np.sum(cosine_similarity)/idx_full
                #score = (score_full-score_diag)/(idx_full-idx_diag)
             

                w_f.write('{} {} {:.5f} {}\n'.format(segs[0], segs[1], score, segs[2]))
    
                

def main(exp_dir,
         eval_scp_path,
         *trials):
    sub = eval_scp_path.split('/')[-2]
    store_score_dir = os.path.join(exp_dir, 'scores')
    trials_cosine_score(eval_scp_path, store_score_dir, trials, sub)
    
if __name__ == "__main__":
    fire.Fire(main)

