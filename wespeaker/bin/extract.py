# Copyright (c) 2021 Hongji Wang (jijijiang77@gmail.com)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#   http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

import os
import fire
import kaldiio
import torch
import copy
from torch.utils.data import DataLoader
from tqdm import tqdm
import torch.nn as nn
from torch.nn.functional import conv1d, pad
from wespeaker.dataset.dataset import Dataset
from wespeaker.models.speaker_model import get_speaker_model,get_pho_model
from wespeaker.utils.checkpoint import load_checkpoint
from wespeaker.utils.utils import parse_config_or_kwargs, validate_path


def extract(config='conf/config.yaml', **kwargs):
    # parse configs first
    configs = parse_config_or_kwargs(config, **kwargs)
    model_path = configs['model_path']
    pho_path = configs['pho_path']
    dur = configs['dur']
    embed_ark = configs['embed_ark']
    pho_ark = configs['pho_ark']
    batch_size = configs.get('batch_size', 1)
    num_workers = configs.get('num_workers', 1)
    # Since the input length is not fixed, we set the built-in cudnn
    # auto-tuner to False
    torch.backends.cudnn.benchmark = False

    model = get_speaker_model(configs['model'])(**configs['model_args'])
    load_checkpoint(model, model_path)
    device = torch.device("cuda")
    model.to(device).eval()
    #pho_model = get_pho_model(configs['model'])()
    #pho_model.eval().cuda()
    # test_configs
    test_conf = copy.deepcopy(configs['dataset_args'])
    test_conf['speed_perturb'] = False
    if 'fbank_args' in test_conf:
        test_conf['fbank_args']['dither'] = 0.0
    elif 'mfcc_args' in test_conf:
        test_conf['mfcc_args']['dither'] = 0.0
    test_conf['spec_aug'] = False
    test_conf['shuffle'] = False
    dataset = Dataset(configs['data_type'],
                      configs['data_list'],
                      test_conf,
                      spk2id_dict={},
                      pho_path = pho_path,
                      whole_utt=(batch_size == 1),
                      reverb_lmdb_file=None,
                      noise_lmdb_file=None,
                      test=True)
    dataloader = DataLoader(dataset,
                            shuffle=False,
                            batch_size=batch_size,
                            num_workers=num_workers,
                            prefetch_factor=4)
    validate_path(embed_ark)
    embed_ark = os.path.abspath(embed_ark)
    embed_scp = embed_ark[:-3] + "scp"
    validate_path(pho_ark)
    pho_ark = os.path.abspath(pho_ark)
    pho_scp = pho_ark[:-3] + "scp"


    with torch.no_grad():
        with kaldiio.WriteHelper('ark,scp:' + embed_ark + "," +embed_scp) as embed_writer, kaldiio.WriteHelper('ark,scp:' + pho_ark + "," +pho_scp) as pho_writer:
            for _, batch in enumerate(dataloader):
                utts = batch['key']
                features = batch['wav'].to(device).squeeze()
                pho = batch['pho'].squeeze()
                if dur>0:
                    if features.shape[-1]>dur*16000:
                        features=features[:dur*16000]
                    if pho.shape[-1]>dur*16000:
                        pho = pho[:dur*16000]
                #print(pho.shape)
                
                win_eye = torch.eye(400).unsqueeze(1)
                #softmax = nn.Softmax()
                pho = conv1d(pho.unsqueeze(0).float(), win_eye, stride=160).squeeze()
                pho, _ = torch.mode(pho, dim=0)
                #unique_label = torch.unique(pho) 
                #unique_weight = softmax(unique_label)
                #for i in range(len(unique_label)):
                    #idx = torch.where(pho==unique_label[i])
                    #pho[idx]=unique_weight[i]
                pho=pho.float().cuda()
                #pho_weight = pho_model(pho.unsqueeze(0))
                features = features.float()  # (B,T*16000)
                # Forward through model
                #encoder = model(features,'encoder')
                embeds,pho = model(features.unsqueeze(0), pho.unsqueeze(0).detach())  # embed or (embed_a, embed_b)
                #embeds = outputs[-1] if isinstance(outputs, tuple) else outputs
                embeds = embeds.cpu().detach().numpy()  # (B,F)
                pho = torch.nn.functional.normalize(pho.squeeze(),p=2,dim=0)
                pho = pho.squeeze().cpu().detach().numpy()  # (B,F)
                for i, utt in enumerate(utts):
                    embed = embeds[i]
                    embed_writer(utt, embed)
                    pho_writer(utt,pho)    


if __name__ == '__main__':
    fire.Fire(extract)
