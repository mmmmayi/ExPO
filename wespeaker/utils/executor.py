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

from contextlib import nullcontext
import tableprint as tp

import torch
import torchnet as tnt
import librosa
import librosa.display
import matplotlib.pyplot as plt
def run_epoch(dataloader,
              loader_size,
              model,
              criterion,
              optimizer,
              scheduler,
              margin_scheduler,
              epoch,
              logger,
              rank,
              weight,
              num_utts,
              log_batch_interval=100,
              device=torch.device('cuda')):
    model.train()
    #print(weight)
    # By default use average pooling
    loss_cls_meter = tnt.meter.AverageValueMeter()
    loss_cstr_meter = tnt.meter.AverageValueMeter()
    loss_diff_meter = tnt.meter.AverageValueMeter()
    loss_selfcstr_meter = tnt.meter.AverageValueMeter()
    acc_meter = tnt.meter.ClassErrorMeter(accuracy=True)

    # https://github.com/wenet-e2e/wenet/blob/main/wenet/utils/executor.py#L40
    if isinstance(model, torch.nn.parallel.DistributedDataParallel):
        model_context = model.join
    else:
        model_context = nullcontext
    torch.autograd.set_detect_anomaly(True)
    with torch.set_grad_enabled(True), model_context():
        for i, batch in enumerate(dataloader):
            utts = batch['key']
            targets = batch['label']
            
            pho = batch['pho'].to(device).squeeze()
            #print(pho.shape)#[B,200]
            features = batch['wav'].to(device).squeeze()
            #print(features.shape)
            #quit()
            cur_iter = (epoch - 1) * loader_size + i
            scheduler.step(cur_iter)
            margin_scheduler.step(cur_iter)

            features = features.float().to(device)  # (B,T,F)
            targets = targets.long().to(device)
            #pho_weight = pho_model(pho)
         
            embeds,first_pooling = model(features,pho)  # (embed_a,embed_b) in most cases
            #print(first_pooling.shape)#[B,F,T]
            ###############################################
            
            idx = torch.any(first_pooling!=0, dim=-2).float()
            idx_num = torch.sum(idx,-1)
            mean = first_pooling.sum(dim=-1)/(1e-6+idx_num.unsqueeze(-1))
            if torch.any(torch.isnan(mean)):
                print(idx_num)
                quit()
            mean = mean.unsqueeze(-1)
            dis = torch.pow((mean-first_pooling),2)
            loss_selfcstr = torch.sum(dis*(idx.unsqueeze(-2)))/torch.sum(idx+1e-6)
            
            ################################################ 
            # loss for pho constraint
            enroll = first_pooling[::2,:]#[B/2,F,T]
            test = first_pooling[1::2,:]

            idx = torch.any(enroll!=0, dim=-2)*torch.any(test!=0, dim=-2).float()#[num_spk,40]
            #idx_num = torch.sum(idx,-2)#[num_spk,40]
            
            same = torch.sum(torch.pow((enroll-test),2),dim=-2)#[num_spk,40]
            loss_cstr = torch.sum(same*idx)/torch.sum(idx+1e-6)       
            ######################################################
            # loss for pho difference

            idx_diff = (torch.any(enroll!=0,-2).unsqueeze(1))*(torch.any(test!=0,-2).unsqueeze(0)).float()#[num_spk,num_spk,40]
            dis_phone = torch.sum(torch.pow((enroll.unsqueeze(1)-test.unsqueeze(0)),2),dim=-2)#[num_spk,num_spk,40]
            dis_phone = dis_phone*(idx_diff.detach())
            #semi_hard = (dis_phone>same.unsqueeze(1)).float()
            #dis_phone = semi_hard*dis_phone
            dis_phone = torch.where(dis_phone==0,float('inf'),dis_phone)#[num_spk,num_spk,40]
            min_phone,_ = torch.min(dis_phone,-2)
            
            diff = torch.sum(min_phone[min_phone!=float('inf')])
            loss_diff = diff/len(min_phone[min_phone!=float('inf')])
            
            #loss_diff = 0
            ######################################################
            outputs = model.module.projection(embeds, targets)
           
            #print(torch.any(torch.isnan(outputs)))
            loss_cls=criterion(outputs,targets)
            #loss = loss_cls 
            #loss = loss_cls+loss_selfcstr*0.0001
            loss = loss_cls+loss_cstr*weight[0]-loss_diff*weight[1]+loss_selfcstr*weight[2]
            #loss = criterion(outputs,targets)
            if torch.isnan(loss):
                print('loss_cstr',loss_cstr)
                print('loss_diff',loss_diff)
                print(torch.any(torch.isnan(outputs)))
                
                #print(torch.any(torch.isnan(features)))
                #print(torch.any(torch.isnan(embeds)))
                #for param in model.parameters():
                    #if torch.any(torch.isnan(param)):
                        #print('nan in model')
                #print(torch.any(torch.isnan(outputs)))
                quit()
            # loss, acc
            loss_cls_meter.add(loss_cls.item())
            loss_cstr_meter.add(loss_cstr.item())
            loss_diff_meter.add(loss_diff.item())
            loss_selfcstr_meter.add(loss_selfcstr.item())
            acc_meter.add(outputs.cpu().detach().numpy(),
                          targets.cpu().numpy())

            # updata the model
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()
            # log
            if (i + 1) % log_batch_interval == 0:
                logger.info(
                    tp.row((epoch, i + 1, scheduler.get_lr(),
                            margin_scheduler.get_margin()) +
                           (loss_cls_meter.value()[0],loss_cstr_meter.value()[0],loss_diff_meter.value()[0],loss_selfcstr_meter.value()[0], acc_meter.value()[0]),
                           width=10,
                           style='grid'))

    logger.info(
        tp.row((epoch, i + 1, scheduler.get_lr(),
                margin_scheduler.get_margin()) +
               (loss_cls_meter.value()[0],loss_cstr_meter.value()[0],loss_diff_meter.value()[0],loss_selfcstr_meter.value()[0], acc_meter.value()[0]),
               width=10,
               style='grid'))
