# Copyright (c) 2021 Mobvoi Inc. (authors: Binbin Zhang)
#               2022 Chengdong Liang (liangchengdong@mail.nwpu.edu.cn)
#               2022 Hongji Wang (jijijiang77@gmail.com)
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
from torch.nn.functional import conv1d, pad
import torch.nn as nn
import io
import kaldiio
import json
import logging

import random
import tarfile
from subprocess import PIPE, Popen
from urllib.parse import urlparse
import soundfile
import numpy as np
from scipy import signal
from scipy.io import wavfile
import torch
import torchaudio
import torchaudio.compliance.kaldi as kaldi
import os
import numpy as np
AUDIO_FORMAT_SETS = set(['flac', 'mp3', 'm4a', 'ogg', 'opus', 'wav', 'wma'])

def _phn2vec(wav_length, phn_path, phn_dict, weight_dict):
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
            label_char = label_char.strip()
            if label_char=='[UNK]':
                label_char = '[SIL]'
            # if label_char == 'q':
            #     continue
            #label_num = phn_dict[label_char]

            #phone_time_seq[sample_start: sample_end] = label_num
            phone_weight[sample_start: sample_end] = phn_dict[label_char]
            
            #print('start:{},end:{},label:{}'.format(str(sample_start),str(sample_end),str(label_num)))
    #np.set_printoptions(threshold=phone_weight.size)
    #print(phone_weight)
    #quit()
    return phone_weight

def url_opener(data):
    """ Give url or local file, return file descriptor
        Inplace operation.

        Args:
            data(Iterable[str]): url or local file list

        Returns:
            Iterable[{src, stream}]
    """
    for sample in data:
        assert 'src' in sample
        # TODO(Binbin Zhang): support HTTP
        url = sample['src']
        try:
            pr = urlparse(url)
            # local file
            if pr.scheme == '' or pr.scheme == 'file':
                stream = open(url, 'rb')
            # network file, such as HTTP(HDFS/OSS/S3)/HTTPS/SCP
            else:
                cmd = f'curl -s -L {url}'
                process = Popen(cmd, shell=True, stdout=PIPE)
                sample.update(process=process)
                stream = process.stdout
            sample.update(stream=stream)
            yield sample
        except Exception as ex:
            logging.warning('Failed to open {}'.format(url))

def load_file(size,pho_path,dict,dict_weight):
        #data = torch.load(self.path / f'{fname}.pt')
        weight = _phn2vec(size[1], pho_path, dict,dict_weight)
        
        label = torch.from_numpy(weight).clone().unsqueeze(0).unsqueeze(0)
        '''
        unique_label = torch.unique(label)
        soft=nn.Softmax()
        unique_weight = soft(unique_label)
        for i in range(len(unique_label)):
            idx = torch.where(label==unique_label[i])
            label[idx]=unique_weight[i]
            print('unique_label',unique_label[i])
            print('unique_weight',unique_weight[i])
        label = label.unsqueeze(0).unsqueeze(0)
        quit()
        '''
        #win_eye = torch.eye(400).unsqueeze(1)

        #label = conv1d(label.float(), win_eye, stride=160).squeeze()
        #label, _ = torch.mode(label, dim=0)
        #label = label.long()


        return label 



def tar_file_and_group(data,path):
    """ Expand a stream of open tar files into a stream of tar file contents.
        And groups the file with same prefix

        Args:
            data: Iterable[{src, stream}]

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    line_idx = 0
    dict_raw = dict()
    dict_weight = np.load('inverse_ppg.npy',allow_pickle=True).item()
    dict_weight['[SIL]']=0
    with open('pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[phoneme] = line_idx
            line_idx += 1
 
    for sample in data:
        assert 'stream' in sample
        stream = tarfile.open(fileobj=sample['stream'], mode="r|*")
        prev_prefix = None
        example = {}
        valid = True
        for tarinfo in stream:
            name = tarinfo.name
            #print(name)#id07641/1s8qpjEwSLM/00019.wav.spk
            
            pos = name.rfind('.')
            assert pos > 0
            prefix, postfix = name[:pos], name[pos + 1:]
            if prev_prefix is not None and prefix != prev_prefix:
                example['key'] = prev_prefix
                #print(prev_prefix)
                pho_path = os.path.join(path,prev_prefix.replace('.wav','.pre.pho'))
                #if example['wav'].shape!=1:
                example['pho'] = load_file(example['wav'].shape,pho_path,dict_raw,dict_weight)
                #print(example['pho'].shape)
                #quit() 
                #print('shape:',test.shape)
                if valid:
                    yield example
                example = {}
                valid = True
            with stream.extractfile(tarinfo) as file_obj:
                try:
                    if postfix in ['spk']:
                        example[postfix] = file_obj.read().decode(
                            'utf8').strip()
                    elif postfix in AUDIO_FORMAT_SETS:
                        waveform, sample_rate = torchaudio.load(file_obj)
                        example['wav'] = waveform
                        
                        example['sample_rate'] = sample_rate
                    else:
                        example[postfix] = file_obj.read()
                except Exception as ex:
                    valid = False
                    logging.warning('error to parse {}'.format(name))
            prev_prefix = prefix
        if prev_prefix is not None:
            example['key'] = prev_prefix
            pho_path = os.path.join(path,prev_prefix.replace('.wav','.pre.pho'))
            example['pho']  = load_file(example['wav'].shape,pho_path,dict_raw,dict_weight)
            yield example
        stream.close()
        if 'process' in sample:
            sample['process'].communicate()
        sample['stream'].close()


def parse_raw(data,path):
    """ Parse key/wav/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/wav/spk

        Returns:
            Iterable[{key, wav, spk, sample_rate}]
    """
    line_idx = 0
    dict_raw = dict()
    dict_weight = np.load('ppg.npy',allow_pickle=True).item()
    dict_weight['[SIL]']=0
    with open('pho_list', 'r') as f:
        for line in f:
            phoneme = line.split(' ')[0].strip()
            dict_raw[phoneme] = line_idx
            line_idx += 1
 
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'wav' in obj
        assert 'spk' in obj
        key = obj['key']
        wav_file = obj['wav']
        spk = obj['spk']
        pho_path = wav_file.replace('.wav','.pre.pho')
        try:
            #waveform, sample_rate = torchaudio.load(wav_file)
 
            waveform, sample_rate = torchaudio.load(wav_file.replace('/hpctmp/','/scratch/'))
            pho = load_file(waveform.shape,pho_path,dict_raw,dict_weight)
            example = dict(key=key,
                           spk=spk,
                           wav=waveform,
                           sample_rate=sample_rate,
                           pho=pho)
            yield example
        except Exception as ex:
            print(ex)
            logging.warning('Failed to read {}'.format(wav_file))
            quit()


def parse_feat(data):
    """ Parse key/feat/spk from json line

        Args:
            data: Iterable[str], str is a json line has key/feat/spk

        Returns:
            Iterable[{key, feat, spk}]
    """
    for sample in data:
        assert 'src' in sample
        json_line = sample['src']
        obj = json.loads(json_line)
        assert 'key' in obj
        assert 'feat' in obj
        assert 'spk' in obj
        key = obj['key']
        feat_ark = obj['feat']
        spk = obj['spk']
        try:
            feat = torch.from_numpy(kaldiio.load_mat(feat_ark))
            example = dict(key=key,
                           spk=spk,
                           feat=feat)
            yield example
        except Exception as ex:
            logging.warning('Failed to load {}'.format(feat_ark))


def shuffle(data, shuffle_size=2500):
    """ Local shuffle the data

        Args:
            data: Iterable[{key, wav/feat, spk}]
            shuffle_size: buffer size for shuffle

        Returns:
            Iterable[{key, wav/feat, spk}]
    """
    buf = []
    for sample in data:
        buf.append(sample)
        if len(buf) >= shuffle_size:
            random.shuffle(buf)
            for x in buf:
                yield x
            buf = []
    # The sample left over
    random.shuffle(buf)
    for x in buf:
        yield x


def spk_to_id(data, spk2id):
    """ Parse spk id

        Args:
            data: Iterable[{key, wav/feat, spk}]
            spk2id: Dict[str, int]

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'spk' in sample
        if sample['spk'] in spk2id:
            label = spk2id[sample['spk']]
        else:
            label = -1
        sample['label'] = label
        yield sample


def resample(data, resample_rate=16000):
    """ Resample data.
        Inplace operation.
        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            resample_rate: target resample rate
        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        if sample_rate != resample_rate:
            sample['sample_rate'] = resample_rate
            sample['wav'] = torchaudio.transforms.Resample(
                orig_freq=sample_rate, new_freq=resample_rate)(waveform)
        yield sample


def speed_perturb(data, num_spks):
    """ Apply speed perturb to the data.
        Inplace operation.

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    speeds = [1.0, 0.9, 1.1]
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        speed_idx = random.randint(0, 2)
        if speed_idx > 0:
            wav, _ = torchaudio.sox_effects.apply_effects_tensor(
                waveform, sample_rate,
                [['speed', str(speeds[speed_idx])], ['rate',
                                                     str(sample_rate)]])
            sample['wav'] = wav
            sample['label'] = sample['label'] + num_spks * speed_idx

        yield sample


def get_random_chunk(wav, pho, chunk_len):
    """ Get random chunk

        Args:
            data: torch.Tensor (random len)
            chunk_len: chunk length

        Returns:
            torch.Tensor (exactly chunk_len)
    """
    data_len = len(wav)
    data_shape = wav.shape
    # random chunk
    if data_len >= chunk_len:
        chunk_start = random.randint(0, data_len - chunk_len)
        wav = wav[chunk_start:chunk_start + chunk_len]
        pho = pho[chunk_start:chunk_start + chunk_len]
         
    else:
        # padding
        repeat_factor = chunk_len // data_len + 1
        repeat_shape = repeat_factor if len(data_shape) == 1 else (repeat_factor, 1)
        wav = wav.repeat(repeat_shape)
        wav = wav[:chunk_len]
        pho = pho.repeat(repeat_shape)
        pho = pho[:chunk_len]

    return wav,pho


def random_chunk(data, chunk_len, data_type='shard/raw/feat'):
    """ Random chunk the data into chunk_len

        Args:
            data: Iterable[{key, wav/feat, label}]
            chunk_len: chunk length for each sample

        Returns:
            Iterable[{key, wav/feat, label}]
    """
    for sample in data:
        assert 'key' in sample

        if data_type == 'feat':
            assert 'feat' in sample
            feat = sample['feat']
            feat = get_random_chunk(feat, chunk_len)
            sample['feat'] = feat
        else:
            assert 'wav' in sample
            wav = sample['wav'][0]
            pho = sample['pho'].squeeze()
            wav,pho = get_random_chunk(wav, pho, chunk_len)
            win_eye = torch.eye(400).unsqueeze(1)
            pho = conv1d(pho.unsqueeze(0).float(), win_eye, stride=160).squeeze()
            pho, _ = torch.mode(pho, dim=0)
            sample['wav'] = wav.unsqueeze(0)
            sample['pho'] = pho.float()

        yield sample


def add_reverb_noise(data,
                     reverb_source,
                     noise_source,
                     resample_rate=16000,
                     aug_prob=0.6):
    """ Add reverb & noise aug

        Args:
            data: Iterable[{key, wav, label, sample_rate}]
            reverb_source: reverb LMDB data source
            noise_source: noise LMDB data source
            resample_rate: resample rate for reverb/noise data
            aug_prob: aug probability

        Returns:
            Iterable[{key, wav, label, sample_rate}]
    """
    for sample in data:
        #print(sample.key)
        
        assert 'wav' in sample
        assert 'key' in sample
        if aug_prob > random.random():
            aug_type = random.randint(1, 2)
            if aug_type == 1:
                # add reverberation
                audio = sample['wav'].numpy()[0]
                audio_len = audio.shape[0]

                rir_data = random.choice(reverb_source).strip('\n')
                #rir_audio, rir_sr = soundfile.read(rir_data)
                rir_audio, rir_sr = soundfile.read(rir_data.replace('/hpctmp','/scratch'))
                rir_audio = rir_audio.astype(np.float32)
                if rir_sr != resample_rate:
                    rir_audio = signal.resample(
                        rir_audio,
                        int(len(rir_audio) / rir_sr * resample_rate))
                rir_audio = rir_audio / np.sqrt(np.sum(rir_audio**2))
                out_audio = signal.convolve(audio, rir_audio,
                                            mode='full')[:audio_len]
            else:
                # add additive noise
                audio = sample['wav'].numpy()[0]
                out_audio=audio
                audio_len = audio.shape[0]
                audio_db = 10 * np.log10(np.mean(audio**2) + 1e-6)
                noise_path = random.choice(noise_source).strip('\n')
                #noise_data,_ = soundfile.read(noise_path)
                noise_data,_ = soundfile.read(noise_path.replace('/hpctmp','/scratch'))
                key = noise_path.split('/')[5]

                noise_list = [noise_data]
                if key.startswith('noise'):
                    type='noise'
                    snr_range = [random.choice([0,5,10,15])]
                    snr_list = [snr_range]
                elif key.startswith('speech'):
                    type='speech'
                    snr_range = [random.choice([13,15,17,20])]
                    num_speech = random.choice([3,4,5,6,7])
                    snr_list = [snr_range]
                    while len(noise_list)<num_speech:
                        noise_path = random.choice(noise_source).strip('\n')
                        key = noise_path.split('/')[5]
                        if key.startswith('speech'):
                            #noise_data,_ = soundfile.read(noise_path)
                            noise_data,_ = soundfile.read(noise_path.replace('/hpctmp','/scratch'))
                            noise_list.append(noise_data)
                            snr_list.append(random.choice([13,15,17,20]))

                elif key.startswith('music'):
                    type='music'
                    snr_range = [random.choice([5,8,10,15])]
                    snr_list = [snr_range]
                else:
                    print('error')
                    quit()
                noise_chunk_list = []
                for noise_audio in noise_list:
                    noise_current,_ = get_random_chunk(noise_audio, noise_audio, audio_len)
                    noise_chunk_list.append(noise_current)

                for i in range(len(noise_chunk_list)):
                    snr = snr_list[i]
                    noise = noise_chunk_list[i]
                    noise_db = 10 * np.log10(np.mean(noise**2) + 1e-6)
                    noise_audio = np.sqrt(10**((audio_db - noise_db - snr) / 10)) * noise
                    out_audio = out_audio + noise_audio
            # normalize into [-1, 1]
            out_audio = out_audio / (np.max(np.abs(out_audio)) + 1e-6)
            '''
            if aug_type==2 and type=='speech':
                soundfile.write(os.path.join('/data_a11/mayi/project/wespeaker_origin/wespeaker/examples/voxceleb/v2/exp/resnet_babble',type+sample['key'].replace('/','_')), out_audio, 16000)
                soundfile.write(os.path.join('/data_a11/mayi/project/wespeaker_origin/wespeaker/examples/voxceleb/v2/exp/resnet_babble','clean'+sample['key'].replace('/','_')), audio, 16000)
                print(type+sample['key'])
            '''
            sample['wav'] = torch.from_numpy(out_audio).unsqueeze(0)
            
        yield sample


def compute_fbank(data,
                  num_mel_bins=80,
                  frame_length=25,
                  frame_shift=10,
                  dither=1.0):
    """ Extract fbank

        Args:
            data: Iterable[{key, wav, label, sample_rate}]

        Returns:
            Iterable[{key, feat, label, sample_rate}]
    """
    for sample in data:
        assert 'sample_rate' in sample
        assert 'wav' in sample
        assert 'key' in sample
        assert 'label' in sample
        sample_rate = sample['sample_rate']
        waveform = sample['wav']
        waveform = waveform * (1 << 15)
        # Only keep key, feat, label
        mat = kaldi.fbank(waveform,
                          num_mel_bins=num_mel_bins,
                          frame_length=frame_length,
                          frame_shift=frame_shift,
                          dither=dither,
                          sample_frequency=sample_rate,
                          window_type='hamming',
                          use_energy=False)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def apply_cmvn(data, norm_mean=True, norm_var=False):
    """ Apply CMVN

        Args:
            data: Iterable[{key, feat, label}]

        Returns:
            Iterable[{key, feat, label}]
    """
    for sample in data:
        assert 'key' in sample
        assert 'feat' in sample
        assert 'label' in sample
        mat = sample['feat']
        if norm_mean:
            mat = mat - torch.mean(mat, dim=0)
        if norm_var:
            mat = mat / torch.sqrt(torch.var(mat, dim=0) + 1e-8)
        yield dict(key=sample['key'], label=sample['label'], feat=mat)


def spec_aug(data, num_t_mask=1, num_f_mask=1, max_t=10, max_f=8, prob=0.6):
    """ Do spec augmentation
        Inplace operation

        Args:
            data: Iterable[{key, feat, label}]
            num_t_mask: number of time mask to apply
            num_f_mask: number of freq mask to apply
            max_t: max width of time mask
            max_f: max width of freq mask
            prob: prob of spec_aug

        Returns
            Iterable[{key, feat, label}]
    """
    for sample in data:
        if random.random() < prob:
            assert 'feat' in sample
            x = sample['feat']
            assert isinstance(x, torch.Tensor)
            # y = x.clone().detach()
            y = x.detach()  # inplace operation
            max_frames = y.size(0)
            max_freq = y.size(1)
            # time mask
            for i in range(num_t_mask):
                start = random.randint(0, max_frames - 1)
                length = random.randint(1, max_t)
                end = min(max_frames, start + length)
                y[start:end, :] = 0
            # freq mask
            for i in range(num_f_mask):
                start = random.randint(0, max_freq - 1)
                length = random.randint(1, max_f)
                end = min(max_freq, start + length)
                y[:, start:end] = 0
            sample['feat'] = y
        yield sample
