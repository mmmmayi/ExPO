#!/bin/bash

# Copyright (c) 2022 Hongji Wang (jijijiang77@gmail.com)
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

stage=4
stop_stage=4
data=data

. tools/parse_options.sh || exit 1

download_dir=${data}/download_data
rawdata_dir=/hpctmp/ma_yi/dataset_vox1/voxceleb1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Download musan.tar.gz, rirs_noises.zip, vox1_test_wav.zip, vox1_dev_wav.zip, and vox2_aac.zip."
  echo "This may take a long time. Thus we recommand you to download all archives above in your own way first."

  ./local/download_data.sh --download_dir ${download_dir}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Decompress all archives ..."
  echo "This could take some time ..."

  for archive in musan.tar.gz rirs_noises.zip vox1_test_wav.zip vox1_dev_wav.zip vox2_aac.zip; do
    [ ! -f ${download_dir}/$archive ] && echo "Archive $archive not exists !!!" && exit 1
  done
  [ ! -d ${rawdata_dir} ] && mkdir -p ${rawdata_dir}

  if [ ! -d ${rawdata_dir}/musan ]; then
    tar -xzvf ${download_dir}/musan.tar.gz -C ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/RIRS_NOISES ]; then
    unzip ${download_dir}/rirs_noises.zip -d ${rawdata_dir}
  fi

  if [ ! -d ${rawdata_dir}/voxceleb1 ]; then
    mkdir -p ${rawdata_dir}/voxceleb1/test ${rawdata_dir}/voxceleb1/dev
    unzip ${download_dir}/vox1_test_wav.zip -d ${rawdata_dir}/voxceleb1/test
    unzip ${download_dir}/vox1_dev_wav.zip -d ${rawdata_dir}/voxceleb1/dev
  fi

  if [ ! -d ${rawdata_dir}/voxceleb2_m4a ]; then
    mkdir -p ${rawdata_dir}/voxceleb2_m4a
    unzip ${download_dir}/vox2_aac.zip -d ${rawdata_dir}/voxceleb2_m4a
  fi

  echo "Decompress success !!!"
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Convert voxceleb2 wav format from m4a to wav using ffmpeg."
  echo "This could also take some time ..."

  if [ ! -d ${rawdata_dir}/voxceleb2_wav ]; then
    ./local/m4a2wav.pl ${rawdata_dir}/voxceleb2_m4a dev ${rawdata_dir}/voxceleb2_wav
    # Here we use 8 parallel jobs
    cat ${rawdata_dir}/voxceleb2_wav/dev/m4a2wav_dev.sh | xargs -P 8 -i sh -c "{}"
  fi

  echo "Convert m4a2wav success !!!"
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config

  #mkdir -p data/vox1_training
  # musan
  #find /hpctmp/ma_yi/dataset_vox1/musan -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >data/musan/wav.scp
  # rirs
  #find /hpctmp/ma_yi/dataset_vox1/RIRS_NOISES/simulated_rirs -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' >data/rirs/wav.scp
  # vox1
  find ${rawdata_dir}/wav ${rawdata_dir}/test -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox1/wav.scp
  
  #find ${rawdata_dir}/wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >data/vox1_training/wav.scp
  awk '{print $1}' data/vox1/wav.scp | awk -F "/" '{print $0,$1}' >data/vox1/utt2spk
  ./tools/utt2spk_to_spk2utt.pl data/vox1/utt2spk >data/vox1/spk2utt
  #if [ ! -d ${data}/vox1/trials ]; then
    #echo "Download trials for vox1 ..."
    #mkdir -p ${data}/vox1/trials
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test.txt -O ${data}/vox1/trials/vox1-O.txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard.txt -O ${data}/vox1/trials/vox1-H.txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all.txt -O ${data}/vox1/trials/vox1-E.txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/veri_test2.txt -O ${data}/vox1/trials/vox1-O\(cleaned\).txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_hard2.txt -O ${data}/vox1/trials/vox1-H\(cleaned\).txt
    #wget --no-check-certificate https://www.robots.ox.ac.uk/~vgg/data/voxceleb/meta/list_test_all2.txt -O ${data}/vox1/trials/vox1-E\(cleaned\).txt
    # transform them into kaldi trial format
  
    #awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vox1/trials/vox1-O\(cleaned\).txt >${data}/vox1/trials/vox1_O_cleaned.kaldi
    #awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vox1/trials/vox1-H\(cleaned\).txt >${data}/vox1/trials/vox1_H_cleaned.kaldi
    #awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/vox1/trials/vox1-E\(cleaned\).txt >${data}/vox1/trials/vox1_E_cleaned.kaldi
  #fi
  # vox2
  #find /home2/mayi_data/dataset/voxceleb/voxceleb2/voxceleb2_wav -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/vox2_pooling/wav.scp
  #awk '{print $1}' ${data}/vox2_pooling/wav.scp | awk -F "/" '{print $0,$1}' >${data}/vox2_pooling/utt2spk
  #./tools/utt2spk_to_spk2utt.pl ${data}/vox2_pooling/utt2spk >${data}/vox2_pooling/spk2utt

  echo "Success !!!"
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config
  mkdir -p data/sitw
  # sitw
  find ${rawdata_dir} -name "*.wav" | awk -F"/" '{filename=$(NF); sub(/\.wav$/, "", filename); print filename,$0}' | sort >${data}/sitw/wav.scp

  awk '{print $1}' data/sitw/wav.scp | awk -F "/" '{print $0,$0}' >data/sitw/utt2spk
  ./tools/utt2spk_to_spk2utt.pl data/sitw/utt2spk >data/sitw/spk2utt
  
  #awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/sitw/trials/sitw_dev >${data}/sitw/trials/sitw_dev.kaldi
  #awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/sitw/trials/sitw_eval >${data}/sitw/trials/sitw_eval.kaldi
  
  
  echo "Success !!!"
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  echo "Prepare wav.scp for each dataset ..."
  export LC_ALL=C # kaldi config
  #mkdir -p data/librispeech
  rawdata_dir=/hpctmp/ma_yi/librispeech
  
  #find ${rawdata_dir} -name "*.wav" | awk -F"/" '{print $(NF-2)"/"$(NF-1)"/"$NF,$0}' | sort >${data}/librispeech/wav.scp
  #awk '{print $1}' data/librispeech/wav.scp | awk -F "/" '{print $0,$1}' >data/librispeech/utt2spk
  #awk '{print $1}' data/librispeech/wav.scp | awk -F "/" '{print $0,$0}' >data/librispeech/utt2spk
  #./tools/utt2spk_to_spk2utt.pl data/librispeech/utt2spk >data/librispeech/spk2utt
  
  awk '{if($1==0)label="nontarget";else{label="target"}; print $2,$3,label}' ${data}/librispeech/trials/clean_new >${data}/librispeech/trials/clean.kaldi
  
  
  echo "Success !!!"
fi
