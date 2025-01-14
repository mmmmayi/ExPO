#!/bin/bash


. ./path.sh || exit 1

stage=7
stop_stage=7

data=data
data_type="raw"  # shard/raw

config=conf/ecapa_tdnn.yaml
#exp_dir=/hpctmp/ma_yi/exp/2pooling_selfcstr0.0001
exp_dir=/hpctmp/ma_yi/exp/vox2_2pooling_cstrPho0.0007_HardDiff0.00001
#exp_dir=/hpctmp/ma_yi/exp/2pooling_cstrPho0.0007_HardDiff.00005_selfcstr0.005 
gpus="[0]"
num_avg=10
checkpoint=

trials="vox1_O_cleaned.kaldi"
#trials="vox1_O_cleaned.kaldi"
score_norm_method="asnorm"  # asnorm/snorm
top_n=300

# setup for large margin fine-tuning
lm_config=conf/resnet_lm.yaml


. tools/parse_options.sh || exit 1

if [ ${stage} -le 1 ] && [ ${stop_stage} -ge 1 ]; then
  echo "Prepare datasets ..."
  ./local/prepare_data.sh --stage 4 --stop_stage 4 --data ${data}
fi

if [ ${stage} -le 2 ] && [ ${stop_stage} -ge 2 ]; then
  echo "Covert train and test data to ${data_type}..."
  for dset in vox1; do
    if [ $data_type == "shard" ]; then
      python tools/make_shard_list.py --num_utts_per_shard 1000 \
          --num_threads 10 \
          --prefix shards \
          --shuffle \
          ${data}/$dset/wav.scp ${data}/$dset/utt2spk  \
          ${data}/$dset/shards ${data}/$dset/shard.list
    else
      python tools/make_raw_list.py ${data}/$dset/wav.scp \
          ${data}/$dset/utt2spk ${data}/$dset/raw.list
    fi
  done
  # Convert all musan data to LMDB
  #python tools/make_lmdb.py ${data}/musan/wav.scp ${data}/musan/lmdb
  # Convert all rirs data to LMDB
  #python tools/make_lmdb.py ${data}/rirs/wav.scp ${data}/rirs/lmdb
fi

if [ ${stage} -le 3 ] && [ ${stop_stage} -ge 3 ]; then
  echo "Start training ..."
  num_gpus=$(echo $gpus | awk -F ',' '{print NF}')
  torchrun --standalone --nnodes=1 --nproc_per_node=$num_gpus \
    wespeaker/bin/train.py --config $config \
      --exp_dir ${exp_dir} \
      --gpus $gpus \
      --num_avg ${num_avg} \
      --data_type "${data_type}" \
      --train_data ${data}/vox1_training/${data_type}.list \
      --train_label ${data}/vox1_training/utt2spk \
      --pho_path /scratch/ma_yi/dataset_vox1/voxceleb1/wav \
      --reverb_data /scratch/ma_yi/dataset_vox1/RIRS_NOISES/file_list \
      --noise_data /scratch/ma_yi/dataset_vox1/musan/file_list \
      ${checkpoint:+--checkpoint $checkpoint}
fi

if [ ${stage} -le 4 ] && [ ${stop_stage} -ge 4 ]; then
  #echo "Do model average ..."
  
  avg_model=$exp_dir/models/model_0.pt
  
  #python wespeaker/bin/average_model.py \
    #--dst_model $avg_model \
    #--src_path $exp_dir/models \
    #--num ${num_avg}
  
  dur=0
  
  echo "Extract embeddings ..."
  local/extract_voxO.sh \
      --dur $dur\
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}
  
  echo "Score ..."
  local/score.sh \
      --dur $dur\
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials "$trials"
  #done
  
fi

if [ ${stage} -le 5 ] && [ ${stop_stage} -ge 5 ]; then
  avg_model=$exp_dir/models/model_0.pt
  dur=0
  trials="sitw_dev.kaldi sitw_eval.kaldi" 
  #trials="sitw_dev.kaldi"
  sub="sitw"
  echo "Extract embeddings for sitw ..."
  '
  local/extract_sitw.sh \
      --dur $dur\
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}
  
  echo "Score ..."
  
  local/score.sh \
      --dur $dur\
      --sub "sitw"\
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials "$trials" \
      --trials_dir ${data}/sitw/trials
  '

  local/pho_score.sh \
      --dur $dur \
      --data ${data} \
      --exp_dir $exp_dir \
      --sub ${sub} \
      --trials "$trials" \
      
  
  local/score.sh \
      --dur $dur\
      --sub "sitw" \
      --stage 2 --stop-stage 2 \
      --data ${data} \
      --suffix "pho_score" \
      --exp_dir $exp_dir \
      --trials "$trials" \
      --trials_dir ${data}/sitw/trials
  
fi

if [ ${stage} -le 6 ] && [ ${stop_stage} -ge 6 ]; then
  avg_model=$exp_dir/models/model_0.pt
  dur=0
  trials="clean.kaldi" 
  sub="librispeech"
  echo "Extract embeddings for librispeech ..."
  '
  local/extract_librispeech.sh \
      --dur $dur\
      --exp_dir $exp_dir --model_path $avg_model \
      --nj 1 --gpus $gpus --data_type $data_type --data ${data}
  
  echo "Score ..."
  
  local/score.sh \
      --dur $dur\
      --sub "librispeech"\
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials "$trials" \
      --trials_dir ${data}/librispeech/trials
  '
  
  local/pho_score.sh \
      --dur $dur \
      --data ${data} \
      --exp_dir $exp_dir \
      --sub ${sub} \
      --trials "$trials" \
      
  
  local/score.sh \
      --dur $dur\
      --sub "librispeech" \
      --stage 2 --stop-stage 2 \
      --data ${data} \
      --suffix "pho_score" \
      --exp_dir $exp_dir \
      --trials "$trials" \
      --trials_dir ${data}/librispeech/trials
  
fi

if [ ${stage} -le 7 ] && [ ${stop_stage} -ge 7 ]; then
  
  avg_model=$exp_dir/models/model_0.pt
  dur=0
  trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi" 
  
  local/extract_vox.sh \
    --dur $dur \
    --exp_dir $exp_dir --model_path $avg_model \
    --nj 1 --gpus $gpus --data_type $data_type --data ${data} 
  
  local/score.sh \
      --dur $dur\
      --sub "vox1" \
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials "$trials"
  
  local/pho_score.sh \
      --dur $dur \
      --data ${data} \
      --exp_dir $exp_dir \
      --sub "vox1" \
      --trials "$trials" 
  
  local/score.sh \
      --dur $dur\
      --sub "vox1" \
      --stage 2 --stop-stage 2 \
      --data ${data} \
      --suffix "pho_score" \
      --exp_dir $exp_dir \
      --trials "$trials"
  
fi

if [ ${stage} -le 9 ] && [ ${stop_stage} -ge 9 ]; then
  
  trials="vox1_O_cleaned.kaldi"
  avg_model=$exp_dir/models/model_0.pt
  dur=0
  '
  local/extract_voxO.sh \
    --dur $dur \
    --exp_dir $exp_dir --model_path $avg_model \
    --nj 1 --gpus $gpus --data_type $data_type --data ${data} 
  
  local/score.sh \
      --dur $dur\
      --sub "vox1_O" \
      --stage 1 --stop-stage 2 \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials "$trials"
  '
  local/pho_score.sh \
      --dur $dur \
      --data ${data} \
      --exp_dir $exp_dir \
      --trials "$trials" 
  
  local/score.sh \
      --dur $dur\
      --sub "vox1_O" \
      --stage 2 --stop-stage 2 \
      --data ${data} \
      --suffix "pho_score" \
      --exp_dir $exp_dir \
      --trials "$trials"
fi

if [ ${stage} -le 10 ] && [ ${stop_stage} -ge 10 ]; then
  exp_dir="exp/2pooling+exp/2pooling_constrain+exp/2pooling_constrain_diff+exp/2pooling_constrain_cosOneSide+exp/ecapa_tstp"
  python wespeaker/bin/tsne.py --exp_dir ${exp_dir} 
fi 

