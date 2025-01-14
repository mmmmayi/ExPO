#!/bin/bash
dur=
exp_dir=
trials="vox1_O_cleaned.kaldi vox1_E_cleaned.kaldi vox1_H_cleaned.kaldi"
data=data
sub="vox1_O"


. tools/parse_options.sh
. path.sh

trials_dir=${data}/${sub}/trials

for x in $trials; do
    python wespeaker/bin/pho_similarity.py \
      --exp_dir ${exp_dir} \
      --eval_scp_path ${exp_dir}/embeddings/${sub}/phovector_${dur}s.scp \
      ${trials_dir}/${x}

    #python wespeaker/bin/analysis.py \
      #--exp_dir ${exp_dir} \
      #--eval_scp_path ${exp_dir}/embeddings/${sub}/phovector_${dur}s.scp \
      #${trials_dir}/${x}
done

