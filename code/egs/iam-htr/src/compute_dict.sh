#!/bin/bash
set -e;
export LC_NUMERIC=C;

batch_size=10;
gpu=1;
checkpoint="experiment.ckpt.lowest-valid-cer*";

if [ $gpu -gt 0 ]; then
  export CUDA_VISIBLE_DEVICES=$((gpu-1));
  gpu=1;
fi;

mkdir -p exper/puigcerver17/decoder_dict;

python3 src/pylaia-htr-compute-dict.py \
    --syms exper/puigcerver17/train/syms_ctc.txt \
    --img_dirs data/imgs/lines_h128 \
    --txt_table data/lang/puigcerver/lines/char/tr.txt \
    --train_path exper/puigcerver17/train \
    --gpu 1 \
    --batch_size 4 \
    --checkpoint experiment.ckpt.lowest-valid-cer*\
    --score_function cer \
    --save_dict_filename exper/puigcerver17/decoder_dict/train.pkl;