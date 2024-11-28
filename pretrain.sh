#!/bin/bash

# EEGConformer pretraining with session 1,2 data (same as EEGNet pretraining configuration)

for subject in {1..9}; do
    python main.py --mode=pretrain --load_all --net=EEGConformer --band=0,40 --label=0,1,2,3 --gpu=0 --sch=cos --eta_min=0 --epoch=500 -lr=2e-3 --stamp=241128_pretrained_all --seed=42 -wd=2e-3 --train_subject=$subject --batch_size=576 --criterion=FOCAL
done