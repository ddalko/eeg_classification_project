#!/bin/zsh

# EEGNet finetuning
# python main.py --net=EEGNet --label=0,1,2,3 --gpu=0 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=baseline --seed=42 --batch_size=72 -wd=2e-4 --train_subject=9

# EEGConformer pretraining
python main.py --mode=pretrain --net=EEGConformer --label=0,1,2,3 --gpu=0 --sch=cos --gamma=0.999 --epoch=500 -lr=2e-3 --stamp=pretrained --seed=42 -wd=2e-3 --train_subject=9
