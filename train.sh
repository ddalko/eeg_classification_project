#!/bin/bash

# EEGNet finetuning
# python main.py --net=EEGNet --label=0,1,2,3 --gpu=0 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=baseline --seed=42 --batch_size=72 -wd=2e-4 --train_subject=9

# EEGConformer finetuning
# python main.py --mode=train --net=EEGConformer --label=0,1,2,3 --gpu=0 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=241129_finetuned --seed=42 -wd=2e-4 --train_subject=1 --batch_size=72 --criterion=FOCAL --pretrained_path=result/241128_pretrained_all/1/checkpoint/best.tar

# EEGConformer finetuning
# python main.py --mode=train --net=EEGConformer --label=0,1,2,3 --gpu=0 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=241129_finetuned --seed=42 -wd=2e-4 --train_subject=1 --batch_size=72 --criterion=FOCAL --pretrained_path=result/241128_pretrained_all_data_aug/1/checkpoint/best.tar

# ATCNet finetuning
# python main.py --mode=train --net=ATCNet --label=0,1,2,3 --gpu=0 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=241129_atcnet_finetuned --seed=42 -wd=2e-4 --train_subject=1 --batch_size=72 --criterion=FOCAL --pretrained_path=result/241129_ATCNet_pretrained_all_data/1/checkpoint/best.tar
python main.py --mode=train --net=ATCNet --label=0,1,2,3 --gpu=1 --sch=exp --gamma=0.999 --epoch=50 -lr=2e-4 --stamp=241129_atcnet_finetuned --seed=42 -wd=2e-4 --train_subject=1 --batch_size=72 --criterion=FOCAL --pretrained_path=result/241129_ATCNet_pretrained_all_data_batch_288/1/checkpoint/best.tar