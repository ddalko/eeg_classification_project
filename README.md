# How to run

ADT for all subjects (bash shell)
```
#!/bin/bash

for subject in {1..9}; do
    python main.py --mode=train --band=0,40 --label=0,1,2,3 --gpu=0 --sch=cos --eta_min=0 --epoch=500 -lr=2e-3 --seed=42 -wd=2e-3 --train_subject=$subject --criterion=FOCAL --batch_size=288 --net=DatEEGNet --stamp=DatEEGNet 
done
```

# Notice
[KOR]

실험 후 ./result/{stamp}/에 실험자 별 폴더가 생성됩니다.

각 폴더 안에는 args.json, checkpoint (DIR), log_dict.json 파일 및 디렉터리가 있는데

args.json은 하이퍼파리미터 등의 실험 세팅 정보가 저장된 파일

log_dict.json에는 실험 결과 (train loss, train acc, test/val loss, test/val acc) 가 저장됩니다.

checkpoint에는 best.tar 파라미터가 저장됩니다. 정확도가 제일 높은 웨이트만 저장되게 설정되어 있습니다.
