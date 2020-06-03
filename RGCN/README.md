## Requirements
- python3
- 

## How to run
1. preprocess the training data
```
python -m RGCN.preprocess --input_dir /data/sjx/dataset/KQApro/release --output_dir /data/sjx/exp/KBQA/RGCN
```
2. train
```
CUDA_VISIBLE_DEVICES=6 python -m RGCN.train --input_dir /data/sjx/exp/KBQA/RGCN --save_dir /data/sjx/exp/KBQA/RGCN/debug
```
