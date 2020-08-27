```
python -m Program.token_input.preprocess --input_dir ./dataset --output_dir /data/sjx/exp/KBQA/Program/
cp ./dataset/kb.json /data/sjx/exp/KBQA/Program/

CUDA_VISIBLE_DEVICES=1 python -m Program.train --input_dir /data/sjx/exp/KBQA/Program --save_dir /data/sjx/exp/KBQA/Program/debug

CUDA_VISIBLE_DEVICES=2 python -m Program.predict --input_dir /data/sjx/exp/KBQA/Program --save_dir /data/sjx/exp/KBQA/Program/debug
```
