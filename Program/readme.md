```
python -m Program.token_input.preprocess --input_dir ./test_dataset --output_dir /data/sjx/exp/KBQA/Program/token_input/
CUDA_VISIBLE_DEVICES=5 python -m Program.train --input_dir /data/sjx/exp/KBQA/Program/token_input --save_dir /data/sjx/exp/KBQA/Program/token_input/debug --kb_json ./test_dataset/kb.json
```
