## Requirements
- python3
- pytorch==1.12.0
- transformers==4.16.2
- kopl==0.0.5

## How to run
1. Install the KoPL engine
```
pip install kopl
```
2. Preprocess the training data, and copy the `./dataset/kb.json` into `output_dir`
```
python -m Bart_Program.preprocess --input_dir ./dataset --output_dir <dir/of/processed/files> --model_name_or_path <dir/of/pretrained/BartModel>
cp ./dataset/kb.json <dir/of/processed/files>
```
3. Train
```
python -m Bart_Program.train --input_dir <dir/of/processed/files> --output_dir <dir/of/checkpoint> --save_dir <dir/of/log/files> --model_name_or_path <dir/of/processed/files>
```
4. Predict answers of the test set. It will produce a file named `predict.txt` in the `--save_dir`, storing the predictions of test questions in order.
```
python -m Bart_Program.predict --input_dir <dir/of/processed/files> --save_dir <dir/of/log/files> --ckpt <dir/of/checkpoint>
```

## Checkpoints
1. The pretrained Bart-base checkpoint without finetuning can be downloaded here [bart-base](https://cloud.tsinghua.edu.cn/f/3b59ec6c43034cfc8841/?dl=1)
2. The checkpoint for finetuned Bart_Program can be downloaded here [finetuned](https://cloud.tsinghua.edu.cn/f/5b82ae04f9f64d1c8d1d/?dl=1)

## Change Log

- [2022/8/8] Upload the evaluation.py; update the KoPL engine based on [KoPL](https://github.com/THU-KEG/KoPL); update kb.json in dataset; 

- A different serializer and add special token in the tokenizer. Note that the argument is <dir/of/processed/files> for --model_name_or_path for Bart_Program.train
