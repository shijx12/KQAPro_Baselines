## Requirements
- python3
- pytorch>=1.2.0
- transformers

## How to run
1. Preprocess the training data, and copy the `./dataset/kb.json` into `output_dir`
```
python -m Bart_Program.preprocess --input_dir ./dataset --output_dir <dir/of/processed/files> --model_name_or_path <dir/of/pretrained/BartModel>
cp ./dataset/kb.json <dir/of/processed/files>
```
2. Train
```
python -m Bart_Program.train --input_dir <dir/of/processed/files> --output_dir <dir/of/checkpoint> --save_dir <dir/of/log/files> --model_name_or_path <dir/of/pretrained/BartModel>
```
3. Predict answers of the test set. It will produce a file named `predict.txt` in the `--save_dir`, storing the predictions of test questions in order.
```
python -m Bart_Program.predict --input_dir <dir/of/processed/files> --save_dir <dir/of/log/files> --ckpt <dir/of/checkpoint>
```
