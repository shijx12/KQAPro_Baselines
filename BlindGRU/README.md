## Requirements
- python3
- pytorch>=1.2.0
- nltk

## How to run
1. Download [GloVe 300d vectors](http://nlp.stanford.edu/data/glove.840B.300d.zip), unzip it to get the file `glove.840B.300d.txt`, and then convert it to a pickle file for faster loading:
```
python -m utils.pickle_glove --input <path/of/glove.840B.300d.txt> --output <path/of/glove/pt>
```
This step can be skipped if you have obtained the glove pickle file in other models.
2. Preprocess the training data
```
python -m BlindGRU.preprocess --input_dir ./dataset --output_dir <dir/of/processed/files>
```
3. Train
```
python -m BlindGRU.train --input_dir <dir/of/processed/files> --save_dir <dir/of/checkpoint> --glove_pt <path/of/glove/pt>
```
4. Predict answers of the test set. It will produce a file named `predict.txt` in the `--save_dir`, storing the predictions of test questions in order.
```
python -m BlindGRU.predict --input_dir <dir/of/processed/files> --save_dir <dir/of/checkpoint>
```
