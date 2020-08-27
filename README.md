# readme
1. Download dataset into the folder `./dataset`.
2. Preprocess json files into pickle files for speed-up of training.
```
python -m KVMemNN.preprocess --input_dir ./dataset --output_dir <your/data/directory>
```
3. Copy `./dataset/kb.json` into `<your/data/directory>`
4. Run train script
```
CUDA_VISIBLE_DEVICES=3 python -m KVMemNN.train --input_dir <your/data/directory> --save_dir <log/directory>
```
