# readme
1. Download dataset via `https://cloud.tsinghua.edu.cn/f/098bfce07e93419b816c/?dl=1`, and unzip it into a folder `./test_dataset`.
2. Preprocess json files into pickle files for speed-up of training.
```
python -m KVMemNN..preprocess --input_dir ./test_dataset --output_dir <your/data/directory>
```
3. Run train script
```
CUDA_VISIBLE_DEVICES=3 python -m KVMemNN.train --input_dir <your/data/directory> --save_dir <log/directory>
```
