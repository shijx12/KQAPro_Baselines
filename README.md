# KQA Pro Baselines
[KQA Pro](https://arxiv.org/abs/2007.03875) is a large-scale dataset of complex question answering over knowledge base, which provides strong supervision of SPARQL and program for each question. 
This repo implements several baselines for the dataset:

- Blind GRU. It predicts the answer in terms of only the input question, ignoring the knowledge base. We use it to measure the dataset bias.
- [KVMemNN](https://www.aclweb.org/anthology/D16-1147/) (Key-Value Memory Networks)
- [RGCN](https://arxiv.org/abs/1703.06103) (Relational Graph Convolutional Networks)
- [SRN](https://dl.acm.org/doi/10.1145/3336191.3371812) (Stepwise Relational Networks)
- Attention-based seq2seq SPARQL parser
- Attention-based seq2seq program parser
- Bart-based seq2seq SPARQL parser
- Bart-based seq2seq program parser

Instructions of how to run these models are described in their README files.
Before trying them, you need to first download the [dataset](https://cloud.tsinghua.edu.cn/f/26d05edaa5d0480bb3ae/?dl=1) and unzip it into the folder `./dataset`.
The file tree should be like
```
.
+-- dataset
|   +-- kb.json
|   +-- train.json
|   +-- val.json
|   +-- test.json
+-- GRU
|   +-- preprocess.py
|   +-- train.py
|   +-- ...
+-- KVMemNN
+-- RGCN
...
```
