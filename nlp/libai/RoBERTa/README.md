## Getting Started
### Prepare the Data and the Vocab

- We have prepared relevant datasets, which can be downloaded from the following links:

1. [VOCAB_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/roberta_dataset/roberta-vocab.json)
2. [MERGES_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/roberta_dataset/roberta-vocab.json)
2. [BIN_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/roberta_dataset/loss_compara_content_sentence.bin)
3. [IDX_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/roberta_dataset/loss_compara_content_sentence.idx)

- Download the dataset and move the data file to the folder. The file structure should be like:
```bash
$ tree data
path/to/roberta_data
├── vocab.json
├── merges.txt
├── loss_compara_content_sentence.bin
└── loss_compara_content_sentence.idx
```
### How to Train RoBERTa Model with Parallelism

We provide `train.sh` for execute training. Before invoking the script, perform the following steps.
<details>
<summary>Training code</summary>

```Shell
bash train.sh
```
</details>
<details>
<summary>Inference code</summary>

```Shell
bash infer.sh
```
</details>

