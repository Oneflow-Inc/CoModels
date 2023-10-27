## Getting Started
### Prepare the Data and the Vocab

- We have prepared relevant datasets, which can be downloaded from the following links:

1. [VOCAB_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/gpt_dataset/gpt2-vocab.json)

2. [VOCAB_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/gpt_dataset/gpt2-merges.txt)

2. [BIN_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.bin)
3. [IDX_DATA_URL](https://oneflow-static.oss-cn-beijing.aliyuncs.com/ci-files/dataset/libai/bert_dataset/loss_compara_content_sentence.idx)

- Download the dataset and move the data file to the folder. The file structure should be like:
```bash
$ tree data
path/to/gpt_data
├── gpt2-vocab.json
├── gpt2-merges.txt
├── loss_compara_content_sentence.bin
└── loss_compara_content_sentence.idx
```
### How to Train gpt2 Model with Parallelism

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

