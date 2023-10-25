## Getting Started
### Prepare the Data and the Vocab

- Prepared relevant datasets:

1. [IMAGRNET_URL]

- Download the dataset and move the data file to the folder. The file structure should be like:
```bash
$ tree data
path/to/imagenet
```
### How to Train ResMLP Model with Parallelism

We provide `train.sh` for execute training. Before invoking the script, perform the following steps.
<details>
<summary>Training code</summary>
We take the ResMLP as an example to show how to train the model.

```Shell
bash train.sh
```
</details>
<details>
<summary>Inference code</summary>
We take the ResMLP as an example to show how to test the model.

```Shell
bash infer.sh
```
</details>

