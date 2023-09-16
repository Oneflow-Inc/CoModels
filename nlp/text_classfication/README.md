## Text Classfication

### Dataset

We use Imdb dataset to test our model first.

```bash
mkdir datasets; cd datasets;
wget https://oneflow-public.oss-cn-beijing.aliyuncs.com/datasets/models/Imdb_ofrecord.tar.gz
tar zxf Imdb_ofrecord.tar.gz
```

### Traininig

The bash script `train.sh` will train our model on imdb dataset.

```bash
sh train.sh
```

The default parameters are displayed below. You can modify them to fit your own environment.

```bash
BATCH_SIZE=32
EPOCH=1
LEARNING_RATE=0.0001

SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FF=1024

IMDB_PATH="./datasets/imdb"
LOAD_DIR="."
SAVE_DIR="best_model"
```

### Inference

Script `infer.sh` can test the result of classification. We use text `"It is awesome! It is nice. The director does a good job!"` as example.

```bash
sh infer.sh
```

The default parameters are displayed below.

```bash
SEQUENCE_LEN=128
VOCAB_SZ=100000
D_MODEL=512
DROPOUT=0.1
NHEAD=8
NUM_LAYERS=4
DIM_FF=1024
LOAD_DIR="best_model"
TEXT="It is awesome! It is nice. The director does a good job!"
```
