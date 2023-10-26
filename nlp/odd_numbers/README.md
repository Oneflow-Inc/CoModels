##  Generating Odd Numbers

This task can generate a sequence of odd numbers according to the input even numbers. For example, if we input `[2422, 2424, 2426]`, then we will get `[0, 2423, 2425, 2427]` as a result.

### Dataset

We generate the data by ourselves. Please read `odd_numbers/train_transformer_odd_numbers.py` for details.

### Training

You can use bash script `train.sh` to train this model.

```bash
sh train.sh
```

Note that the default parameters are following:

```bash
BATCH_SIZE=128
EPOCH=30
LEARNING_RATE=0.0001

VOCAB_SZ=10000
D_MODEL=512
DROPOUT=0.0
NHEAD=2
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
DIM_FF=128

LOAD_DIR="."
SAVE_DIR="best_model"
```

### inference

Bash script `infer.sh` is used to infer the trained model.

```bash
sh infer.sh
```

The default parameters are set as below:

```bash
VOCAB_SZ=10000
D_MODEL=512
DROPOUT=0.0
NHEAD=2
NUM_ENCODER_LAYERS=1
NUM_DECODER_LAYERS=1
DIM_FF=128

LOAD_DIR="best_model"
INPUT_START=4386
```

The parameter `input_start` is the first number of the sequence input. If it is 4386, then the program will generate the sequence `[4386, 4388, 4390]` as input.
