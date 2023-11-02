#!/bin/bash

# Using our provided weight 
python val.py configs/ms1mv3_r101 --pretrained

# Or using your trained weight
# python val.py configs/ms1mv3_r101 --model_path output_ckpt/epoch_0
