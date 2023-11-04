#!/bin/bash

# Using our provided weight 
python val.py configs/ms1mv3_r50 --pretrained

# Or using your trained weight
# python val.py configs/ms1mv3_r50 --model_path output_ckpt/epoch_0
