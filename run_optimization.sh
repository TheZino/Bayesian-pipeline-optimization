#!/bin/bash

# clear

iddate=`date '+%Y%m%d_%H%M'`

exp_id='exp_'$iddate


python3 ./source/optimize_on_SID.py \
    --exp_id $exp_id \
    --trials 400 \
    --save_dir ./experiments/Fuji/ \
    --seed 1234

