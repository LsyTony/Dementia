#!/bin/bash
INPUT_LENGTH=4096
OUTPUT_DIR=
#where the model checkpoint and result will output to
DATA_DIR=
#the path where the input dataset at
CACHE_DIR=
#where to put model cache

#add --do_train \ if training else no
#add --do_validate \ if validating else no
#add --do_predict \ if testing else no
python -m models.transformers.main \
    --max_epochs 5 \
    --max_seq_length ${INPUT_LENGTH} \
    --output_dir $OUTPUT_DIR \
    --data_dir $DATA_DIR \
    --model_name_or_path allenai/longformer-base-4096 \
    --warmup_steps 500 \
    --learning_rate 5e-5 \
    --adam_epsilon 1e-3 \
    --do_train \
    --do_predict \
    --cache_dir $CACHE_DIR \
    --fp16 
    
