#!/bin/bash

gpu_id=1
ner_lr=2e-5
ner_max_norm=0.25

re_lr=2e-5
re_max_norm=0.25
re_fusion_attention_drop=0.1
re_fusion_hidden_drop=0.1

epoch=10
batch_size=32
hidden_act="gelu"

for seed in 2024;do
    ner_output="output/${seed}/ner"
    CUDA_VISIBLE_DEVICES=${gpu_id} python run_entity.py \
        --do_train --do_eval --eval_test \
        --learning_rate=${ner_lr}  \
        --train_batch_size=32 --num_epoch 30 --max_context_length 55 \
        --seed ${seed} \
        --question --multiquestion --directional \
        --data_dir  ../data/txt_merge \
        --model bert-base-uncased \
        --fusion_attention_drop ${re_fusion_attention_drop} --fusion_hidden_drop ${re_fusion_hidden_drop} --fusion_attention_head 12 --fusion_num_hidden_layers 12 \
        --max_norm ${ner_max_norm} --hidden_act ${hidden_act} \
        --output_dir ${ner_output} 

    re_output="output/${seed}/re"
    CUDA_VISIBLE_DEVICES=${gpu_id} python run_relation.py \
        --do_train --do_eval --eval_test --train_file ../data/txt_merge/train.json \
        --model bert-base-uncased \
        --train_batch_size ${batch_size} --learning_rate ${re_lr} --num_train_epochs 20 --max_context_length 55 \
        --question --multiquestion --directional \
        --seed ${seed} \
        --fusion_attention_drop ${re_fusion_attention_drop} --fusion_hidden_drop ${re_fusion_hidden_drop} --fusion_attention_head 12 --fusion_num_hidden_layers 12 \
        --max_norm ${re_max_norm}  --hidden_act ${hidden_act} \
        --entity_output_dir ${ner_output} \
        --output_dir  ${re_output}  
done;