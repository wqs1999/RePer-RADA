#!/bin/bash

# Run RPTF model on ETTh1 dataset
# Example script for long-term forecasting

dataset="ETTh1"
root_path="./data/ETT/"
data_path="ETTh1.csv"
model="RPTF"
task="long_term_forecast"
seq_len=672
label_len=576
token_len=96
test_pred_len=96

# Model hyperparameters (from paper Section 4.3)
llm_ckp_dir="./llama"
learning_rate=0.0001
weight_decay=0.01
batch_size=32
train_epochs=20
patience=5
mlp_hidden_layers=2
mlp_hidden_dim=256
mlp_activation="tanh"

# RPTF specific parameters
pooling_kernels="1 4 8 24 168"
period_list="24"
max_offset_d=96
basis_dim=64
lambda_scale=0.5
eta_entropy=0.01

# Set CUDA device
export CUDA_VISIBLE_DEVICES=0

python run.py \
  --task_name $task \
  --is_training 1 \
  --model_id ${dataset}_RPTF \
  --model $model \
  --data $dataset \
  --root_path $root_path \
  --data_path $data_path \
  --seq_len $seq_len \
  --label_len $label_len \
  --token_len $token_len \
  --test_seq_len $seq_len \
  --test_label_len $label_len \
  --test_pred_len $test_pred_len \
  --llm_ckp_dir $llm_ckp_dir \
  --learning_rate $learning_rate \
  --weight_decay $weight_decay \
  --batch_size $batch_size \
  --train_epochs $train_epochs \
  --patience $patience \
  --mlp_hidden_layers $mlp_hidden_layers \
  --mlp_hidden_dim $mlp_hidden_dim \
  --mlp_activation $mlp_activation \
  --pooling_kernels $pooling_kernels \
  --period_list $period_list \
  --max_offset_d $max_offset_d \
  --basis_dim $basis_dim \
  --lambda_scale $lambda_scale \
  --eta_entropy $eta_entropy \
  --cosine \
  --tmax $train_epochs \
  --gpu 0 \
  --des "rptf_exp"
