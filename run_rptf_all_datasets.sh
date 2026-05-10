#!/bin/bash

# Run RPTF model on all benchmark datasets from the paper
# Datasets: ETTh1, ETTh2, ETTm1, ETTm2, ECL, Weather, Traffic, Solar

# Common hyperparameters (from paper Section 4.3)
LLM_CKPT="./llama"
LR=0.0001
WD=0.01
BS=32
EPOCHS=20
PATIENCE=5
SEQ_LEN=672
LABEL_LEN=576
TOKEN_LEN=96

# RPTF specific parameters
LAMBDA=0.5
ETA=0.01
MAX_OFFSET=96
BASIS_DIM=64

# Prediction horizons
HORIZONS=(96 192 336 720)

# Dataset configurations
declare -A DATASETS
declare -A ROOT_PATHS
declare -A DATA_PATHS
declare -A PERIODS

DATASETS=(
    ["ETTh1"]="ETTh1"
    ["ETTh2"]="ETTh2"
    ["ETTm1"]="ETTm1"
    ["ETTm2"]="ETTm2"
    ["ECL"]="custom"
    ["Weather"]="custom"
    ["Traffic"]="custom"
    ["Solar"]="Solar"
)

ROOT_PATHS=(
    ["ETTh1"]="./data/ETT/"
    ["ETTh2"]="./data/ETT/"
    ["ETTm1"]="./data/ETT/"
    ["ETTm2"]="./data/ETT/"
    ["ECL"]="./data/electricity/"
    ["Weather"]="./data/weather/"
    ["Traffic"]="./data/traffic/"
    ["Solar"]="./data/solar/"
)

DATA_PATHS=(
    ["ETTh1"]="ETTh1.csv"
    ["ETTh2"]="ETTh2.csv"
    ["ETTm1"]="ETTm1.csv"
    ["ETTm2"]="ETTm2.csv"
    ["ECL"]="electricity.csv"
    ["Weather"]="weather.csv"
    ["Traffic"]="traffic.csv"
    ["Solar"]="solar_AL.txt"
)

PERIODS=(
    ["ETTh1"]="24"
    ["ETTh2"]="24"
    ["ETTm1"]="96"
    ["ETTm2"]="96"
    ["ECL"]="24"
    ["Weather"]="144"
    ["Traffic"]="24"
    ["Solar"]="144"
)

# Function to run experiment
run_exp() {
    local dataset=$1
    local pred_len=$2

    echo "========================================"
    echo "Running $dataset with pred_len=$pred_len"
    echo "========================================"

    local data=${DATASETS[$dataset]}
    local root=${ROOT_PATHS[$dataset]}
    local dpath=${DATA_PATHS[$dataset]}
    local period=${PERIODS[$dataset]}

    python run.py \
        --task_name long_term_forecast \
        --is_training 1 \
        --model_id ${dataset}_RPTF_il${SEQ_LEN}_pl${pred_len} \
        --model RPTF \
        --data $data \
        --root_path $root \
        --data_path $dpath \
        --seq_len $SEQ_LEN \
        --label_len $LABEL_LEN \
        --token_len $TOKEN_LEN \
        --test_seq_len $SEQ_LEN \
        --test_label_len $LABEL_LEN \
        --test_pred_len $pred_len \
        --llm_ckp_dir $LLM_CKPT \
        --learning_rate $LR \
        --weight_decay $WD \
        --batch_size $BS \
        --train_epochs $EPOCHS \
        --patience $PATIENCE \
        --mlp_hidden_layers 2 \
        --mlp_hidden_dim 256 \
        --mlp_activation tanh \
        --pooling_kernels 1 4 8 24 168 \
        --period_list $period \
        --max_offset_d $MAX_OFFSET \
        --basis_dim $BASIS_DIM \
        --lambda_scale $LAMBDA \
        --eta_entropy $ETA \
        --cosine \
        --tmax $EPOCHS \
        --gpu 0 \
        --des "rptf_${dataset}_pl${pred_len}"
}

# Run all experiments
for dataset in ETTh1 ETTh2 ETTm1 ETTm2 ECL Weather Traffic Solar; do
    for pl in "${HORIZONS[@]}"; do
        run_exp $dataset $pl
    done
done

echo "All experiments completed!"
