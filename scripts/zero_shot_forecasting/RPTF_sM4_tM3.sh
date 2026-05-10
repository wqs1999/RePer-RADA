model_name=RPTF_Llama

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 1 \
  --root_path ./dataset/m4 \
  --data_path m4_Monthly \
  --seasonal_patterns 'Monthly' \
  --model_id m4_Monthly_672_96 \
  --model $model_name \
  --data m4 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 24 \
  --batch_size 512 \
  --learning_rate 0.001 \
  --mlp_hidden_layers 0 \
  --train_epochs 5 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10

python -u run.py \
  --task_name zero_shot_forecast \
  --is_training 0 \
  --root_path ./dataset/m4 \
  --data_path m4_Monthly \
  --test_data_path ./dataset/m3 \
  --model_id m4_Monthly_672_96 \
  --model $model_name \
  --data m4 \
  --seq_len 672 \
  --label_len 576 \
  --token_len 96 \
  --test_seq_len 672 \
  --test_label_len 576 \
  --test_pred_len 18 \
  --batch_size 256 \
  --learning_rate 0.001 \
  --mlp_hidden_layers 0 \
  --train_epochs 5 \
  --use_amp \
  --gpu 0 \
  --cosine \
  --tmax 10 \
  --test_dir zero_shot_forecast_m4_Monthly_672_96_RPTF_Llama_m4_sl672_ll576_tl96_lr0.001_bt512_wd0_hd256_hl0_cosTrue_mixFalse_test_0
