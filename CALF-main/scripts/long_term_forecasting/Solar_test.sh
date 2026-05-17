# 切换到项目根目录（脚本位于 CALF-main/scripts/long_term_forecasting/）
cd "$(dirname "$0")/../.."

GPU=0
model=CALF
data_name=Solar
logs_dir="logs/$model/$data_name"

if [ ! -d "./logs" ]; then mkdir ./logs; fi
if [ ! -d "./logs/$model" ]; then mkdir ./logs/$model; fi
if [ ! -d "$logs_dir" ]; then mkdir "$logs_dir"; fi

# ========== 手动参数配置区 ==========
seq_len=96
pred_lens=" 192 336 720"
batch_size=32
learning_rate=0.0002
train_epochs=10
d_model=768
n_heads=4
d_ff=$((4 * d_model))
dropout=0.2
enc_in=137
c_out=137
gpt_layers=3

r=8
lora_alpha=32
lora_dropout=0.1

feature_w=0.1
output_w=0.2
task_w=0.7

random_seed=2025
cycle=144
# ====================================

for pred_len in $pred_lens; do

log_file="$logs_dir/${model}_${seq_len}_${pred_len}_dff${d_ff}_r${r}_fw${feature_w}_ow${output_w}_${random_seed}.logs"

CUDA_VISIBLE_DEVICES=$GPU \
python -u run.py \
  --num_workers 8 \
  --root_path ./dataset/Solar/ \
  --data_path solar_AL.txt \
  --is_training 1 \
  --task_name long_term_forecast \
  --model_id "${model}_${seq_len}_${pred_len}_fw${feature_w}_ow${output_w}_r${r}_dff${d_ff}" \
  --data Solar \
  --seq_len $seq_len \
  --label_len 0 \
  --pred_len $pred_len \
  --batch_size $batch_size \
  --learning_rate $learning_rate \
  --train_epochs $train_epochs \
  --d_model $d_model \
  --n_heads $n_heads \
  --d_ff $d_ff \
  --dropout $dropout \
  --enc_in $enc_in \
  --c_out $c_out \
  --gpt_layers $gpt_layers \
  --itr 1 \
  --model $model \
  --cos 1 \
  --tmax 10 \
  --r $r \
  --lora_alpha $lora_alpha \
  --lora_dropout $lora_dropout \
  --patience 3 \
  --bestmodel \
  --task_w $task_w \
  --task_loss smooth_l1 \
  --feature_loss smooth_l1 \
  --output_loss smooth_l1 \
  --feature_w $feature_w \
  --output_w $output_w \
  --random_seed $random_seed \
  --cycle $cycle \
  --use_amp \
  2>&1 | tee "$log_file"

done