export CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7

current_dir="$( cd "$( dirname "${BASH_SOURCE[0]}" )" >/dev/null 2>&1 && pwd )"

basepath=/path/to/data/

dataset=reddit-dual-6M-sem-roberta
datapath=$basepath/$dataset

MODEL=$basepath/roberta-base/
AMRMODEL=$basepath/roberta-base/

lr=1e-5
amr_lr=1e-5
mlm_rate=0.15
ratio1=1.0			# sent-amr agreement
ratio2=0.1			# rel prediction
#ratio3=1.0			# amrseq generation
temperature=1.0

outpath=$basepath/output/${dataset}-base-${lr}-SemMLM-${mlm_rate}-dualConCos-${ratio1}-temp-$temperature-rel-${ratio2}-bsz2048

cache=$datapath/.cache
dump_dir=$cache/dump

export HF_DATASETS_CACHE=$cache 

ls -lh $MODEL
mkdir -p $outpath
mkdir -p $dump_dir

python -u main.py \
  --train_data_file $datapath/train.jsonl \
  --eval_data_file $datapath/val.jsonl \
  --test_data_file $datapath/test.jsonl \
  --output_dir $outpath \
  --model_type "roberta" \
  --tokenizer_name $MODEL \
  --gpus 8 \
  --sem_mlm \
  --mlm_probability $mlm_rate \
  --use_amr \
  --cross_ratio $ratio1 \
  --use_rel \
  --rel_ratio $ratio2 \
  --temperature $temperature \
  --per_gpu_train_batch_size 8 \
  --per_gpu_eval_batch_size 2 \
  --accumulate_grad_batches 32 \
  --model_name_or_path $MODEL \
  --amr_model_name_or_path $AMRMODEL \
  --cache_dir $dump_dir \
  --do_train \
  --do_eval \
  --evaluate_during_training  \
  --src_block_size 512 \
  --tgt_block_size 512 \
  --max_epochs 5 \
  --save_total_limit 1 \
  --save_interval -1 \
  --val_check_interval 0.5 \
  --early_stopping_patience 50 \
  --learning_rate $lr \
  --amr_learning_rate $amr_lr \
  --log_every_n_steps 500 \
  --train_num_workers 8 \
  --eval_num_workers 8 \
  --process_num_workers 32 \
  --fp16 \
  --overwrite_output_dir 2>&1 | tee $outpath/run.log
