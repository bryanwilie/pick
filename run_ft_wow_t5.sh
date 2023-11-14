CUDA_LAUNCH_BLOCKING=1
TOKENIZERS_PARALLELISM=false
CUDA_VISIBLE_DEVICES=3 python finetune.py \
   --model_name_or_path=t5-small \
   --tokenizer_name=t5-small \
   --dataset='wow'\
   --max_seq_len=512 \
   --experiment_mode=tdhkn \
   --dh_uttr_count=3 \
   --use_speaker_token \
   --sep_token="\n" \
   --prompt_resp_sep_token="" \
   --pad_token="!" \
   --preprocessing_num_workers=32 \
   --dataloader_num_workers=32 \
   --per_device_train_batch_size=8 \
   --per_device_eval_batch_size=8 \
   --dataloader_pin_memory \
   --group_by_length \
   --seed=0 \
   --num_train_epochs=10 \
   --learning_rate=8e-5 \
   --fp16 \
   --logging_strategy=steps \
   --logging_steps=5000 \
   --report_to=tensorboard \
   --evaluation_strategy=steps \
   --eval_set="valid_seen" \
   --eval_steps=5000 \
   --eval_accumulation_steps=5000 \
   --save_strategy=steps \
   --save_steps=5000 \
   --save_total_limit=1 \
   --do_train=True \
   --do_eval=True \
   --overwrite_output_dir=True \
   --output_dir="./save/" \
   --early_stopping_patience=3 2>&1 | tee "log/ft_wow_t5-small_uttr3.log"