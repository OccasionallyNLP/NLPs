NGPUS=1
TRAIN_DATA=D:/jupyter_notebook/wiki_data_qa/common_sense_qa.jsonl.jsonl
VAL_DATA=D:/jupyter_notebook/wiki_data_qa/common_sense_qa.jsonl.jsonl
EPOCHS=20
BATCH_SIZE=16
LR=5e-5
WARM_UP=1
FP16=False
DECAY=0.05
INCLUDE_TITLE=True
CONTEXT_MAX_LEN=512
ANSWER_MAX_LEN=64
DISTRIBUTED=False
OUTPUT_DIR=../../output/qa
ACCUMULATION_STEP=16
EARLY_STOP=True
PATIENCE=3
LOGGING_TERM=100
EVAL_EPOCHS=1
PTM_PATH=KETI-AIR/ke-t5-small

python -m torch.distributed.launch --nproc_per_node $NGPUS train.py --output_dir $OUTPUT_DIR --train_data $TRAIN_DATA --val_data $VAL_DATA --include_title $INCLUDE_TITLE --logging_term $LOGGING_TERM --epochs $EPOCHS --eval_epoch $EVAL_EPOCHS --batch_size $BATCH_SIZE --lr $LR --warmup $WARM_UP --decay $DECAY --fp16 $FP16 --accumulation_step $ACCUMULATION_STEP --ptm_path $PTM_PATH --context_max_length $CONTEXT_MAX_LEN --answer_max_length $ANSWER_MAX_LEN --distributed $DISTRIBUTED --early_stop $EARLY_STOP --patience $PATIENCE 