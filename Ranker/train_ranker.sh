python train_ranker.py --output_dir ../../output/ranker/question_filtering/point --train_data D:/jupyter_notebook/data/q_filtering_data/dev.jsonl --val_data D:/jupyter_notebook/data/q_filtering_data/dev.jsonl --logging_term 100 --epochs 5 --eval_epoch 1 --batch_size 16 --warmup 1 --ptm_path KETI-AIR/ke-t5-small --context_max_length 1024 --patience 1 --include_title True --rank_type point --lr 1e-4 --weighted_sampling True --distributed False --fp16 False

