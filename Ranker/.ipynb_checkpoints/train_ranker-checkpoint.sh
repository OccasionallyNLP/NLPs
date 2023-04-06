#python train_t5.py --output_dir ../../output/ranker/klue_koquard --train_data ../../data/dpr/train_w_negative.jsonl --val_data ../../data/dpr/test_w_negative.jsonl --logging_term 100 --epochs 5 --eval_epoch 1 --batch_size 16 --warmup 200 --ptm_path KETI-AIR/ke-t5-small --context_max_length 512 --patience 2

python train_ranker.py --output_dir ../../output/ranker/list/klue --train_data ../../data/dpr/klue/dev.jsonl --val_data ../../data/dpr/klue/dev.jsonl --logging_term 100 --epochs 5 --eval_epoch 1 --batch_size 1 --warmup 1 --ptm_path KETI-AIR/ke-t5-small --context_max_length 512 --patience 2 --include_title True --n_docs 10 --rank_type list --lr 1e-4