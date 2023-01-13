model="t5-base"
data_path=./data/
ckpt_path=./ckpt/
rm -rf $ckpt_path"_"$model
mkdir -p $ckpt_path

CUDA_VISIBLE_DEVICES=7 \
python3 t5_summarization.py --model=$model \
                  --dropout 0.1 \
                  --ctx_len 500 \
                  --tgt_len 150 \
                  --beam 2 \
                  --train_data $data_path/news_summary.csv \
                  --batch_size 4 \
                  --lr 1e-4 \
                  --epoch 50 \
                  --gpuid 0 \
                  --gradient_accumulation 5 \
                  --seed 1024 \
                  --print_every 50 \
                  --save_every 10000 \
                  --save_dir $ckpt_path \
