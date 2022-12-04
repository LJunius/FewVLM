for prompt_seq_len in 2 4 6 8 10 12 16 20;
do # 48 64 96 128 196 256 512
  for prompt_hidden_size in 0;
  do
    echo $prompt_seq_len, $prompt_hidden_size
    CUDA_VISIBLE_DEVICES=0 python3 -u /root/FewVLM/src/flickr.py --train karpathy_train \
      --valid karpathy_train \
      --test karpathy_test \
      --optim adamw \
      --warmup_ratio 0.1 \
      --clip_grad_norm 5 \
      --lr 5e-5 \
      --epochs 50 \
      --num_workers 4 \
      --backbone t5-base \
      --output /root/autodl-tmp/fewvlm_result/flickr30k \
      --load /root/autodl-tmp/Epoch30_base \
      --num_beams 5 \
      --batch_size 16 \
      --valid_batch_size 100 \
      --caption_data dataset_flickr30k \
      --subsample \
      --dataseed 42 \
      --num_data 16 \
      --prefix image \
      --prompt_seq_len $prompt_seq_len \
      --prompt_hidden_size $prompt_hidden_size
  done
done