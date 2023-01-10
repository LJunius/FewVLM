for i in -1 0 1 2 3 4 5;
do
  CUDA_VISIBLE_DEVICES=0 python3 -u /root/FewVLM/src/flickr.py --train karpathy_train \
      --valid karpathy_train \
      --test karpathy_test \
      --optim adamw \
      --warmup_ratio 0.1 \
      --clip_grad_norm 5 \
      --lr 5e-5 \
      --epochs 20 \
      --num_workers 4 \
      --backbone t5-base \
      --output /root/autodl-tmp/fewvlm_result/flickr30k/2896 \
      --load /root/autodl-tmp/fewvlm_result/pretrain_prompt_data_all_dist/base/len_2_size_896/Epoch10 \
      --num_beams 5 \
      --batch_size 16 \
      --valid_batch_size 100 \
      --caption_data dataset_flickr30k \
      --subsample \
      --dataseed 42 \
      --num_data 16 \
      --prefix image \
      --prompt_seq_len 2 \
      --prompt_hidden_size 896 \
      --init_from_pool \
      --choose_pool_key $i
#      --test_only
  done

