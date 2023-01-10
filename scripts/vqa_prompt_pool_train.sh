for i in -1 0 1 2 3 4 5;
do
  PYTHONPATH=$PYTHONPATH:./src \
  CUDA_VISIBLE_DEVICES=1 python src/vqa.py \
        --train karpathy_train \
        --valid karpathy_train \
        --test karpathy_test \
        --optim adamw \
        --warmup_ratio 0.1 \
        --clip_grad_norm 5 \
        --lr 5e-5 \
        --epochs 200 \
        --num_workers 4 \
        --backbone 't5-base' \
        --output /root/autodl-tmp/fewvlm_result/vqa \
        --num_beams 5 \
        --batch_size 16 \
        --valid_batch_size 1000 \
        --load /root/autodl-tmp/fewvlm_result/pretrain_prompt_data_vqa/base/len_2_size_896/Epoch04 \
        --num_data 16 \
        --prompt 3 \
        --subsample \
        --prompt_seq_len 2 \
        --prompt_hidden_size 896 \
        --init_from_pool \
        --dataseed 48 \
        --test_only \
        --choose_pool_key $i
  done