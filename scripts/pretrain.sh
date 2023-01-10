output=snap/pretrain

PYTHONPATH=$PYTHONPATH:./src \

export NGPU=$1
python -m torch.distributed.launch \
    --nproc_per_node=$1 \
    src/pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train,vgnococo \
        --valid mscoco_resplit_val \
        --batch_size 512 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --backbone 't5-base' \
        ${@:2} \
        --epoch 30 \
output=snap/pretrain

PYTHONPATH=$PYTHONPATH:./src \

export NGPU=2
python -m torch.distributed.launch \
        --nproc_per_node=2 \
        src/pretrain.py \
        --distributed --multiGPU --fp16 \
        --train mscoco_resplit_train \
        --valid mscoco_resplit_val \
        --batch_size 112 \
        --num_workers 4 \
        --prompt_seq_len 4 \
        --prompt_hidden_size 896 \
        --optim adamw \
        --warmup_ratio 0.05 \
        --lr 1e-4 \
        --num_workers 1 \
        --clip_grad_norm 1.0 \
        --backbone t5-large \
        --epoch 3 \
        --load /root/autodl-tmp/Epoch30_large \
        --coco_only \
        --output /root/autodl-tmp/fewvlm_result/pretrain_prompt \
        --train_prompt_pool
# bash scripts/pretrain.sh 2 