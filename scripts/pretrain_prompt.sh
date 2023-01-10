len_array=(2 4 6)
hidden_size_array=(896 896 96)
size_array=(base large)
batch_size_array=(400 112)
for j in 0;
  do
    for i in 0;
      do
      size=${size_array[j]}
      batch_size=${batch_size_array[j]}
      prompt_seq_len=${len_array[i]}
      prompt_hidden_size=${hidden_size_array[i]}
      echo $prompt_seq_len $prompt_hidden_size $size
      output_dir=/root/autodl-tmp/fewvlm_result/pretrain_prompt_data_vqa/"$size"/len_"$prompt_seq_len"_size_"$prompt_hidden_size"
      load_dir=/root/autodl-tmp/Epoch30_"$size"
      backbone=t5-"$size"
      echo $output_dir $load_dir $backbone
      mkdir -p $output_dir

      PYTHONPATH=$PYTHONPATH:./src \
      export NGPU=2
      python -m torch.distributed.launch \
              --nproc_per_node=2 \
              src/pretrain.py \
              --distributed --multiGPU --fp16 \
              --train vgnococo \
              --valid mscoco_resplit_val \
              --batch_size $batch_size \
              --num_workers 8 \
              --prompt_seq_len $prompt_seq_len \
              --prompt_hidden_size $prompt_hidden_size \
              --optim adamw \
              --warmup_ratio 0.05 \
              --lr 1e-4 \
              --clip_grad_norm 1.0 \
              --backbone $backbone \
              --epoch 4 \
              --load $load_dir \
              --output $output_dir \
              --train_prompt_pool
        done
done