# Distributed Experiments LR : 0.0001
python distributed_cred.py distributed \
        --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --non_uniform

# Distributed Experiments LR : 0.001
python distributed_cred.py distributed \
        --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model CNN \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model FF \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model FF \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model FF \
        --non_uniform

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.001 \
        --model FF \
        --non_uniform