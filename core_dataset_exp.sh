# Distributed Experiments
python distributed_cred.py distributed \
        --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_2

# Distributed Experiments For n_5_b_112
python distributed_cred.py distributed \
        --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model CNN \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 256 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 512 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 1024 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112

python distributed_cred.py distributed --backend nccl \
        --world_size 5 \
        --batch_size 2048 \
        --epochs 30 \
        --learning_rate 0.0001 \
        --model FF \
        --note "Tianbo Dataset Core Run" \
        --use_split n_5_b_112
