#!/usr/bin/env bash
echo $1,$2
python3  -u main.py --dataset=$1 --optimizer='fedavg'  \
            --learning_rate=0.01 --num_rounds=200 --clients_per_round=10 \
            --eval_every=1 --batch_size=10 \
            --num_epochs=20 \
            --model='mclr' \
            --drop_percent=$2 \

