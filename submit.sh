#!/bin/bash
#SBATCH -t 3:00:00
#SBATCH -p GPU
#SBATCH --gpus=v100-32:8 
#SBATCH -N 1

cd "/jet/home/bvw546/MAIN/De-fencing"

module load anaconda3/2020.11
# conda env create --name defencing --file environment.yaml

activate /jet/home/bvw546/.conda/envs/defencing


echo RED20_ADV_128_3e5_551_dlr1_256

python3 train_ADV.py --test_name RED20_ADV_128_3e5_551_dlr1_256 \
--loss_weights 5 5 1 \
--features 128 \
--img_size 256 \
--lr 3e-5 \
--dlr 0.1
> log/slurmOUT_ADV.txt



