#! /bin/bash
cd /cs/labs/dina/meitar/rhodopsins
source /cs/labs/dina/meitar/rhodopsins_venv/bin/activate 
export WANDB_CACHE_DIR="/cs/labs/dina/meitar/rhodopsins/wandb/cache"
export WANDB_CONFIG_DIR="/cs/labs/dina/meitar/rhodopsins/wandb/config"
echo "1"
module load python
module load cuda/11.7
module load cudnn/8.4.1
echo $1
echo $2
echo "2"
python -u train.py $1
echo "3"
