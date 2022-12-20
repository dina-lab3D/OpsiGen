#! /bin/bash
cd /cs/labs/dina/meitar/rhodopsins
source /cs/labs/dina/meitar/rhodopsins_venv/bin/activate 
echo "1"
module load python
module load cuda/10.2
module load cudnn/8.0.4
echo $1
echo $2
echo "2"
python -u train.py $1 $2
echo "3"
