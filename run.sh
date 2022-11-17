#! /bin/bash
cd /cs/labs/dina/meitar/rhodopsins
source /cs/labs/dina/meitar/rhodopsins_venv/bin/activate 
echo "1"
module load python
echo "2"
python -u train.py
echo "3"
