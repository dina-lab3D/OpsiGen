#! /bin/bash
cd /cs/labs/dina/meitar/rhodopsins
source /cs/labs/dina/meitar/rhodopsins_venv/bin/activate 
echo "1"
module load python
module load cuda/11.4
module load cudnn/8.2.2
echo "2"
python -u excel_parser.py
echo "3"
