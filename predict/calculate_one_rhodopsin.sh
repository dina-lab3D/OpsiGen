#! /bin/bash
# cd /cs/labs/dina/meitar/rhodopsins
source /cs/labs/dina/meitar/rhodopsins_venv/bin/activate 
# echo "1"
# module load python
# module load cuda/10.2
# module load cudnn/8.0.4
# echo $1
# echo $2
# echo "2"
pip freeze > /cs/labs/dina/meitar/colab_notebook/requirements.txt
python -u calculate_one_rhodopsin.py $1 $2 $3
# echo "3"
