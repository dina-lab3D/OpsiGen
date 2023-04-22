#! /bin/bash
# cd /cs/labs/dina/meitar/rhodopsins
source /cs/labs/dina/meitar/rhodopsins_venv/bin/activate 
echo "1"
pip freeze > /cs/labs/dina/meitar/colab_notebook/requirements.txt
python -u calculate_one_rhodopsin.py $1 $2 $3 $4 $5
echo "2"
