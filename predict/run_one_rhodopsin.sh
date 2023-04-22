#! /bin/bash
sbatch --killable --mem=15000m -c1 --time=6:0:0 --gres=gpu:1,vmem:7g ./calculate_one_rhodopsin.sh $1 $2 $3 $4 $5
