#! /bin/bash
sbatch --killable --mem=10000m -c1 --time=23:0:0 --gres=gpu:1,vmem:7g ./run.sh $1 
