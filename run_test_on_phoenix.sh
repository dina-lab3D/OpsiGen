#! /bin/bash
sbatch --killable --mem=10000m -c1 --time=6:0:0 --gres=gpu:1,vmem:8g /cs/labs/dina/meitar/rhodopsins/run_test.sh
