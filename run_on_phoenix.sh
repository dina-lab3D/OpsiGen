#! /bin/bash
sbatch --killable --mem=10000m -c1 --time=12:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model0.pkl /cs/labs/dina/meitar/rhodopsins/splits/train0
sbatch --killable --mem=10000m -c1 --time=12:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model1.pkl /cs/labs/dina/meitar/rhodopsins/splits/train1
sbatch --killable --mem=10000m -c1 --time=12:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model2.pkl /cs/labs/dina/meitar/rhodopsins/splits/train2
sbatch --killable --mem=10000m -c1 --time=12:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model3.pkl /cs/labs/dina/meitar/rhodopsins/splits/train3
sbatch --killable --mem=10000m -c1 --time=12:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model4.pkl /cs/labs/dina/meitar/rhodopsins/splits/train4
