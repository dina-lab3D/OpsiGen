#! /bin/bash
sbatch --killable --mem=10000m -c1 --time=19:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model0_$1.pkl /cs/labs/dina/meitar/rhodopsins/splits/train0 /cs/labs/dina/meitar/rhodopsins/splits/test0
sbatch --killable --mem=10000m -c1 --time=19:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model1_$1.pkl /cs/labs/dina/meitar/rhodopsins/splits/train1 /cs/labs/dina/meitar/rhodopsins/splits/test1
sbatch --killable --mem=10000m -c1 --time=19:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model2_$1.pkl /cs/labs/dina/meitar/rhodopsins/splits/train2 /cs/labs/dina/meitar/rhodopsins/splits/test2
sbatch --killable --mem=10000m -c1 --time=19:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model3_$1.pkl /cs/labs/dina/meitar/rhodopsins/splits/train3 /cs/labs/dina/meitar/rhodopsins/splits/test3
sbatch --killable --mem=10000m -c1 --time=19:0:0 --gres=gpu:1,vmem:7g /cs/labs/dina/meitar/rhodopsins/run.sh /cs/labs/dina/meitar/rhodopsins/pickles/model4_$1.pkl /cs/labs/dina/meitar/rhodopsins/splits/train4 /cs/labs/dina/meitar/rhodopsins/splits/test4
