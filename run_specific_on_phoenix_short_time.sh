#! /bin/bash
for i in {1..30}
do
	sbatch --killable --mem=10000m -c1 --time=5:0:0 --gres=gpu:1,vmem:7g ./run.sh $1
done
