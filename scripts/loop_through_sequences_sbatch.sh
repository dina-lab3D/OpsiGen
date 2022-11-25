#! /bin/bash
BLAST_BIN='/cs/usr/tomer.cohen13/lab/Blast/bin/blastp'
FASTA_FOLDER='/cs/labs/dina/meitar/rhodopsins/fastas/'
echo $FASTA_FOLDER
for d in /cs/labs/dina/meitar/rhodopsins/fastas/*.fasta
do
	echo $d
	sbatch --time=1-0 /cs/labs/dina/meitar/rhodopsins/pdb_validator/fetch_pdb_from_net.sh $d
done
