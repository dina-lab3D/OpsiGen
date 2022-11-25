#! /bin/bash
BLAST_BIN='/cs/usr/tomer.cohen13/lab/Blast/bin/blastp'
FASTA_FOLDER='/cs/labs/dina/meitar/rhodopsins/fastas/'
d=$1
output_file=${d//.fasta/.seq}
output_file=${output_file//fastas/pdb_validator}
# echo $output_file
$BLAST_BIN -query $d -db pdb -max_target_seqs 10 -out $output_file -remote
