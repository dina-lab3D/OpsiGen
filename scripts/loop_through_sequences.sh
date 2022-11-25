#! /bin/bash
BLAST_BIN='/cs/usr/tomer.cohen13/lab/Blast/bin/blastp'
FASTA_FOLDER='/cs/labs/dina/meitar/rhodopsins/fastas/'
echo $FASTA_FOLDER
cat failed_files.txt | while read line || [[ -n $line ]];
do
	# do something with $line here
	file_name=/cs/labs/dina/meitar/rhodopsins/fastas/$line
	cat $file_name
	/cs/labs/dina/meitar/rhodopsins/pdb_validator/fetch_pdb_from_net.sh $file_name
done
# for d in /cs/labs/dina/meitar/rhodopsins/fastas/*.fasta
# do
	# echo $d
	# /cs/labs/dina/meitar/rhodopsins/pdb_validator/fetch_pdb_from_net.sh $d
# done
