if [ $# -ne 2 ]; then
	echo "Usage: program <emdb_code> <path>"
else
	cd $2
	wget "https://ftp.ebi.ac.uk/pub/databases/emdb/structures/EMD-$1/map/emd_$1.map.gz"
fi
