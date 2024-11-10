#!/bin/bash
export PATH=$(pwd)/alignments/foldmason/bin/:$PATH
foldmason easy-msa $1 $2 tmpFolder  # 1 - folder, 2 - result



