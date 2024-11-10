
# install RhoMax: 
```
git clone ?
py -m venv rhoenv
source .\venv\rhoenv\bin\activate
py -m pip install -r requirements.txt
pip install torch-cluster -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch-scatter -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch-scatter torch-sparse -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
pip install torch-spline-conv -f https://data.pyg.org/whl/torch-1.13.0+cpu.html
```


# install foldmason:
```
cd ./cutting/alignments/
wget https://mmseqs.com/foldmason/foldmason-linux-avx2.tar.gz
tar xvzf foldmason-linux-avx2.tar.gz
cp ./run_mason.sh ./foldmason
```

# run RhoMax:
```
source .\venv\rhoenv\bin\activate
python3 run_rhomax.py {pdb_path}
```

The result will be printed, but also saved to ./prediction/result.txt
