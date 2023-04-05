import numpy as np
import pandas as pd
import json

CSV_PATH = "/cs/labs/dina/meitar/ionet-meitar/Interface_grid/add_amino_acid_features/amino_acid_feature.csv"

def main():
    df = pd.read_csv(CSV_PATH)
    amino_features = df.iloc[:,1:].to_numpy()
    amino_acid_names = df.iloc[:,0].tolist()
    mapping_dict = {amino_acid_names[i]: list(amino_features[i]) for i in range(len(amino_features))}
    vals = np.array(list(mapping_dict.values()))
    max_points = np.max(np.abs(vals), axis=0)
    for key in mapping_dict:
        print(len(mapping_dict[key]))
        for i in range(len(mapping_dict[key])):
            mapping_dict[key][i] = mapping_dict[key][i] / max_points[i]

    vals = np.array(list(mapping_dict.values()))
    print(mapping_dict)
    with open("amino_mapping", "w") as f:
        f.write(json.dumps(mapping_dict))
    print(mapping_dict)

if __name__ == "__main__":
    main()
