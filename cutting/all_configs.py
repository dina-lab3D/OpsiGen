

class Configs:
    """Base configuration class."""
    all = {
        "config":
            {
                # run_rhomax vars:
                "orig_pdb": "./cutting/cur.pdb",
             "cutting_dir": "./cutting/",
             "pred_dir": "./prediction/",
                # cutting_vars:
             "fasta_path": "./sample_fasta.fasta",
             "wavelength_file": "../excel/wavelength.dat",
             "pdb_path": "./cur.pdb",
             "aligning_index": -1,
             "sequences": "../excel/sequences.fas",
             "cutted_parts_dir": "./cuts/",
             "features": "./features/",
             "cutted_result_pdb_path": "./cuts/cutted_parts0.pdb",
             "feature_maker_script": "./feature_maker.sh",
             "amino_acid_feature_script": "./add_amino_acid_features.sh",
             "graph_maker_script": "./edge_maker.py",
             "edge_dists_path": "./dists/",
             # "aligned_id": "unique_id_for_fm_alignment",
                "aligned_id": "unique_id_for_fm_alignment_try",
             "mafft_id": "unique_id_for_fm_alignment",
                # prediction vars:
                "graph_features_path": "../inoue_graphs/features/",
                "graph_dists_path": "../inoue_graphs/dists/",
                "excel_path": "../excel/data.xlsx",
                "pickle_file": "./model_pickle",
                "train_wildtypes_list": "../excel/splits/train_all",
                "test_wildtypes_list": "../excel/splits/test_all",
                "lr": 0.0001,
                "hidden_layer_size": 40,
                "out_layer_size": 30,
                "graph_th": 2,
                "model_name": "GAT21Model",
                "number_features": 34,
                "dataset_normalize_last": True,
                "test_goal": 9,
                "indexes_to_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                "pickles_folder": "./model_pickles/",
                "output_file": "./result.txt",
                "features_file": "../cutting/features/cutted_parts0.npz",
                "dists_file": "../cutting/dists/cutted_parts0_dists.npy"
            },
        "ablation":
            {
                # run_rhomax vars:
                "orig_pdb": "./cutting/cur.pdb",
                "cutting_dir": "./cutting/",
                "pred_dir": "./prediction/",
                # cutting_vars:
                "fasta_path": "./sample_fasta.fasta",
                "wavelength_file": "../excel/wavelength.dat",
                "aligning_index": -1,
                "sequences": "../excel/sequences.fas",
                "cutted_parts_dir": "./cuts/",
                "features": "./features/",
                "cutted_result_pdb_path": "./cuts/cutted_parts0.pdb",
                "feature_maker_script": "./feature_maker.sh",
                "amino_acid_feature_script": "./add_amino_acid_features.sh",
                "graph_maker_script": "./edge_maker.py",
                "edge_dists_path": "./dists/",
                "aligned_id": "unique_id_for_fm_alignment",
                "mafft_id": "unique_id_for_fm_alignment",
                # prediction vars:
                "graph_features_path": "../inoue_graphs/features/",
                "graph_dists_path": "../inoue_graphs/dists/",
                "excel_path": "../excel/data.xlsx",
                "pickle_file": "./model_pickle",
                "train_wildtypes_list": "../excel/splits/train_all",
                "test_wildtypes_list": "../excel/splits/test_all",
                "lr": 0.0001,
                "hidden_layer_size": 40,
                "out_layer_size": 30,
                "graph_th": 2,
                "model_name": "GAT21Model",
                "number_features": 34,
                "dataset_normalize_last": True,
                "test_goal": 9,
                "indexes_to_keep": [0, 1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 11, 12, 13, 14, 15, 16, 17, 18, 19, 20, 21, 22,
                                    23, 24, 25, 26, 27, 28, 29, 30, 31, 32, 33, 34, 35],
                "pickles_folder": "./model_pickles/",
                "output_file": "./result.txt",
                "features_file": "../cutting/features/cutted_parts0.npz",
                "dists_file": "../cutting/dists/cutted_parts0_dists.npy"
            }
    }


