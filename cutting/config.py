import json
import argparse
import os


def validate_config(config):
    assert os.path.exists(config["cutted_parts_dir"]), "cutted parts dir path: " + config["cutted_parts_dir"] +\
                                                       ", current folder: " + os.getcwd()

def parse_args():
    # parser = argparse.ArgumentParser()
    # parser.add_argument("config_file")

    # config = parser.parse_args().config_file
    config = "./configs/config.json"
    with open(config, "r") as f:
        data = f.read()

    config = json.loads(data)
    validate_config(config)

    return config
