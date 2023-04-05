import json
import argparse
import os

def validate_config(config):
    assert os.path.exists(config["cutted_parts_dir"])

def parse_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("config_file")

    config = parser.parse_args().config_file

    with open(config, "r") as f:
        data = f.read()

    config = json.loads(data)
    validate_config(config)

    return config
