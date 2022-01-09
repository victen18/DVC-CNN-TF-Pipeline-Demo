import os
import yaml
import logging
import time
import pandas as pd
import json
import shutil
from tqdm import tqdm 


def read_yaml(path_to_yaml: str) -> dict:
    with open(path_to_yaml) as yaml_file:
        content = yaml.safe_load(yaml_file)
    logging.info(f"yaml file: {path_to_yaml} loaded successfully")
    return content


def create_directories(path_to_directories: list) -> None:
    for path in path_to_directories:
        os.makedirs(path, exist_ok=True)
        logging.info(f"created directory at: {path}")


def copy_files(source_data_dir: str, local_data_dir: str) -> None:
    list_of_files = os.listdir(source_data_dir)
    N = len(list_of_files)

    for file in tqdm(
        list_of_files,
        total=N,
        desc=f"copying file from {source_data_dir} to {local_data_dir}",
        colour="green",
    ):
        src = os.path.join(source_data_dir, file)
        dest = os.path.join(local_data_dir, file)

        shutil.copy(src, dest)

    logging.info(f"all the files has been copied from {source_data_dir} to {local_data_dir}")


def save_json(path: str, data: dict) -> None:
    with open(path, "w") as f:
        json.dump(data, f, indent=4)

    logging.info(f"json file saved at: {path}")


def get_timestamp(name: str) -> str:
    timestamp = time.asctime().replace(" ","_").replace(":",".")
    unique_name = f"{name}_at_{timestamp}"
    return unique_name
