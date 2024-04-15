import json
import pandas as pd
import os
from datasets import Dataset

def dump2json(obj, file_name):
    with open(file_name, 'w') as f:
        json.dump(obj, f, indent=4)

def pandas_serialize(obj: pd.DataFrame, file_name: str):
    obj.to_pickle(file_name)

def pandas_deserialize(file_name) -> pd.DataFrame:
    return pd.read_pickle(file_name)

def to_hf_dataset(obj: pd.DataFrame):
    return Dataset.from_pandas(obj)

def rows_to_hf_dataset(obj: list):
    return Dataset.from_list(obj)

def save_dataframes_to_csv(data_dict, directory):
    import os
    if not os.path.exists(directory):
        os.makedirs(directory)

    path_dict = {}

    for key, df in data_dict.items():
        filename = f"{key}.csv"
        df.to_csv(os.path.join(directory, filename), index=False)
        path_dict[key] = filename
    # Optionally, save the path_dict as well to easily reload the DataFrames
    with open(os.path.join(directory, 'metadata.json'), 'w') as f:
        import json
        json.dump(path_dict, f)

def load_dataframes_from_csv(directory):
    with open(os.path.join(directory, 'metadata.json'), 'r') as f:
        path_dict = json.load(f)
        data_dict = {}
        for key, filename in path_dict.items():
            data_dict[key] = pd.read_csv(os.path.join(directory, filename))
    return data_dict
