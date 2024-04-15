import pandas
import logging
import json
import random
from torch.utils.data import DataLoader, Dataset
import torch
from multiprocessing import Pool
from typing import Union
from utils.io import pandas_deserialize
from utils.io import to_hf_dataset
# from utils.pos import extract_noun_phrases


def random_categories(dataset:pandas.DataFrame, num_of_category, min_sample=100):
    all_categories = get_all_categories(dataset)
    # first filter out the categories that satisfy the min_sample
    all_categories = [c for c in all_categories if len(dataset[dataset['attribute'] == c]) >= min_sample]
    # random select num_of_category categories with data, return a dict
    random_categories =  random.sample(all_categories, num_of_category)
    res = {}
    for c in random_categories:
        res[c] = dataset[dataset['attribute'] == c]
    return res


'''
print ae-100k dataset information
'''
def print_dataset_info(data: pandas.DataFrame):
    print(f"Number of samples: {len(data)}")
    print(f"Number of attributes: {len(data['attribute'].unique())}")
    print(f"Number of values: {len(data['value'].unique())}")

    print("Top 10 attributes ")
    print("Attribute | #SampleNumber |#Negative Samples|#Positive samples")
    for attribute in data['attribute'].value_counts().index[:10]:
        print(f"{attribute} | {len(data[data['attribute'] == attribute])} | {len(data[(data['attribute'] == attribute) & (data['is_negative'])])} | {len(data[(data['attribute'] == attribute) & (~data['is_negative'])])}")


def load_dataset_by_attributes(dataset: str, attributes: list):
    # assure there is no duplicate attributes
    raw_dataset = load_dataset(dataset)
    assert (len(attributes) == len(set(attributes)))

    res = {}

    # group by attribute
    data_grouped = raw_dataset.groupby('attribute')

    for attribute in attributes:
        if attribute in data_grouped.groups:
            logging.info(f"loaded attribute: {attribute}, number of values: {len(data_grouped.groups[attribute])}")
            frame = data_grouped.get_group(attribute)
            res[attribute] = frame
        else:
            logging.error(f"can not find attribute: {attribute}")
            raise ValueError(f"can not find attribute: {attribute}")
    return res

def load_mave():
    is_negative = False
    res = []
    with open('./MAVE/reproduce/mave_positives.jsonl') as positive_f:
        positive_lines = positive_f.readlines()
        # multiprocessing

        for line in positive_lines:
            sample = json.loads(line)
            paragraphs = sample['paragraphs']
            attributes = sample['attributes']
            for attribute in attributes:
                attr = attribute['key'] # attribute
                vals = attribute['evidences']
                for val in vals:
                    v = val['value'] # value
                    p_idx = val['pid']
                    context = paragraphs[p_idx] # context
                    _b_idx = val['begin']
                    _e_idx = val['end']
                    res.append([context, attr, v, is_negative])

    is_negative = True
    with open('./MAVE/reproduce/mave_negatives.jsonl') as negative_f:
        negative_lines = negative_f.readlines()
        for line in negative_lines:
            sample = json.loads(line)
            paragraphs = sample['paragraphs']
            attributes = sample['attributes']
            for attribute in attributes:
                attr = attribute['key'] # attribute
                vals = attribute['evidences'] # it will be empty in the negative sample

                for para in paragraphs:
                    context = para['text']
                    v = ''
                    res.append([context, attr, v, is_negative])
    return pandas.DataFrame(res, columns=['context', 'attribute', 'value', 'is_negative'])

def load_ae_100k():
    with open('AE-100K/publish_data.txt') as f:
        data_lines = f.readlines()
        # pandas dataframe
        data = pandas.DataFrame([line.split('\u0001') for line in data_lines])
        data[2] = data[2].str.strip()
        data[1] = data[1].str.strip()
        data.columns = ['context', 'attribute', 'value']
        data['is_negative'] = data['value'].apply(lambda x: x == 'NULL')
    filtered_rows = []
    for _, row in data.iterrows():
        if row['value'] in row['context']:
            filtered_rows.append(row)

    data = pandas.DataFrame(filtered_rows)
    data = data.reset_index()
    data.rename(columns={'index': 'ID'}, inplace=True)
    return data

def _process_positive_line(line):
    sample = json.loads(line)
    paragraphs = sample['paragraphs']
    attributes = sample['attributes']
    result = []
    for attribute in attributes:
        attr = attribute['key']  # attribute
        vals = attribute['evidences']
        for val in vals:
            v = val['value']  # value
            p_idx = val['pid']
            context = paragraphs[p_idx]  # context
            _b_idx = val['begin']
            _e_idx = val['end']
            result.append([context, attr, v, False])
    return result

def _process_negative_line(line):
    sample = json.loads(line)
    paragraphs = sample['paragraphs']
    attributes = sample['attributes']
    result = []
    for attribute in attributes:
        attr = attribute['key']  # attribute
        # In negative samples, 'evidences' will be empty
        for para in paragraphs:
            context = para['text']
            v = ''
            result.append([context, attr, v, True])
    return result

def load_mave():
    with open('./MAVE/reproduce/mave_positives.jsonl') as positive_f:
        positive_lines = positive_f.readlines()

    with open('./MAVE/reproduce/mave_negatives.jsonl') as negative_f:
        negative_lines = negative_f.readlines()

    pool = Pool()  # Initialize multiprocessing Pool

    positive_results = pool.map(_process_positive_line, positive_lines)
    negative_results = pool.map(_process_negative_line, negative_lines)

    pool.close()
    pool.join()

    # Flatten the list of lists
    res = [item for sublist in positive_results for item in sublist]
    res += [item for sublist in negative_results for item in sublist]

    df = pandas.DataFrame(res, columns=['context', 'attribute', 'value', 'is_negative'])
    return df

def load_dataset(dataset: str, attributes=None) -> pandas.DataFrame:
    if dataset == 'ae-100k':
        return load_ae_100k()
    elif dataset == 'mave':
        return load_mave()
    elif dataset == 'oa-mine':
        pass
        # return load_oa_mine(split)
    else:
        raise ValueError(f"Unknown dataset: {dataset}")

def get_all_categories(dataset: pandas.DataFrame):
    # return a list of unique attributes, if none return []
    return list(dataset['attribute'].unique())


if __name__ == '__main__':
    import config
    dataset = load_dataset_by_attributes('ae-100k', attributes=config.attributes)
    for k in dataset:
        print_dataset_info(dataset[k])
