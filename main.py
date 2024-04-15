from ae_datasets import load_dataset
from ae_datasets import load_dataset_by_attributes
from ae_datasets import get_all_categories
from ae_datasets import random_categories

from utils.logging_utils import init_logging
import logging
import config

def umain(device, dataset_name, batch_size, learning_rate, num_epochs):
    dataset = load_dataset(dataset_name)
    dataset = load_dataset_by_attributes(dataset, config.attributes)

    # print the select dataset and the number of samples(positive and negative)
    # print the head
    # model = 


if __name__ == '__main__':
    init_logging()
    umain(config.device, config.dataset_name, config.batch_size, config.learning_rate, config.num_epochs)
