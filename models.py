from datasets import load_metric
from transformers import AutoModelForMaskedLM, AutoTokenizer
from transformers import TrainingArguments
from peft import get_peft_model, PromptTuningInit, PromptTuningConfig, TaskType
from ae_datasets import load_dataset_by_attributes
from utils.pos import extract_noun_phrases, PosStrategy
from utils.io import dump2json, rows_to_hf_dataset,save_dataframes_to_csv, load_dataframes_from_csv 
from utils.labels import get_labels, get_indices
from utils.logging_utils import init_logging
from utils.data import random_split
from utils.prompt import PromptInitStrategy, get_prompt_init_text
from torch.utils.data import DataLoader
from string import Template as Template
from logging import warning, info, error
from datetime import datetime
from pandas import DataFrame
from utils.prompt import get_mask_idx
from typing import List, Dict
import unittest
import config
import torch
import trainer


class ModelForAE():
    def __init__(self, 
                 prompt_length: int,
                 base_model: str,
                 dataset: str,
                 attributes: list,
                 hard_prompt_template: Template,
                 pos_strategy: PosStrategy,
                 shots: List[int],
                 prompt_init_strategy: PromptInitStrategy,
                 none_label: str,
                 metrics: List[str],
                 check_point_name=None,
                 load_dataset_from_file=None,
                 dump_dataset_to_file=None,
                 ):
        self.base_model_name = base_model

        # set tokenizer and model
        self.tokenizer = AutoTokenizer.from_pretrained(base_model)
        if self.tokenizer.pad_token_id is None:
            self.tokenizer.pad_token_id = self.tokenizer.eos_token_id

        # set peft_config
        self.prompt_init_text = get_prompt_init_text(prompt_init_strategy, attributes)
        info(f"prompt init text: {self.prompt_init_text}")
        info(f"prompt init stategy: {prompt_init_strategy}")
        self.peft_config = PromptTuningConfig(
            task_type=TaskType.SEQ_CLS,
            prompt_tuning_init=PromptTuningInit.TEXT,
            num_virtual_tokens=len(self.tokenizer(self.prompt_init_text)["input_ids"]),
            prompt_tuning_init_text=self.prompt_init_text,
            tokenizer_name_or_path=base_model,
        )
        self.prompt_offset = len(self.tokenizer(self.prompt_init_text)["input_ids"])

        self.base_model = AutoModelForMaskedLM.from_pretrained(base_model)
        self.model = get_peft_model(self.base_model, self.peft_config)
        info(f'chosen base model:{self.base_model_name}')
        info(f'peft config:{self.peft_config}')
        # info(f'trainable parameters:{self.model.print_trainable_parameters()}')
        
        # check_point_name
        if check_point_name is None:
            self.check_point_name = f"{dataset}_{base_model}__v1.pt"

        # preprocess the dataset
        self.load_dataset(
            dataset,
            attributes,
            load_dataset_from_file,
            dump_dataset_to_file
        )
            
        self.none_label = none_label
        self.label_info = get_labels(attributes, self.base_model_name, none_label=none_label)
        self._get_metric(metrics)
        self._preprocess_dataset(
            self.dataset,
            pos_strategy,
            hard_prompt_template,
            shots,
            none_label
        )
        self.K_shots = shots

    def load_dataset(self, dataset, attributes, load_dataset_from_file, dump_dataset_to_file):
        if load_dataset_from_file is not None:
            self.dataset = load_dataframes_from_csv(load_dataset_from_file)
        else:
            self.dataset = load_dataset_by_attributes(dataset, attributes)

        if dump_dataset_to_file is not None:
            dump_dataset_to_file(dump_dataset_to_file)


    def eval(self):
        pass

    def collate_fn(self, examples):
        return self.tokenizer.pad(examples, padding="longest", return_tensors="pt")

    def preprocess_function(self, examples, text_column='input', label_column='label'):
        # TODO: change here to add multiple attribute extraction
        label2idx = self.label_info["label2vec"]
        labels_ids = [label2idx[label] for label in examples[label_column]]
        outputs = self.tokenizer(examples[text_column])
        outputs["labels"] = labels_ids
        return outputs

    def _get_metric(self, metrics: List[str]):
        metrics_info = {}
        for metric in metrics:
            metrics_info[metric] = load_metric(metric)
        self.metrics_info = metrics_info

    def _get_dataset(self, dataset, top_n=100):
        dataset = rows_to_hf_dataset(dataset)
        maped_dataset = dataset.map(self.preprocess_function,
                                    batched=True,
                                    load_from_cache_file=True,
                                    num_proc=4,
                                    remove_columns=['input', 'sentence_id', 'label'],
                                    desc='map the dataset')
        return maped_dataset
    
    def _get_dataloader(self, dataset, batch_size, top_n=100):
        maped_dataset = self._get_dataset(dataset, batch_size)
        return DataLoader(
            maped_dataset,
            shuffle=True,
            collate_fn=self.collate_fn,
            batch_size=batch_size,
            pin_memory=True
        )
        
    def seperate_train(self, lr, batch_size, epochs, K, optimizer=torch.optim.AdamW, device='cpu'):
        # init traindataset
        self.model = self.model.to(device)

        # dataset preprocess
        K_dataset = self.dataset[str(K)]
        K_dataset_train = K_dataset["train"]
        K_dataset_test = K_dataset["test"]

        for attribute in K_dataset_train:
            info(f"train settings:{self.base_model_name}")
            info(f"train settings:K-{K}")
            info(f"train settings:attribute-{attribute}")

            training_args = TrainingArguments(
                output_dir=f"train_out/seperate_train/{K}-shots/{attribute}",
                learning_rate=lr,
                per_device_train_batch_size=16,
                per_device_eval_batch_size=16,
                num_train_epochs=20,
                weight_decay=0.01,
                evaluation_strategy="epoch",
                save_strategy="epoch",
                load_best_model_at_end=True,
                logging_dir='./train_logs',
                remove_unused_columns=False
            )

            train_dataset = self._get_dataset(K_dataset_train[attribute])
            info(f"training dataset: {train_dataset}")
            eval_dataset = self._get_dataset(K_dataset_test[attribute])
            info(f"eval dataset: {eval_dataset}")

            label_indices = get_indices(
                [attribute],
                self.label_info["label2sim"],
                self.none_label
            )
            my_trainer = trainer.CustomTrainer(
                label_indices,
                self.metrics_info,
                model=self.model,
                args=training_args,
                train_dataset=train_dataset,
                eval_dataset=eval_dataset,
                tokenizer=self.tokenizer,
                compute_metrics=trainer.compute_metrics
            )
            my_trainer.train()

    def mix_train(self, lr, batch_size, epochs, K, optimizer=torch.optim.AdamW, device='cpu'):
        pass

    def _process_dataset(self, dataset: Dict[str, DataFrame], hard_prompt: Template, pos_strategy: PosStrategy):
        input_and_labels = {}
        not_found_count = {}
        for attribute in dataset:
            not_found_count[attribute] = 0
            input_and_labels[attribute] = []
            df = dataset[attribute]
            for _, row in df.iterrows():
                context = row['context']
                id_ = row['ID']
                value = row['value']
                is_negative = row['is_negative']

                candidates = extract_noun_phrases(context, pos_strategy)
                found_value = False
                buffer = []
                for can in candidates:
                    to_substitute = {
                        "context": context,
                        "candidate": can
                    }
                    input_str = hard_prompt.substitute(**to_substitute)
                    mask_idx = get_mask_idx(input_str, self.tokenizer)
                    if can == value:
                        found_value = True
                        label = attribute
                        buffer.append({
                            "input": input_str,
                            "label": label,
                            "sentence_id": id_,
                            "mask_idx": mask_idx + self.prompt_offset + 1
                        })
                    else:
                        label = self.none_label
                        buffer.append({
                            "input": input_str,
                            "label": label,
                            "sentence_id": id_,
                            "mask_idx": mask_idx + self.prompt_offset + 1
                        })

                if found_value is False and is_negative is False:
                    # info(f"value not found:{context}:{value}:{attribute}")
                    not_found_count[attribute] += 1
                else:
                    input_and_labels[attribute].extend(buffer)

        info(f"value not found:{not_found_count}")
        return input_and_labels

    def _preprocess_dataset(self, 
                            dataset: Dict[str, DataFrame],
                            pos_strategy: PosStrategy,
                            hard_prompt: Template,
                            shots: List[int],
                            none_label: str
                            ):
        self.raw_dataset = dataset
        self.raw_K_shots_dataset = {}
        info(f"K shots setting: {shots}")
        for K in shots:
            self.raw_K_shots_dataset[K] = {"train": {}, "test": {}}
            for attribute in dataset:
                train_data, test_data = random_split(dataset[attribute], K)
                self.raw_K_shots_dataset[K]["train"][attribute] = train_data
                self.raw_K_shots_dataset[K]["test"][attribute] = test_data
        '''
        K:
            {
                "train":{
                    "BrandName":{
                        "context", "attr", "value", "is_negative"
                        "..."
                    }
                },
                "test":{
                    ...
                }
            }
        '''

        K_shots_input_and_labels = {}
        for K in shots:
            K_shots_input_and_labels[str(K)] = {}
            shot_dataset = self.raw_K_shots_dataset[K]
            for _type, dataset in shot_dataset.items(): # _type could be train or test
                cur_dataset = self._process_dataset(dataset, hard_prompt, pos_strategy)
                K_shots_input_and_labels[str(K)][_type] = cur_dataset
        self.dataset = K_shots_input_and_labels
        date_str = datetime.now()
        dump2json(K_shots_input_and_labels, f"ae_100k_post_process_{date_str}.json".replace(' ', '_'))
        info("dumped processed dataset to file: " + f"ae_100k_post_process_{date_str}.json".replace(' ', '_'))
        '''
        K:
            {
                "train":{
                    "BrandName":{
                        (input, label, source_id)
                    }
                },
                "test":{

                }
            }
        '''

class TestAddNumbers(unittest.TestCase):
    def test_model_construct(self):
        model = ModelForAE(config.prompt_length,
                           config.base_model,
                           config.dataset_name,
                           config.attributes,
                           config.hard_prompt_template,
                           config.pos_strategy,
                           config.shots,
                           config.prompt_init_strategy,
                           config.none_label,
                           config.metrics,
                           )
        model.seperate_train(config.learning_rate,
                             config.batch_size,
                             config.num_epochs,
                             20)


if __name__ == '__main__':
    init_logging()
    unittest.main()
