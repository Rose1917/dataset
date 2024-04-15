from peft import PromptTuningConfig, PromptTuningInit, get_peft_model
from peft.utils.config import TaskType
from transformers import AutoModelForMaskedLM
from transformers import AutoTokenizer
from datasets import load_dataset
import torch.nn as nn
import torch.nn.functional as F
import torch
from utils.labels import get_labels


model_name = "bert-base-uncased"
model = AutoModelForMaskedLM.from_pretrained("bert-base-uncased")

prompt_tuning_init_text = "color or material? "
tokenizer = AutoTokenizer.from_pretrained(model_name)
peft_config = PromptTuningConfig(
    task_type=TaskType.SEQ_CLS,
    prompt_tuning_init=PromptTuningInit.TEXT,
    num_virtual_tokens=len(tokenizer(prompt_tuning_init_text)["input_ids"]),
    prompt_tuning_init_text=prompt_tuning_init_text,
    tokenizer_name_or_path=model_name,
)

model = get_peft_model(model, peft_config)

offset = len(tokenizer(prompt_tuning_init_text)["input_ids"])
input = "red is a kind of [MASK].[SEP]"
input_tokens = tokenizer.tokenize(input)
print(f'input tokens:{input_tokens}')
masked_index = input_tokens.index("[MASK]")
print(f'masked index is {masked_index}')
input_ids = tokenizer(input)["input_ids"]
output = model(torch.tensor([input_ids]))

labels_info = get_labels(['color', 'none'], model_name)
label2vec = labels_info['label2vec']
idx2label = labels_info['idx2label']

model.print_trainable_parameters()

loss_func = nn.BCEWithLogitsLoss()
loss = loss_func(output.logits[0, offset + masked_index + 1].unsqueeze(0), torch.tensor([label2vec['color']]))
print(f'loss is {loss}')
breakpoint()
predictions = output.logits[0, offset + masked_index + 1]

predicted_idx = torch.argmax(predictions).item()
print(f'predicted token is {tokenizer.decode(predicted_idx)}')
print(predicted_idx)
print(f'predicted token is {idx2label[predicted_idx]}')


