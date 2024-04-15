from transformers import Trainer, EvalPrediction
from transformers import DataCollatorWithPadding
from typing import Optional, List, Dict
from datasets import Dataset
import torch.nn as nn
import torch
from logging import info


class CustomTrainer(Trainer):
    def __init__(self, associate_indices, metrics_info, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.loss_fn = nn.KLDivLoss(reduction='batchmean')
        self.log_softmax = nn.LogSoftmax(dim=-1)
        self.associate_indices = associate_indices
        self.metrics_info = metrics_info

    def compute_loss(self, model, inputs, return_outputs=False):
        labels = inputs.pop("labels", None)
        # breakpoint()
        mask_idx = inputs.pop("mask_idx")
        outputs = model(**inputs)
        logits = outputs.get("logits")
        batch_indices = torch.arange(mask_idx.shape[0])
        target_logits = logits[batch_indices, mask_idx]

        # breakpoint()
        indices_tensor = torch.tensor(self.associate_indices, dtype=torch.long, device=logits.device)
        filtered_logits = target_logits[:, indices_tensor]
        filtered_labels = labels[:, indices_tensor]

        log_probs = self.log_softmax(filtered_logits)
        loss = self.loss_fn(log_probs, filtered_labels)
        return (loss, outputs) if return_outputs else loss

def compute_metrics(self, p: EvalPrediction) -> Dict[str, float]:
    logits = p.predictions
    labels = p.label_ids
    inputs = p.inputs
    breakpoint()
    return {
        "f1": 0.1,
        "precision": 0.1,
        "recall": 0.2
    }
