import torch
from torch import nn
from transformers import RobertaModel, RobertaConfig

class PrefixTuningModel(nn.Module):
    def __init__(self, base_model, prefix_length=20, prefix_size=768):
        super().__init__()
        self.base_model = base_model
        self.prefix_length = prefix_length
        self.prefix_embeddings = nn.Parameter(torch.randn(prefix_length, prefix_size))

    def forward(self, input_ids, attention_mask=None):
        # Get the base model embeddings
        base_embeddings = self.base_model.embeddings(input_ids)

        # Prepend the prefix embeddings
        prefix_embeddings = self.prefix_embeddings.unsqueeze(0).expand(base_embeddings.size(0), -1, -1)
        extended_embeddings = torch.cat([prefix_embeddings, base_embeddings], dim=1)

        # Handle attention mask if provided
        if attention_mask is not None:
            extended_attention_mask = torch.cat([torch.ones_like(input_ids)[:, :self.prefix_length], attention_mask], dim=1)
        else:
            extended_attention_mask = None

        # Pass the combined embeddings through the base model
        outputs = self.base_model(inputs_embeds=extended_embeddings, attention_mask=extended_attention_mask)

        return outputs

# Load the base model
config = RobertaConfig.from_pretrained('roberta-base')
base_model = RobertaModel(config)

# Initialize the prefix-tuning model
prefix_model = PrefixTuningModel(base_model)

for epoch in range(3):  # number of epochs
    for batch in dataloader:
        input_ids, labels = batch

        # Forward pass
        outputs = prefix_model(input_ids)
        prediction_logits = outputs.logits

        # Compute the loss
        # Need to reshape logits and labels for CrossEntropyLoss
        loss = loss_function(prediction_logits.view(-1, tokenizer.vocab_size), labels.view(-1))

        # Backward pass
        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        print(f"Epoch {epoch}, Loss: {loss.item()}")


# Example usage
input_ids = torch.randint(0, config.vocab_size, (1, 10))  # Example input
outputs = prefix_model(input_ids)
print(outputs)
