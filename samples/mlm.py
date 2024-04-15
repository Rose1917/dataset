from transformers import BertForMaskedLM
from transformers import AutoTokenizer # tokenizer is task-free 
import torch
import torch.nn.functional as F
from utils.plm import decode


model_name = 'bert-base-uncased'
tokenizer = AutoTokenizer.from_pretrained(model_name)
model = BertForMaskedLM.from_pretrained(model_name)

text = "[CLS] color or none? red is a [MASK]. [SEP]"
tokenized_text = tokenizer.tokenize(text)
print(type(tokenized_text))
print(len(tokenized_text))
# print(tokenized_text.keys())

masked_index = tokenized_text.index("[MASK]")
token_ids = tokenizer.convert_tokens_to_ids(tokenized_text)
print(f'tokens ids:{token_ids}')
print(f'tokens:{decode(tokenizer, token_ids)}')
tokens_tensor = torch.tensor([token_ids])

for name, param in model.named_parameters():
    print(f"{name} -> Trainable: {param.requires_grad}")

# Get model predictions
with torch.no_grad():
    outputs = model(tokens_tensor, return_dict=True)
    predictions = outputs.logits

# Get the prediction for the masked token
# Get the predicted token
probability = F.softmax(predictions[0, masked_index], dim=0)
predicted_index = torch.argmax(probability).item()
predicted_token = tokenizer.convert_ids_to_tokens([predicted_index])[0]

print("Original:", text)
print("Masked:", tokenized_text)
print("Predicted token:", predicted_token)
print(f'Predicted Probability:{ probability[predicted_index]}')

# # Filter logits to only include those for the allowed tokens
# filtered_logits = mask_logits[allowed_token_ids]
# filtered_probs = F.softmax(filtered_logits, dim=0)

# # Get top-3 predictions from the allowed list
# top_filtered_probs, top_filtered_indices = torch.topk(filtered_probs, 3)
# print("\nTop-3 predictions after restriction:")
# for i in range(len(top_filtered_indices)):
#     token_id = allowed_token_ids[top_filtered_indices[i]]
#     token = tokenizer.decode([token_id])
#     prob = top_filtered_probs[i].item()
#     print(f"{i+1}: {token} with probability {prob}")


# List of allowed words to fill the mask
# allowed_tokens = ['color', 'Sarah', 'Robert']

# # Tokenize the allowed words and get their token IDs
# allowed_token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in allowed_tokens]
# allowed_token_ids = [token_id for sublist in allowed_token_ids for token_id in sublist]
