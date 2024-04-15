from transformers import RobertaTokenizer, RobertaForMaskedLM
import torch

# Initialize the model and tokenizer
tokenizer = RobertaTokenizer.from_pretrained('roberta-base')
model = RobertaForMaskedLM.from_pretrained('roberta-base')
model.eval()

# Example text with multiple masks
text = "The <mask> jumps over the <mask>."
input_ids = tokenizer(text, return_tensors="pt").input_ids

# Lists of allowed words for each mask
allowed_words_lists = [
    ['cat', 'dog', 'fox'],  # Words allowed for the first mask
    ['fence', 'moon', 'sun']  # Words allowed for the second mask
]

predicted_tokens = []
with torch.no_grad():
    outputs = model(input_ids)
    predictions = outputs[0]

masked_indices = torch.where(input_ids == tokenizer.mask_token_id)[1]

for i, masked_index in enumerate(masked_indices):
    mask_logits = predictions[0, masked_index]
    allowed_words = allowed_words_lists[i]
    
    # Tokenize the allowed words and get their token IDs
    allowed_token_ids = [tokenizer.encode(word, add_special_tokens=False) for word in allowed_words]
    allowed_token_ids = [token_id for sublist in allowed_token_ids for token_id in sublist]
    
    # Filter logits to only include those for the allowed tokens
    filtered_logits = mask_logits[allowed_token_ids]
    
    # Find the highest scoring token from the allowed list
    predicted_token_id = allowed_token_ids[filtered_logits.argmax()]
    predicted_token = tokenizer.decode(predicted_token_id)
    predicted_tokens.append(predicted_token)

# Fill in the masks with the predicted tokens
for token in predicted_tokens:
    text = text.replace("<mask>", token, 1)

print(text)
