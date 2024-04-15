'''
from text to list of ids
'''
def encode(tokenizer, text):
    return tokenizer.convert_tokens_to_ids(tokenizer.tokenize(text))


'''
from ids to tokens
'''
def decode(tokenizer, ids):
    return tokenizer.decode(ids)
