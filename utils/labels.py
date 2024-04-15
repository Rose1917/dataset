from enum import Enum
from typing import Dict, List
from transformers import AutoModel, AutoTokenizer
import faiss
import numpy as np
import torch
from .op import my_softmax
import unittest

class NoneLabelStrategy(Enum):
    StrStrategy = 1, # treat the none label as a special phrase
    ExcludeStrategy = 2 # exclude all valid labels


'''
`find_phrase_similar_tokens` function will return two values:
    similar_tokens: [(token1, similarity), (token2, similarity)..]
    in which len(similar_tokens) = n
```python
>>> sims, disms = find_phrase_similar_tokens('lense color', n=20)
>>> print(sims[0]) # (color, 0.7924)
>>> len(sims[0]) # 10
```

NOTE: by default we use the AVERAGE POOLING to calculate the embeddding of the phrase
'''

def _build_faiss_index(embeddings):
    dimension = embeddings.shape[1]
    index = faiss.IndexFlatL2(dimension)
    index.add(embeddings)
    return index

def _cosine_similarity_from_l2(l2_distances):
    return 1 - (l2_distances / 2)


'''
return_type: indicate return the token or id in the model's vocab
can be `token` or `id`
'''
def find_phrase_similar_tokens(phrase, model_name, n=10, return_type='id'):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)

    index = _build_faiss_index(embeddings)

    word_tokens = tokenizer.tokenize(phrase)
    word_ids = tokenizer.convert_tokens_to_ids(word_tokens)
    word_embeddings = torch.tensor([[embeddings[idx] for idx in word_ids]])
    phrase_embedding = torch.mean(word_embeddings, dim=1) # average pooling
    phrase_embedding = phrase_embedding / np.linalg.norm(phrase_embedding, axis=1, keepdims=True)

    # similar_tokens finding
    distances, indices = index.search(phrase_embedding, n + 1)  # +1 to exclude the word itself
    if return_type == 'token':
        similar_tokens = [(tokenizer.decode([idx]), _cosine_similarity_from_l2(distances[0][i])) for i, idx in enumerate(indices[0])][:n]
    else:
        similar_tokens = [(idx, _cosine_similarity_from_l2(distances[0][i])) for i, idx in enumerate(indices[0])][:n]

    # dissimilar tokens finding
    total_tokens = embeddings.shape[0]
    distances, least_indices = index.search(phrase_embedding, total_tokens)
    least_tokens_indices = least_indices[0][-n-1:-1]  # exclude the word itself
    least_tokens_distances = distances[0][-n-1:-1]

    if return_type == 'token':
        least_similar_tokens = [(tokenizer.decode([idx]), _cosine_similarity_from_l2(least_tokens_distances[i])) for i, idx in enumerate(least_tokens_indices)]
    else:
        least_similar_tokens = [(idx, _cosine_similarity_from_l2(least_tokens_distances[i])) for i, idx in enumerate(least_tokens_indices)]

    return similar_tokens, least_similar_tokens

def get_indices(labels, label2sim, none_label, none_label_strategy=NoneLabelStrategy.StrStrategy):
    '''
    this function is used to get the indices of labels
    when seperate train(binary trainer), `labels` will only contain one element
    when together train(multiple labels), `labels` contains all the relevant labels
    >>> get_seperate_indices(['material'], 'none', label_info)
    >>> # [-2, 3, -5]
    '''
    if none_label_strategy == NoneLabelStrategy.StrStrategy:
        labels_set = set(labels)
        labels_set.add(none_label)
        labels = list(labels_set)
    indices = []
    for label in labels:
        indices.extend(label2sim[label]['ids'])
    return indices

def find_similar_tokens(word, model_name, n=10):
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)

    # Extract static embeddings directly from the model's word embedding layer
    embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()

    # Normalize the embeddings for cosine similarity
    embeddings = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    index = _build_faiss_index(embeddings)
    word_id = tokenizer.convert_tokens_to_ids(word)
    word_embedding = embeddings[word_id:word_id + 1]

    # Perform the search
    distances, indices = index.search(word_embedding, n + 1)  # +1 to exclude the word itself
    similar_tokens = [(tokenizer.decode([idx]), _cosine_similarity_from_l2(distances[0][i])) for i, idx in enumerate(indices[0])][:n]

    # For least similar tokens
    total_tokens = embeddings.shape[0]
    distances, least_indices = index.search(word_embedding, total_tokens)
    least_tokens_distances = distances[0][-n - 1:-1]
    least_tokens_indices = least_indices[0][-n - 1:-1]  # exclude the word itself
    least_similar_tokens = [(tokenizer.decode([idx]), _cosine_similarity_from_l2(least_tokens_distances[i])) for i, idx in enumerate(least_tokens_indices)]

    return similar_tokens, least_similar_tokens

def get_labels(
        raw_labels: List[str],
        model_name,
        top_n=10,
        none_label='none',
        none_label_strategy=NoneLabelStrategy.StrStrategy):
    if none_label_strategy == NoneLabelStrategy.StrStrategy:
        raw_labels.append(none_label)

    raw_labels = list(set(raw_labels))
    label2sim = {}
    # calculate the label's tokens and corresponding value
    for label in raw_labels:
        sim, _ = find_phrase_similar_tokens(label, model_name, n=top_n)
        # sim: [(id, cos_sim), (id, cos_sim)]
        sim_tokens_id = [item[0] for item in sim]
        sim_values = my_softmax([item[1] for item in sim])
        label2sim[label] = {
            'ids': sim_tokens_id,
            'value': sim_values
        }

    tokenizer = AutoTokenizer.from_pretrained(model_name)
    label2vec = {}
    # setup the idx => label mapper
    idx2label = [none_label for _ in range(tokenizer.vocab_size)]
    employed_set = set()
    for label in label2sim:
        zeros = np.zeros(tokenizer.vocab_size)
        values = label2sim[label]['value']
        ids = label2sim[label]['ids']
        for value, idx in zip(values, ids):
            employed_set.add(idx)
            zeros[idx] = value
            if (idx2label[int(idx)] != none_label):
                raise ValueError('token collasion')
            else:
                idx2label[idx] = label
        label2vec[label] = zeros

    if none_label_strategy == NoneLabelStrategy.ExcludeStrategy:
        zeros = np.zeros(tokenizer.vocab_size)
        avg_mean = 1.0 / (tokenizer.vocab_size - len(employed_set))
        for i in range(tokenizer.vocab_size):
            if i not in employed_set:
                zeros[i] = avg_mean
        label2vec[none_label] = zeros

    return {
        "label2vec": label2vec,
        "idx2label": idx2label,
        "associate_indices": list(employed_set),
        "label2sim": label2sim
    }


class TestLabelsFunc(unittest.TestCase):
    def setUp(self):
        self.model_name = 'bert-base-uncased'
        self.word = 'color'
        self.phrase = 'product type'

    def test_find_similar_tokens(self):
        similar_tokens, least_similar_tokens = find_similar_tokens(self.word, self.model_name, n=10)
        print("Most similar tokens and their similarities:", similar_tokens)
        print("Least similar tokens and their similarities:", least_similar_tokens)

    def test_find_similar_phrase(self):
        similar_tokens, least_similar_tokens = find_phrase_similar_tokens(self.phrase, self.model_name, n=10, return_type='token')
        print("Most similar tokens and their similarities:", similar_tokens)
        print("Least similar tokens and their similarities:", least_similar_tokens)

    # def test_get_labels(self):
    #     output = get_labels([self.word, 'none'], self.model_name)
        # print(output)


if __name__ == '__main__':
    unittest.main()
    # model_name = 'roberta-base'
    # tokenizer = AutoTokenizer.from_pretrained(model_name)
    # model = AutoModel.from_pretrained(model_name)

    # embeddings = model.get_input_embeddings().weight.detach().cpu().numpy()
    # print(embeddings.shape)
