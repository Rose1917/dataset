from transformers import BertTokenizer, BertModel
import numpy as np
from sklearn.neighbors import NearestNeighbors

# Load pre-trained BERT tokenizer and model
tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
model = BertModel.from_pretrained('bert-base-uncased')

# Function to get BERT embeddings for a word
def get_word_embedding(word):
    inputs = tokenizer(word, return_tensors='pt')
    outputs = model(**inputs)
    word_embedding = outputs.last_hidden_state.mean(dim=1).detach().numpy()
    return word_embedding

# Pre-compute embeddings for the vocabulary
def precompute_embeddings(vocab):
    embeddings = []
    for word in vocab:
        embedding = get_word_embedding(word)
        embeddings.append(embedding)
    return np.vstack(embeddings)

# Find similar words using nearest neighbor search
def find_similar_words(word, n=5):
    word_embedding = get_word_embedding(word).flatten()
    nn = NearestNeighbors(n_neighbors=n, metric='cosine').fit(embeddings)
    distances, indices = nn.kneighbors([word_embedding])
    similar_words = [list(tokenizer.vocab.keys())[index] for index in indices[0]]
    return similar_words

# Precompute embeddings for the vocabulary
vocab = list(tokenizer.vocab.keys())  # Limiting to first 1000 words for efficiency
embeddings = precompute_embeddings(vocab)

# Example usage
word = 'cat'
similar_words = find_similar_words(word)
print(f"Words similar to '{word}': {similar_words}")
