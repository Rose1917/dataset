import spacy
import nltk
from enum import Enum
from nltk.corpus import wordnet
from torchtext.data.utils import ngrams_iterator
from torchtext.data.utils import get_tokenizer
from nltk.tokenize import word_tokenize
from nltk import pos_tag
from nltk import RegexpParser

class PosStrategy(Enum):
    SPACY = 1,
    TOKEN = 2,
    BIN_GRAM = 3
    NLTK = 4

    
nlp = spacy.load('en_core_web_sm')
nltk.download('wordnet')
nltk.download('punkt')
nltk.download('averaged_perceptron_tagger')

def is_noun_word(word:str):
    synsets = wordnet.synsets(word)
    return any(s.pos() == 'n' for s in synsets)

def n_gram(tokens, n):
    n_grams = []
    for k in range(1, n + 1):
        n_grams.extend([' '.join(tokens[i:i + k]) for i in range(len(tokens) - k + 1)])
    return n_grams

def nltk_parse(sentence: str, pattern='NP:{<DT>?<JJ>?<NN>?<NN>}'):
    res = []
    sentence = sentence.lower()
    tokens = word_tokenize(sentence)
    tagged = pos_tag(tokens)
    print(tagged)
    cp = RegexpParser(pattern)
    noun_phrases_chunk = cp.parse(tagged)
    for npstr in noun_phrases_chunk.subtrees(filter=lambda x: x.label() == 'NP'):
        res.append(' '.join(word for word, pos in npstr.leaves()))
    return res


def extract_noun_phrases(sentence: str, strategy: PosStrategy = PosStrategy.TOKEN):
    if strategy == PosStrategy.SPACY:
        doc = nlp(sentence)
        noum_phrases = [chunk.text for chunk in doc.noun_chunks]
        return noum_phrases
    elif strategy == PosStrategy.TOKEN:
        # 由于本身属性抽取任务本身句子比较混乱
        # 常规的POS工具都不太好使，所以这里提供一种特殊的策略
        words = sentence.split()
        return list(filter(is_noun_word, words))
    elif strategy == PosStrategy.BIN_GRAM:
        words = sentence.split()
        return n_gram(words, 2)
    elif strategy == PosStrategy.NLTK:
        return nltk_parse(sentence)
    else:
        raise ValueError(f"unknown strategy: {strategy}")


if __name__ == '__main__':
    input = 'Stockings Santa Claus Sock Gift Kids Candy Bag Decoration for Home Sports Socks'
    input = 'Macbook Air high quality'
    input = 'Camping Stove Convert Propane Small Tank Input Output Outdoor Cylinder Canister Adapter ABS'
    input = 'chrome steel'
    # print(extract_noun_phrases(input, strategy=PosStrategy.BIN_GRAM))
    print(extract_noun_phrases(input, strategy=PosStrategy.NLTK))
