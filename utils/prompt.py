from enum import Enum


class PromptInitStrategy(Enum):
    RANDOM_INIT = 0,
    LABEL_INIT = 1,
    CLS_INIT = 2

# TODO: add other init strategy impl
def get_prompt_init_text(strategy: PromptInitStrategy, labels=[]):
    if strategy == PromptInitStrategy.LABEL_INIT:
        return " or ".join(labels)

def adapt_prompt(model_name: str, template: str):
    template = template.lower()
    if 'robert' in model_name.lower():
        template = template.replace('[MASK]', '<mask>')
        template = template.replace('[SEP]', '</s>')
        return template
    return template

def get_mask_idx(input_str, tokenizer):
    tokens = tokenizer.tokenize(input_str)
    if tokens.index("[MASK]") != -1:
        return tokens.index("[MASK]")
    if tokens.index("<mask>") != -1:
        return tokens.index("<mask>")
    raise ValueError(f"can not find the <mask> in input sequence:{input_str}")
