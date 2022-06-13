add_tokens = ['<', '<=', '<>', '<96']

def load_model(model_name: str):
    if model_name in ["t5-small", "t5-base", "t5-large", "t5-3B"]:        
        from transformers import T5ForConditionalGeneration
        return T5ForConditionalGeneration.from_pretrained(model_name)
    else:
        raise Exception('Invalid model. Supporting T5 only.')

def load_tokenizer(model_name: str):
    if model_name in ["t5-small", "t5-base", "t5-large", "t5-3B"]:
        from transformers import T5Tokenizer
        tokenizer = T5Tokenizer.from_pretrained(model_name)
        tokenizer.add_tokens(add_tokens)
        return tokenizer
    else:
        raise Exception('Invalid tokenizer. Supporting T5 only.')


