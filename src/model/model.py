from transformers import (
    GPT2Tokenizer,
    GPT2LMHeadModel,
    BartTokenizer,
    BartForConditionalGeneration,
    T5Tokenizer,
    T5ForConditionalGeneration
)


ATTR_TO_SPECIAL_TOKEN = {'additional_special_tokens': ['<speaker1>', '<speaker2>']}


def add_special_tokens(model, tokenizer):
    """ Add special tokens to the tokenizer and the model if they have not already been added. """
    orig_num_tokens = len(tokenizer)
    num_added_tokens = tokenizer.add_special_tokens(ATTR_TO_SPECIAL_TOKEN) # doesn't add if they are already there
    if num_added_tokens > 0:
        model.resize_token_embeddings(new_num_tokens=orig_num_tokens + num_added_tokens)


def load_model_and_tokenizer(data_args, model_args):
    
    if 'gpt2' in model_args.model_name_or_path:
        
        model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.tokenizer_name)
        
        if data_args.pad_token == '':
            tokenizer.pad_token = tokenizer.eos_token
        else:
            tokenizer.add_special_tokens({'pad_token': data_args.pad_token})
            tokenizer.pad_token_id = tokenizer.encode(data_args.pad_token)[0]
        tokenizer.padding_side = model_args.padding_side

    elif 'dialogpt' in model_args.model_name_or_path:
        
        model = GPT2LMHeadModel.from_pretrained(model_args.model_name_or_path)
        tokenizer = GPT2Tokenizer.from_pretrained(model_args.tokenizer_name)

    elif 'bart' in model_args.model_name_or_path:

        raise NotImplementedError('model_args.model_name_or_path: {} is not yet implemented'.\
                                            format(model_args.model_name_or_path))
    
    elif 't5' in model_args.model_name_or_path:
        
        data_args.use_token_type_ids = False
        model = T5ForConditionalGeneration.from_pretrained(model_args.model_name_or_path)
        tokenizer = T5Tokenizer.from_pretrained(model_args.tokenizer_name)
        tokenizer.padding_side = model_args.padding_side

    # adding additional_special_tokens
    add_special_tokens(model, tokenizer)
        
    return model, tokenizer