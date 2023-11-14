import re
from src.dataset.wow_json import WoWJSONDataset

regex = re.compile('[^a-zA-Z]')


def load_dataset(data_args, model_args, tokenizer, split):
    dataset_name = regex.sub('', data_args.dataset_name.lower())
    
    if dataset_name == 'wow':
        data = WoWJSONDataset(data_args, model_args, tokenizer, split)
        
    else:
        NotImplementedError('Dataset loader for {} dataset is not yet implemented'.format(data_args.dataset_name))
        
    return data


def load_splits(data_args):
    
    dataset_name = regex.sub('', data_args.dataset_name.lower())
    
    if dataset_name == 'wow':
        full_test_list = ["test_seen", "test_unseen", "valid_seen", "valid_unseen"]
        minimum_test_list = ["test_unseen", "test_seen"]
        
    else:
        NotImplementedError('Dataset splits for {} dataset is not yet implemented'.format(data_args.dataset_name))
    
    return full_test_list, minimum_test_list