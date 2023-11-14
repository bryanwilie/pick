import os
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from nltk import word_tokenize
from datasets import load_dataset
from torch.utils.data import Dataset



class WoWJSONDataset(Dataset):
    
    def __init__(self, data_args, model_args, tokenizer,
                 split='valid_seen',
                 *args, **kwargs):
        """
        dh_uttr_count: how many of the utterances before are included in the dialogue history
        """
        
        self.pad_token_id = tokenizer.pad_token_id
        self.data_pool = 'data_pool/'
        self.split = split
        self.sep_token = tokenizer.eos_token if data_args.sep_token == '' \
                                                else data_args.sep_token
        self.prompt_resp_sep_token = tokenizer.eos_token if data_args.prompt_resp_sep_token == '' \
                                                            else data_args.prompt_resp_sep_token
        self.tokenizer = tokenizer
        self.data_args = data_args
        self.model_args = model_args
        self.dataset = self.load_dataset()
        
        
    def load_dataset(self):
        
        def load_cached_dataset(mode = 'wizard_oriented_record'):
        
            data_path = self.data_pool+'wizard_of_wikipedia/'
            csv_path = data_path+'wizard_oriented_record_'+self.split+\
                        '_dhuttrcount_'+str(self.data_args.dh_uttr_count)+\
                        '_speaker_token_'+str(self.data_args.use_speaker_token)+'.csv'

            if not os.path.isfile(csv_path):
                json_path_to_split_dict = {'train':'train', 
                                           'valid_random_split':'valid_seen', 
                                           'valid_topic_split':'valid_unseen', 
                                           'test_random_split':'test_seen', 
                                           'test_topic_split':'test_unseen'}
                split_to_json_path_dict = {v:k for k,v in json_path_to_split_dict.items()}

                json_path = data_path+split_to_json_path_dict[self.split]+'.json'
                
                if mode == 'wizard_oriented_record':
                    
                    df = self.wizard_oriented_record(pd.read_json(json_path))
                    df.to_csv(csv_path, index=False)
                    df = pd.read_csv(csv_path)
            else:
                df = pd.read_csv(csv_path)
                
            if self.data_args.debug_mode:
                df = df.head(100)
                
            return df

        df = load_cached_dataset()
            
        if self.data_args.no_passages_used_settings == "nan":
            df['checked_sentences'] = df.apply(lambda x: np.nan if x['checked_sentences']=='no_passages_used' else x['checked_sentences'], axis=1)
        elif self.data_args.no_passages_used_settings == "no_passage_used":
            df['checked_sentences'] = df.apply(lambda x: 'no_passages_used' if pd.isna(x['checked_sentences']) else x['checked_sentences'], axis=1)
            
        # Default filling nan values with empty string
        df = df.fillna("")
        
        # Constructing the whole prompts
        prompts, responses, texts, dialogue_histories, knowledges = [], [], [], [], []
        for i in df.index:
            
            if self.model_args.prompt_mode:

                if self.data_args.experiment_mode == 'tdhkn':
                    prompt = df['chosen_topic'][i] + '. ' + \
                                df['checked_sentences'][i] + ' ' + \
                                df['dialogue_history'][i]
                else:
                    raise NotImplementedError('Data_args: {} is not yet implemented'.\
                                            format(self.data_args.experiment_mode))
                
            else:

                if self.data_args.experiment_mode == 'kndh': 
                    prompt = self.tokenizer.bos_token + df['checked_sentences'][i] +\
                                self.sep_token + df['dialogue_history'][i]
                elif self.data_args.experiment_mode == 'dh':
                    prompt = self.tokenizer.bos_token + df['dialogue_history'][i]
                elif self.data_args.experiment_mode == 'tdh':
                    prompt = self.tokenizer.bos_token + df['chosen_topic'][i] +\
                                self.sep_token + df['dialogue_history'][i]
                elif self.data_args.experiment_mode == 'tpkndh': 
                    prompt = self.tokenizer.bos_token + df['chosen_topic'][i] +\
                                self.sep_token + df['persona'][i]+\
                                self.sep_token + df['checked_sentences'][i] +\
                                self.sep_token + df['dialogue_history'][i]
                elif self.data_args.experiment_mode == 'tdhkn':
                    if "gpt2" in self.model_args.tokenizer_name or "bart" in self.model_args.tokenizer_name:
                        prompt = self.tokenizer.bos_token+df['chosen_topic'][i] +\
                                    self.sep_token + df['dialogue_history'][i] +\
                                    self.sep_token + df['checked_sentences'][i]
                    elif "t5" in self.model_args.tokenizer_name:
                        prompt = df['chosen_topic'][i] +\
                                    self.sep_token + df['dialogue_history'][i] +\
                                    self.sep_token + df['checked_sentences'][i]
                    else:
                         raise NotImplementedError('Model_args.tokenizer_name: {} is not yet implemented'.\
                                            format(self.model_args.tokenizer_name))
                else:
                    raise NotImplementedError('Data_args: {} is not yet implemented'.\
                                            format(self.data_args.experiment_mode))
            
                # zero-shot prompting pre-trained GPT-2 will fail if the prompt ended with eos_token
                prompt += self.prompt_resp_sep_token
                
            response = df['texts'][i] + self.tokenizer.eos_token
            text = prompt + response
            
            if self.data_args.addprompt_w_n_gold_words > 0:
                prompt = prompt + " ".join(response.split(' ')[:self.data_args.addprompt_w_n_gold_words])
            
            prompts.append(prompt)
            responses.append(response)
            texts.append(text)

        if "gpt" in self.model_args.model_name_or_path:
            dataset = self.tokenizer(texts,
                                     add_special_tokens=False,
                                     padding=True,
                                     truncation=True,
                                     max_length=self.model_args.max_seq_len, 
                                     pad_to_multiple_of=self.model_args.max_seq_len)
        
            # Change padding to custom_pad_token
            if self.data_args.pad_token != '':
                dataset['input_ids'] = np.array(dataset['input_ids'])
                dataset['input_ids'] = np.where(dataset['input_ids'] == self.tokenizer.pad_token_id, \
                                                    self.pad_token_id, dataset['input_ids']).tolist()
                
            # We only compute loss on masked tokens
            # pad_token_id alone is not good enough to know whether label should be excluded or not
            dataset['labels'] = np.array(dataset['input_ids'])
            dataset['labels'][np.array(dataset['attention_mask'])==False] = -100
            dataset['labels'] = dataset['labels'].tolist()
        

        # Change for T5 here
        elif "t5" in self.model_args.model_name_or_path:
            dataset = self.tokenizer(prompts,
                                     add_special_tokens=False,
                                     padding=True,
                                     truncation=True,
                                     return_token_type_ids=self.data_args.use_token_type_ids,
                                     max_length=self.model_args.max_seq_len, 
                                     pad_to_multiple_of=self.model_args.max_seq_len)  
            
            labels = self.tokenizer(responses,
                                   add_special_tokens=False,
                                   padding=True,
                                   truncation=True,
                                   return_token_type_ids=self.data_args.use_token_type_ids,
                                   max_length=self.model_args.max_seq_len,
                                   pad_to_multiple_of=self.model_args.max_seq_len)
            
            # Change padding to custom_pad_token
            if self.data_args.pad_token != '':
                dataset['input_ids'] = np.array(dataset['input_ids'])
                dataset['input_ids'] = np.where(dataset['input_ids'] == self.tokenizer.pad_token_id, \
                                                     self.pad_token_id, dataset['input_ids']).tolist()

            # We only compute loss on masked tokens
            # pad_token_id alone is not good enough to know whether label should be excluded or not
            dataset['labels'] = np.array(labels['input_ids'])
            dataset['labels'][np.array(labels['attention_mask'])==False] = -100
            dataset['labels'] = dataset['labels'].tolist()


        # Add token type, 1 for all prompt until before prompt_resp_sep_token.
        if self.data_args.use_token_type_ids:
            dataset['token_type_ids'] = self.tokenizer(prompts,
                                                       add_special_tokens=False,
                                                       padding=True,
                                                       truncation=True,
                                                       max_length=self.model_args.max_seq_len, 
                                                       pad_to_multiple_of=self.model_args.max_seq_len)['attention_mask']
            
            # Ensure prompt_resp_sep_token token type is 0
            if self.tokenizer.padding_side == 'left':
                dataset['token_type_ids'] = np.array(dataset['token_type_ids'])
                dataset['token_type_ids'][:,-1]=0
            else:
                dataset['token_type_ids'] = np.append(np.array(dataset['token_type_ids'])[:,1:], \
                                                      [[0]]*np.array(dataset['token_type_ids']).shape[0], axis=1)
                
        knowledges = df['checked_sentences'].values.tolist()
        dialogue_histories = df['dialogue_history'].values.tolist()
        
        dataset['index'] = df.index
        dataset['prompts'] = prompts
        dataset['responses'] = responses
        dataset['texts'] = texts
        dataset['knowledges'] = knowledges
        dataset['dialogue_histories'] = dialogue_histories
        
        # consider converting the list to a single numpy.ndarray with numpy.array() before converting to a tensor
        dataset['input_ids'] = np.array(dataset['input_ids'])
        dataset['labels'] = np.array(dataset['labels']) 
        dataset['attention_mask'] = np.array(dataset['attention_mask'])
        if self.data_args.use_token_type_ids:
            dataset['token_type_ids'] = np.array(dataset['token_type_ids'])
            
        return dataset
        
        
    def __getitem__(self, index):
        
        forward_inputs = {}
        forward_inputs['input_ids'] = self.dataset['input_ids'][index]
        forward_inputs['labels'] = self.dataset['labels'][index]
        forward_inputs['attention_mask'] = self.dataset['attention_mask'][index]
        if self.data_args.use_token_type_ids:
            forward_inputs['token_type_ids'] = self.dataset['token_type_ids'][index]
        
        return forward_inputs
    
    
    def wizard_oriented_record(self, df_wow):
        
        wiz_dict = {}
        wiz_dict['texts'] = []
        wiz_dict['checked_sentences'] = []
        wiz_dict['dialogue_history'] = []
        wiz_dict['chosen_topic'] = []
        wiz_dict['persona'] = []
        for i, df_row in df_wow.iterrows():
            for turn, dialog in enumerate(df_row['dialog']):
                if 'Wizard' in dialog['speaker'] and turn != 0:
                    wiz_dict['texts'].append(dialog['text'])
                    wiz_dict['checked_sentences'].append(dialog['checked_sentence'])
                    wiz_dict['chosen_topic'].append(df_row['chosen_topic'])
                    wiz_dict['persona'].append(df_row['persona'])
                    dialogue_history = []

                    for prev in np.arange(1, self.data_args.dh_uttr_count+1, 1):
                        if turn-prev >= 0:
                            dialogue_history.append(df_wow.iloc[[i]]['dialog'].values[0][turn-prev]['text'])
                        else:
                            break
                    dialogue_history.reverse()
                    wiz_dict['dialogue_history'].append(dialogue_history)

        df_result = pd.DataFrame(wiz_dict)
        df_result['checked_sentences'] \
            = df_result.apply(lambda x: list(x['checked_sentences'].values()), axis=1)
        df_result['checked_sentences'] \
            = [df_row[0] if len(df_row)>0 else "" for df_row in df_result['checked_sentences']]
        
        if self.data_args.use_speaker_token:

            # Assuming only 2 speakers and always alternating (12121..)
            speaker_tokens = []
            for special_token in self.tokenizer.additional_special_tokens:
                if 'speaker' in special_token:
                    speaker_tokens.append(special_token)

            def put_alternating_speaker_token(dialogue_history_per_row):
                
                dialogue_history = ''
                for i, utterance in enumerate(dialogue_history_per_row):
                    if i%2 == 0:
                        dialogue_history += utterance + ' ' + speaker_tokens[0] + ' '
                    else:
                        dialogue_history += utterance + ' ' + speaker_tokens[1] + ' '

                return dialogue_history.strip()

            df_result['dialogue_history'] = df_result.apply(lambda x: put_alternating_speaker_token(x['dialogue_history']), axis=1)
        else:
            df_result['dialogue_history'] = df_result.apply(lambda x: ' '.join(x['dialogue_history']), axis=1)
        
        return df_result
    


    
    def __len__(self):
        return len(self.dataset['texts'])