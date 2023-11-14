import torch
import numpy as np
import pandas as pd
from tqdm import tqdm

from src.utils.general_helper import duplicate_samples


def baseline_generation(data_args, model_args, gen_args, file_save_path,
                        model, tokenizer, data, use_cuda=True):
    
    preds_list, golds_list = [], []
    raw_preds_list, prompts_list = [], []
    knowledges_list, dialogue_histories_list = [], []
    
    for i in tqdm(range((len(data))//gen_args.batch_size+1)):
        start_idx = i*gen_args.batch_size
        end_idx = (i+1)*gen_args.batch_size

        prompts = data.dataset['prompts'][start_idx:end_idx]
        golds = data.dataset['responses'][start_idx:end_idx]
        golds = [gold[:-len(tokenizer.eos_token)] for gold in golds]
        knowledges = data.dataset['knowledges'][start_idx:end_idx]
        dialogue_histories = data.dataset['dialogue_histories'][start_idx:end_idx]
        
        if len(prompts) > 0:
        
            ### GPT-2 generate
            inputs = tokenizer(prompts,
                               return_tensors='pt',
                               padding=True)

            if use_cuda:
                inputs['input_ids'] = inputs['input_ids'].cuda()
                inputs['attention_mask'] = inputs['attention_mask'].cuda()

            ## BART .generate needs attention_mask, not attention_masks.     T5 too.
            outputs = model.generate(inputs['input_ids'],
                                     num_beams=gen_args.num_beams,
                                     do_sample=gen_args.do_sample,
                                     top_k=gen_args.top_k,
                                     top_p=gen_args.top_p,
                                     num_return_sequences=gen_args.num_return_sequences,
                                     temperature=gen_args.temperature,
                                     typical_p=gen_args.typical_p,
                                     num_beam_groups=gen_args.num_beam_groups,
                                     diversity_penalty=gen_args.diversity_penalty,
                                     min_length=gen_args.min_length,
                                     max_length=inputs['input_ids'].shape[-1]+gen_args.max_length,
                                     pad_token_id=tokenizer.pad_token_id,
                                     eos_token_id=tokenizer.eos_token_id,
                                     attention_mask=inputs['attention_mask'])

            if use_cuda:
                inputs['input_ids'] = inputs['input_ids'].cpu()
                inputs['attention_mask'] = inputs['attention_mask'].cpu()

            raw_preds = tokenizer.batch_decode(outputs, skip_special_tokens=gen_args.skip_special_tokens)
            decoded_prompts = tokenizer.batch_decode(inputs['input_ids'], skip_special_tokens=gen_args.skip_special_tokens)
            duplicated_prompts = duplicate_samples(decoded_prompts, gen_args.num_return_sequences)

            preds = []
            if "gpt2" in model_args.tokenizer_name:      
                for k, raw_pred in enumerate(raw_preds):
                    gen_idx = len(duplicated_prompts[k])
                    eos_idx = raw_pred.rfind(tokenizer.eos_token)
                    clean_pred = raw_pred[gen_idx:eos_idx].strip()

                    if data_args.addprompt_w_n_gold_words > 0:
                        clean_pred = " ".join(golds[k].split(' ')[:data_args.addprompt_w_n_gold_words]) + clean_pred
                    preds.append(clean_pred)

            # For T5
            elif "t5" in model_args.tokenizer_name:
               for k, raw_pred in enumerate(raw_preds):
                   eos_idx = raw_pred.rfind(tokenizer.eos_token)
                   clean_pred = raw_pred[len(tokenizer.pad_token):eos_idx].strip()
            
                   if data_args.addprompt_w_n_gold_words > 0:
                       clean_pred = " ".join(golds[k].split(' ')[:data_args.addprompt_w_n_gold_words]) + clean_pred
                   preds.append(clean_pred)

            if gen_args.num_return_sequences > 0:
                prompts = duplicate_samples(prompts, gen_args.num_return_sequences)
                golds = duplicate_samples(golds, gen_args.num_return_sequences)
                knowledges = duplicate_samples(knowledges, gen_args.num_return_sequences)
                dialogue_histories = duplicate_samples(dialogue_histories, gen_args.num_return_sequences)

            if gen_args.print_raw_preds:
                raw_preds_list.extend(raw_preds)
            prompts_list.extend(prompts)
            preds_list.extend(preds)
            golds_list.extend(golds)
            knowledges_list.extend(knowledges)
            dialogue_histories_list.extend(dialogue_histories)

            torch.cuda.empty_cache()
        
    ## Constructing generations as csv
    if gen_args.print_raw_preds:
        df = pd.DataFrame({'raw_preds' : raw_preds_list,
                           'prompts' : prompts_list,
                           'preds' : preds_list,
                           'golds' : golds_list,
                           'knowledges' : knowledges_list,
                           'dialogue_histories' : dialogue_histories_list})
    else:
        df = pd.DataFrame({'prompts' : prompts_list,
                           'preds' : preds_list,
                           'golds' : golds_list,
                           'knowledges' : knowledges_list,
                           'dialogue_histories' : dialogue_histories_list})
        
    ## Get sample_group
    
    # Get unique_sample id
    dupe_factor = gen_args.num_return_sequences
    sample_size = df.shape[0]//dupe_factor

    # Define group
    sample_to_index = {}
    for n_sample in range(sample_size):
        sample_to_index.update({n_sample:np.arange(n_sample*dupe_factor,(n_sample+1)*dupe_factor,1)})
    index_to_sample = {v:k for k, vs in sample_to_index.items() for v in vs}
    
    # Fill the unique_sample id
    df = df.reset_index()
    df['sample_group'] = df.apply(lambda x: index_to_sample[x['index']], axis=1)
    
    ## Save
    df.to_csv(file_save_path, index=False, compression='gzip')
        
    return df