import os
import sys
import math
import torch
import json
import numpy as np
import pandas as pd
from tqdm import tqdm
from transformers import (
    LogitsProcessorList,
    MinLengthLogitsProcessor,
    StoppingCriteriaList,
    MaxLengthCriteria,
    HfArgumentParser,
    set_seed
)

from src.model.model import load_model_and_tokenizer
from src.dataset.load_dataset import load_dataset, load_splits
from src.metrics.measurements import filter_by_rowwise_generation_metrices, corpuswise_generation_metrices
from src.generation.baseline import baseline_generation
from src.utils.general_helper import set_all_seeds
from src.utils.inference_args_helper import (
    ModelArguments, DataArguments, GenerationArguments, TrainingArguments
)


def do_inference(model_args, data_args, gen_args, training_args=None):
    
    torch.no_grad()
    
    # Init seed
    set_seed(gen_args.gen_seed)
    set_all_seeds(gen_args.gen_seed)

    ##### FOLDERING #####
    if 'save' in model_args.model_name_or_path:
        folder_path = model_args.model_name_or_path+'/'
    else:
        folder_path = gen_args.gen_output_dir+'/'+model_args.model_name_or_path.replace('/','_')+'/'

    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
    
    hf_pretrained_models = ['gpt2']
    if model_args.model_name_or_path in hf_pretrained_models:
        # this is a prompting run
        if not os.path.isdir(folder_path):
            os.mkdir(folder_path)
        folder_path += 'prompt_'
        model_args.prompt_mode = True
    else:
        # this is a finetuning evaluation run
        folder_path += 'fteval_'
        
    if data_args.debug_mode:
        folder_path += 'debug_'
        
    if gen_args.n_token_add > 0:
        NotImplementedError('This mode is not implemented here')
    else:
        folder_path += '{}_{}_{}sep_{}prs_spktoken{}_{}pad_{}dhcount_nb{}_sample{}_k{}_p{}_numseq{}_{}ngoldinprompt_{}-{}genlen_{}bs_{}seed_temp{}_typp{}_nbeamgr{}_divp{}_tokentype_{}'\
                                        .format(data_args.dataset_name,
                                                data_args.experiment_mode,
                                                data_args.sep_token,
                                                data_args.prompt_resp_sep_token,
                                                data_args.use_speaker_token,
                                                data_args.pad_token,
                                                data_args.dh_uttr_count,
                                                gen_args.num_beams,
                                                gen_args.do_sample,
                                                gen_args.top_k,
                                                gen_args.top_p,
                                                gen_args.num_return_sequences,
                                                data_args.addprompt_w_n_gold_words,
                                                gen_args.min_length,
                                                gen_args.max_length,
                                                gen_args.batch_size,
                                                gen_args.gen_seed,
                                                gen_args.temperature,
                                                gen_args.typical_p,
                                                gen_args.num_beam_groups,
                                                gen_args.diversity_penalty,
                                                data_args.use_token_type_ids)
    print(folder_path)
    
    if not os.path.isdir(folder_path):
        os.mkdir(folder_path)
        
    ##### MODEL and TOKENIZER #####
    model, tokenizer = load_model_and_tokenizer(data_args, model_args)
    model.config.pad_token_id = model.config.eos_token_id
    model.eval()
    model.cuda()

    ##### DATASET and INFERENCE #####
    full_test_list, minimum_test_list = load_splits(data_args)
    if gen_args.full_test:
        splits = full_test_list
    else:
        splits = minimum_test_list
        
    metrics_dict = {}
    metric_save_path = folder_path+'/'+'corpuswise_evaluation_metrics.json'
    for split in splits:
        
        data = load_dataset(data_args, model_args, tokenizer, split)
        
        file_save_path = folder_path+'/'+ split+'_'
        if gen_args.n_token_add > 0:
            NotImplementedError('This mode is not implemented here')
        else:
            file_save_path += 'nb{}_sample{}_k{}_p{}_numseq{}_preds_golds.gzip'\
                                        .format(gen_args.num_beams,
                                                gen_args.do_sample,
                                                gen_args.top_k,
                                                gen_args.top_p,
                                                gen_args.num_return_sequences)
        
        if not os.path.isfile(file_save_path):
            if gen_args.n_token_add > 0:
                NotImplementedError('This mode is not implemented here')
            else:
                df = baseline_generation(data_args, model_args, gen_args, file_save_path,
                                         model, tokenizer, data, split)
        
        else:
            df = pd.read_csv(file_save_path, compression='gzip')
            df.fillna('', inplace=True)
        
        if gen_args.n_token_add > 0 or gen_args.num_return_sequences > 1:
            print("Filtering by rowwise metrices")
            df = filter_by_rowwise_generation_metrices(df, gen_args, file_save_path)
        
        print("Calculating corpuswise metrices")
        metrics_dict[split] = corpuswise_generation_metrices(df, model, tokenizer, split,
                                                             data_args, model_args)
        
        with open(metric_save_path, "w") as write_file:
            json.dump(metrics_dict, write_file, indent=4)
        
#####
# Entry Point
#####
def main():
    
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, GenerationArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, gen_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, gen_args, training_args = parser.parse_args_into_dataclasses()
        
    # Start
    do_inference(model_args, data_args, gen_args, training_args)
    

if __name__ == '__main__':
    main()