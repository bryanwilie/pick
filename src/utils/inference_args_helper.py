from dataclasses import dataclass, field
from transformers import (
    TrainingArguments,
)
from typing import Optional

@dataclass
class ModelArguments:
    """
    Arguments pertaining to which model/config/tokenizer we are going to utilize.
    """
    model_name_or_path: Optional[str] = field(
        default="gpt2", 
        metadata={"help": "Name of the chosen HF pretrained model or path to your own HF pretrained/finetuned model"}
    )
    max_seq_len: Optional[int] = field(
        default=512,
        metadata={"help": "Maximum sequence length the model can process."},
    )
    tokenizer_name: Optional[str] = field(
        default="gpt2", 
        metadata={"help": "Name of the chosen HF pretrained tokenizer"}
    )
    padding_side: Optional[str] = field(
        default="left", 
        metadata={"help": "Padding side",
                  'choices':['right', 'left']}
    )
    prompt_mode: bool = field(
        default=False, 
        metadata={"help": "Is this a prompt run and not a finetuning evaluation run?"}
    )
        

@dataclass
class DataArguments:
    """
    Arguments pertaining to the data loading and preprocessing pipeline.
    """
    dataset_name: Optional[str] = field(
        default="wow",
        metadata={"help": "Dataset name, i.e. wow for Wizard of Wikipedia"},
    )
    debug_mode: bool = field(
        default=False, 
        metadata={"help": "Enter debug mode by setting the data processed to be only 100"}
    )
    preprocessing_num_workers: Optional[int] = field(
        default=16,
        metadata={"help": "The number of processes to use for the preprocessing."},
    )
    experiment_mode: Optional[str] = field(
        default="tdhkn",
        metadata={"help": "Dataset alignment to construct the dataset as input"},
    )
    dh_uttr_count: Optional[int] = field(
        default=1,
        metadata={"help": "how many of the utterances before are \
                           included in the dialogue history"},
    )
    sep_token: Optional[str] = field(
        default='\n', 
        metadata={"help": "Set to empty string to use tokenizer.eos_token to segment different dialogue sections within the prompt"}
    )
    prompt_resp_sep_token: Optional[str] = field(
        default='', 
        metadata={"help": "Set to empty string to use tokenizer.eos_token to segment input's prompt and response"}
    )
    use_speaker_token: bool = field(
        default=False, 
        metadata={"help": "True sets the <speaker1> <speaker2> to be used alternatingly in dialogue history, assuming there's only 2 speakers."}
    )
    pad_token: Optional[str] = field(
        default='!', 
        metadata={"help": "Set empty to use tokenizer.eos_token to pad different input length for batching"}
    )
    use_token_type_ids: bool = field(
        default=False, 
        metadata={"help": "Will you use token type ids in model.forward?"}
    )
    no_knowledge_dataset: Optional[str] = field(
        default="letitbe", 
        metadata={"help": "Do you want the dataset with empty knowledge sentence or no_passage_used exist in the dataset?",
                  "choices": ["yes", "no", "letitbe"]}
    )
    no_passages_used_settings: Optional[str] = field(
        default="nosettings",
        metadata={"help": "'no_settings' defaults to letting the dataset be as they are\
                           'nan' means setting the no_passage_used to nan. This is nullifying the effect of 'no_passage_used' term in the sample\
                           'no_passage_used' setts every nan to no_passage_used. This is making the effect of 'no_passage_used' term sample-wide effective",
                  "choices": ["nosettings"]}
    )
    add_dataset: Optional[str] = field(
        default="no", 
        metadata={"help": "Augment the dataset used with the dataset of choice",
                  "choices": ["no"]}
    )
    addprompt_w_n_gold_words: Optional[int] = field(
        default=0,
        metadata={"help": "Add n first words exists in gold, to the prompt"},
    )
    
    
@dataclass
class GenerationArguments:
    """
    Arguments pertaining to the generation pipeline.
    """
    full_test: bool = field(
        default=False,
        metadata={"help": "True: sets eval to all valid and test, seen and unseen."
                          "False: sets eval to test_seen and test_unseen only"}
    )
    num_beams: Optional[int] = field(
        default=1,
        metadata={"help": "Beam size"},
    )
    top_k: Optional[int] = field(
        default=50,
        metadata={"help": "The number of highest probability vocabulary tokens to keep for top-k-filtering."},
    )
    top_p: Optional[float] = field(
        default=1,
        metadata={"help": "If set to float < 1, only the most probable tokens with probabilities that add up to `top_p` or higher are kept for generation."},
    )
    num_return_sequences: Optional[int] = field(
        default=1,
        metadata={"help": "The number of independently computed returned sequences for each element in the batch."},
    )
    do_sample: bool = field(
        default=False,
        metadata={"help": "Whether or not to use sampling ; use greedy decoding otherwise."}
    )
    batch_size: Optional[int] = field(
        default=1,
        metadata={"help": "Batch size"},
    )
    min_length: Optional[int] = field(
        default=10,
        metadata={"help": "Set p(EOS) to -inf if minimum not achieved yet"},
    )
    max_length: Optional[int] = field(
        default=80,
        metadata={"help": "Maximum generation token length"},
    )
    gen_seed: Optional[int] = field(
        default=0,
        metadata={"help": "Random seed"},
    )
    gen_output_dir: Optional[str] = field(
        default="./save",
        metadata={"help": "Output directory"},
    )
    print_raw_preds: bool = field(
        default=False,
        metadata={"help": "True will include all raw preds in the output eval csv"}
    )
    exploration_filter: Optional[str] = field(
        default="kndhbleu1",
        metadata={"help": "Filter on the nfirst token exploration"},
    )
    skip_special_tokens: bool = field(
        default=False,
        metadata={"help": "Should tokenizer decode skip special tokens?"}
    )
    explore_nfirst_token: bool = field(
        default=True,
        metadata={"help": "Decoding style is to explore first tokens' n_topk and filter for the best one. \
                           Setting to False means no exploration and will set n_token_add to 0"},
    )
    n_token_add: Optional[int] = field(
        default=0,
        metadata={"help": "How many n first token to explore? \
                            Set 0 to explore nothing whether or not the explore_nfirst_token==False or True,\
                            Set 1 to explore the first token if explore_nfirst_token==True,\
                            Set 2 to explore the first and second token if explore_nfirst_token==True, \
                            and 3 to explore the first until the third token if explore_nfirst_token==True, and so on"},
    )
    n_topk_in_nfirst_token: Optional[int] = field(
        default=32,
        metadata={"help": "n top k to be explored. Will not be used if explore_nfirst_token is set to 0."},
    )
    pandas_parallel_processing: Optional[int] = field(
        default=16,
        metadata={"help": "parallelize pandas.apply for the ones that has been implemented"},
    )
    temperature: Optional[float] = field(
        default=1.0,
        metadata={"help": "The value used to module the next token probabilities."},
    )
    typical_p: Optional[float] = field(
        default=1.0,
        metadata={"help": "The amount of probability mass from the original distribution to be considered in typical decoding. If set to 1.0 it takes no effect. See [this paper](https://arxiv.org/pdf/2202.00666.pdf) for more details."},
    )
    num_beam_groups: Optional[int] = field(
        default=1,
        metadata={"help": "Number of groups to divide `num_beams` into in order to ensure diversity among different groups of beams. [this paper](https://arxiv.org/pdf/1610.02424.pdf) for more details."},
    )
    diversity_penalty: Optional[float] = field(
        default=0.0,
        metadata={"help": "This value is subtracted from a beam's score if it generates a token same as any beam from other group at a particular time. Note that `diversity_penalty` is only effective if `group beam search` is enabled."},
    )
        
        
@dataclass
class TrainingArguments(TrainingArguments):
    """
    Arguments pertraining to the training pipeline.
    """
    output_dir: Optional[str] = field(
        default="./save",
        metadata={"help": "Output directory"},
    )
    eval_accumulation_steps: Optional[int] = field(
        default=None,
        metadata={"help": "Evaluation accumulation steps"}
    )
    seed: Optional[int] = field(
        default=0,
        metadata={"help": "Random seed"},
    )