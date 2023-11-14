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
        metadata={"help": "Name of the chosen HF pretrained model"}
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
        default="right", 
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
    eval_set: Optional[str] = field(
        default="test_seen",
        metadata={"help": "Evaluation set to use as eval",
                  "choices": ["train", "val", "valid_seen", "valid_unseen", "validation", "test_seen", "test_unseen", "test"]}
    )
    experiment_mode: Optional[str] = field(
        default="tdhkn",
        metadata={"help": "Dataset alignment to construct the dataset as input"},
    )
    dh_uttr_count: Optional[int] = field(
        default=3,
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
        metadata={"help": "Set to empty string to use tokenizer.eos_token to pad different input length for batching"}
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
                  "choices": ["nosettings", "nan", "npu"]}
    )
    add_dataset: Optional[str] = field(
        default="no", 
        metadata={"help": "Augment the dataset used with the dataset of choice",
                  "choices": ["dailydialog", "no"]}
    )
    addprompt_w_n_gold_words: Optional[int] = field(
        default=0,
        metadata={"help": "Add n first words exists in gold, to the prompt"},
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
    early_stopping_patience: Optional[int] = field(
        default=100,
        metadata={"help": "Early stopping patience for EarlyStoppingCallback"}
    )
    load_best_model_at_end: bool = field(
        default=True, 
        metadata={"help": "Needed for EarlyStoppingCallback"}
    )