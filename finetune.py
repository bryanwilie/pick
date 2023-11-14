import os
import sys
import math
import torch
import logging
import datasets
import transformers
from tqdm import tqdm
from itertools import chain
from functools import cache
from transformers import (
    default_data_collator,
    set_seed,
    GPT2Tokenizer,
    GPT2LMHeadModel,
    EarlyStoppingCallback,
    HfArgumentParser,
    Trainer
)
from transformers.trainer_utils import get_last_checkpoint, is_main_process

from src.model.model import load_model_and_tokenizer
from src.dataset.load_dataset import load_dataset
from src.utils.train_args_helper import DataArguments, ModelArguments, TrainingArguments
from src.utils.trainer_helper import preprocess_logits_for_metrics, compute_metrics
from src.utils.general_helper import set_all_seeds

datasets.enable_caching()
logger = logging.getLogger(__name__)
       

def do_train(model_args, data_args, training_args):
    
    # Init
    set_seed(training_args.seed)
    set_all_seeds(training_args.seed)
    os.environ["TOKENIZERS_PARALLELISM"] = "false"
    
    ##### ARGS #####
    run_name = '{}_{}_{}_{}_{}dhuttr_nokn{}_npu{}_adddata{}_maxseqlen{}_bs{}_gradacc{}_lr{}_spktoken{}_{}pad_{}epoch_wd{}_ws{}'\
                    .format(model_args.model_name_or_path,
                            data_args.dataset_name,
                            data_args.experiment_mode,
                            data_args.eval_set,
                            data_args.dh_uttr_count,
                            data_args.no_knowledge_dataset,
                            data_args.no_passages_used_settings,
                            data_args.add_dataset,
                            model_args.max_seq_len,
                            training_args.per_device_train_batch_size,
                            training_args.gradient_accumulation_steps,
                            training_args.learning_rate,
                            data_args.use_speaker_token,
                            data_args.pad_token,
                            training_args.num_train_epochs,
                            training_args.weight_decay,
                            training_args.warmup_steps).replace('/', '-')
    
    
    ##### MODEL #####
    model, tokenizer = load_model_and_tokenizer(data_args, model_args)
    
    
    ##### DATASET #####
    
    # Load the preprocessed dataset splits
    dataset_dict = {}
    for split in ['train', data_args.eval_set]:
        dataset_dict[split] = load_dataset(data_args, model_args, tokenizer, split)
        
    
    ##### TRAINING #####
    print('Preparing Trainer...')
    training_args.output_dir = training_args.output_dir + run_name
    
    # Initialize Trainer
    trainer = Trainer(
        train_dataset=dataset_dict['train'],
        eval_dataset=dataset_dict[data_args.eval_set],
        model=model,
        data_collator=default_data_collator,
        args=training_args,
        compute_metrics=compute_metrics if training_args.do_eval else None,
        preprocess_logits_for_metrics=preprocess_logits_for_metrics if training_args.do_eval else None,
        callbacks=[EarlyStoppingCallback(early_stopping_patience=training_args.early_stopping_patience)]
    )

    ###
    # Training Phase
    ###

    if training_args.do_train:
        print('*** Training Phase ***')
        checkpoint = None

        # Detecting last checkpoint.
        last_checkpoint = None
        if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
            last_checkpoint = get_last_checkpoint(training_args.output_dir)
            if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
                raise ValueError(
                    f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                    "Use --overwrite_output_dir to overcome."
                )
            elif last_checkpoint is not None and training_args.resume_from_checkpoint is None:
                logger.info(
                    f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                    "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
                )

        if training_args.resume_from_checkpoint is not None:
            checkpoint = training_args.resume_from_checkpoint
        elif last_checkpoint is not None:
            checkpoint = last_checkpoint
        train_result = trainer.train(resume_from_checkpoint=checkpoint)

        ### Saving
        trainer.save_model() # Saves the tokenizer too for easy upload

        metrics = train_result.metrics
        metrics["train_samples"] = len(dataset_dict['train'])

        trainer.log_metrics("train", metrics)
        trainer.save_metrics("train", metrics)
        trainer.save_state()
    
    ###
    # Evaluation Phase
    ###
    
    if training_args.do_eval:
        print("*** Evaluation Phase ***")

        metrics = trainer.evaluate()
        metrics["eval_samples"] = len(dataset_dict[data_args.eval_set])
        
        try:
            perplexity = math.exp(metrics["eval_loss"])
        except OverflowError:
            perplexity = float("inf")
        metrics["perplexity"] = perplexity

        trainer.log_metrics("eval", metrics)
        trainer.save_metrics("eval", metrics)
        
        
#####
# Entry Point
#####
def main():

    ###
    # Parsing & Initialization
    ###
    
    # Parse argument
    parser = HfArgumentParser((ModelArguments, DataArguments, TrainingArguments))
    if len(sys.argv) == 2 and sys.argv[1].endswith(".json"):
        model_args, data_args, training_args = parser.parse_json_file(json_file=os.path.abspath(sys.argv[1]))
    else:
        model_args, data_args, training_args = parser.parse_args_into_dataclasses()
    
    
    # Set random seed
    set_seed(training_args.seed)
    
    # Detect last checkpoint
    last_checkpoint = None
    if os.path.isdir(training_args.output_dir) and training_args.do_train and not training_args.overwrite_output_dir:
        last_checkpoint = get_last_checkpoint(training_args.output_dir)
        if last_checkpoint is None and len(os.listdir(training_args.output_dir)) > 0:
            raise ValueError(
                f"Output directory ({training_args.output_dir}) already exists and is not empty. "
                "Use --overwrite_output_dir to overcome."
            )
        elif last_checkpoint is not None:
            logger.info(
                f"Checkpoint detected, resuming training at {last_checkpoint}. To avoid this behavior, change "
                "the `--output_dir` or add `--overwrite_output_dir` to train from scratch."
            )

    ###
    # Prepare logger
    ###
    
    # Init logging
    os.makedirs("./log", exist_ok=True)
    logging.basicConfig(
        format="%(asctime)s - %(levelname)s - %(name)s - %(message)s",
        datefmt="%m/%d/%Y %H:%M:%S",
        handlers=[logging.StreamHandler(sys.stdout), logging.FileHandler(
            "./log/log__{}".format(model_args.model_name_or_path.replace("/", "_")), mode="w")],
    )
    logger.setLevel(logging.INFO if is_main_process(training_args.local_rank) else logging.WARN)

    # Log on each process the small summary:
    logger.warning(
        f"Process rank: {training_args.local_rank}, device: {training_args.device}, n_gpu: {training_args.n_gpu}"
        f"distributed training: {bool(training_args.local_rank != -1)}, 16-bits training: {training_args.fp16}"
    )
    # Set the verbosity to warn of the Transformers logger (on main process only):
    if is_main_process(training_args.local_rank):
        transformers.utils.logging.set_verbosity(transformers.logging.WARNING)
    logger.info("Training/evaluation parameters %s", training_args)
    
    do_train(model_args, data_args, training_args)
    

if __name__ == '__main__':
    main()