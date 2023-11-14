import os
import re
import math
import datasets
import collections
import numpy as np
import pandas as pd
from nltk import word_tokenize
from fed import fed
from caffeinated_pandas.caffeinated_pandas_utils import multiproc_dataframe

from src.dataset.wow_json import WoWJSONDataset
from src.utils.trainer_helper import preprocess_logits_for_metrics, compute_metrics
from src.metrics.knowexpert_evaluation import get_unigram_F1, get_unigram_precision, get_unigram_recall,\
                                                get_bigram_F1, get_bigram_precision, get_bigram_recall


bleu = datasets.load_metric('bleu')
rouge = datasets.load_metric('rouge')
sacrebleu = datasets.load_metric('sacrebleu')


def corpuswise_generation_metrices(df, model=None, tokenizer=None, split=None, 
                                   data_args=None, model_args=None, training_args=None,
                                   pred_colname='preds',
                                   gold_colname='golds',
                                   kn_colname='knowledges',
                                   dh_colname='dialogue_histories'):
    
    list_hyp = df[pred_colname].values.tolist()
    list_label = df[gold_colname].values.tolist()
    list_kn = df[kn_colname].values.tolist()
    list_dh = df[dh_colname].values.tolist()

    # hyp and label are both list of string
    list_hyp_bleu = list(map(lambda x: word_tokenize(x), list_hyp))
    list_label_bleu = list(map(lambda x: [word_tokenize(x)], list_label))
    list_label_sacrebleu = list(map(lambda x: [x], list_label))
    
    metrics = {}
    metrics['sample_size'] = df.shape[0]
    
    metrics["BLEU-1"] = bleu._compute(list_hyp_bleu, list_label_bleu, max_order=1)['bleu'] * 100
    metrics["BLEU-2"] = bleu._compute(list_hyp_bleu, list_label_bleu, max_order=2)['bleu'] * 100
    metrics["BLEU-3"] = bleu._compute(list_hyp_bleu, list_label_bleu, max_order=3)['bleu'] * 100
    metrics["BLEU-4"] = bleu._compute(list_hyp_bleu, list_label_bleu, max_order=4)['bleu'] * 100
    metrics["SacreBLEU"] = sacrebleu._compute(list_hyp, list_label_sacrebleu)['score']
    
    rouge_score = rouge._compute(list_hyp,list_label)
    metrics["ROUGE1"] = rouge_score['rouge1'].mid.fmeasure * 100
    metrics["ROUGE2"] = rouge_score['rouge2'].mid.fmeasure * 100
    metrics["ROUGEL"] = rouge_score['rougeL'].mid.fmeasure * 100
    metrics["ROUGELsum"] = rouge_score['rougeLsum'].mid.fmeasure * 100
    
    metrics['unigram-F1'] = get_unigram_F1(list_hyp, list_label)
    metrics['unigram-precision'] = get_unigram_precision(list_hyp, list_label)
    metrics['unigram-recall'] = get_unigram_recall(list_hyp, list_label)
    metrics['bigram-F1'] = get_bigram_F1(" ".join(list_hyp), " ".join(list_label))
    metrics['bigram-precision'] = get_bigram_precision(" ".join(list_hyp), " ".join(list_label))
    metrics['bigram-recall'] = get_bigram_recall(" ".join(list_hyp), " ".join(list_label))
    
    return metrics



def filter_by_rowwise_generation_metrices(df, gen_args, save_path,
                                          pred_colname='preds',
                                          gold_colname='golds',
                                          kn_colname='knowledges',
                                          dh_colname='dialogue_histories'):

    def words(text):
           pattern = re.compile(r"[^\s]+")
           non_alpha = re.compile(r"[^a-z]", re.IGNORECASE)
           for match in pattern.finditer(text):
               nxt = non_alpha.sub("", match.group()).lower()
               if nxt:  # skip blank, non-alpha words
                   yield nxt

    def phrases(words):
            phrase = []
            for word in words:
                phrase.append(word)
                if len(phrase) > 3:
                    phrase.remove(phrase[0])
                if len(phrase) == 3:
                    yield tuple(phrase)

    def check_repetition(text):
        counts = collections.defaultdict(int)
        for phrase in phrases(words(text)):
                counts[phrase] += 1

        is_repetitive = sum([v > 1 for v in counts.values()]) > 0
        return is_repetitive
    
    def check_impossible_words(text):
        return sum([len(word) > 30 for word in text.split()]) > 0
    
    
    def neutralizes_strings_uniquecases(string):
        if pd.isna(string):
            string = str(string)
        else:
            string = re.sub(' +', ' ', string)
            string = str(np.nan) if len(string)<3 else string
            
        return string
    
    def prep_pred_bleu(list_of_string):
        return list(map(lambda x: word_tokenize(x), list_of_string))

    def prep_gold_bleu(list_of_string):
        return list(map(lambda x: [word_tokenize(x)], list_of_string))
    
    def prep_gold_sacrebleu(list_of_string):
        return list(map(lambda x: [x], list_of_string))
    
    def compute_bleu(df, pred_colname, gold_colname, max_order):
        df['score'] = df.apply(lambda x: bleu._compute(\
                                        prep_pred_bleu([x[pred_colname]]),
                                        prep_gold_bleu([x[gold_colname]]),
                                        max_order=max_order)['bleu']*100,
                                      axis=1)
        return df
    
    def compute_rouge(df, pred_colname, gold_colname, kind):
        df['score'] = df.apply(lambda x: rouge._compute([x[pred_colname]], [x[gold_colname]])\
                                            ['rouge'+str(kind)].mid.fmeasure*100, axis=1)
        return df
    
    def compute_sacrebleu(df, pred_colname, gold_colname):
        df['score'] = df.apply(lambda x: sacrebleu._compute(\
                                                [x[pred_colname]],
                                                prep_gold_sacrebleu([x[gold_colname]]))['score'],
                                              axis=1)
        return df
    
    def compute_unigram_metrics(df, func, pred_colname, gold_colname):
        df['score'] = df.apply(lambda x: func([x[pred_colname]], [x[gold_colname]]), axis=1)
        return df
    
    def compute_bigram_metrics(df, func, pred_colname, gold_colname):
        df['score'] = df.apply(lambda x: func(x[pred_colname], x[gold_colname]), axis=1)
        return df
    
    
    # Measure all the metrices for all the topk generations
    analysed_path = save_path[:-5]+'_analysed.gzip'
    selector_column = gen_args.exploration_filter

    def check_availability(analysed_path, selector_column):
        if os.path.isfile(analysed_path):
            if selector_column in pd.read_csv(analysed_path, compression='gzip').columns:
                return True
            else:
                return False
        else:
            return False    

    if check_availability(analysed_path, selector_column):
        df = pd.read_csv(analysed_path, compression='gzip')
        df.fillna('', inplace=True)

    else:
        # Freeing from errors
        df[pred_colname] = df.apply(lambda x: neutralizes_strings_uniquecases(x[pred_colname]), axis=1)
        df[gold_colname] = df.apply(lambda x: neutralizes_strings_uniquecases(x[gold_colname]), axis=1)
        df[kn_colname] = df.apply(lambda x: neutralizes_strings_uniquecases(x[kn_colname]), axis=1)
        df[dh_colname] = df.apply(lambda x: neutralizes_strings_uniquecases(x[dh_colname]), axis=1)

        colprefix = ['', 'KN-', 'DH-']
        colsuffix = ['F1']
        for i, ref_colname in enumerate([gold_colname, kn_colname, dh_colname]):

            for n in [1,2,3,4]:
                df[colprefix[i]+'BLEU-'+str(n)] = multiproc_dataframe(function=compute_bleu,
                                                                      df=df[[pred_colname, ref_colname]],
                                                                      pred_colname=pred_colname,
                                                                      gold_colname=ref_colname,
                                                                      max_order=n, 
                                                                      folderpath=save_path[:save_path.rfind('/')],
                                                                      procs=gen_args.pandas_parallel_processing)['score']

            for j, metric_func in enumerate([get_unigram_F1]):
                df[colprefix[i]+'unigram-'+colsuffix[j]] = multiproc_dataframe(function=compute_unigram_metrics,
                                                                                 df=df[[pred_colname, ref_colname]],
                                                                                 func=metric_func,
                                                                                 pred_colname=pred_colname,
                                                                                 gold_colname=ref_colname,
                                                                                 folderpath=save_path[:save_path.rfind('/')],
                                                                                 procs=gen_args.pandas_parallel_processing)['score']
                
            df[colprefix[i]+'SacreBLEU'] = multiproc_dataframe(function=compute_sacrebleu,
                                                               df=df[[pred_colname, ref_colname]],
                                                               pred_colname=pred_colname,
                                                               gold_colname=ref_colname,
                                                               folderpath=save_path[:save_path.rfind('/')],
                                                               procs=gen_args.pandas_parallel_processing)['score']

        df['is_repetitive'] = df.apply(lambda x: check_repetition(x[pred_colname]), axis=1)
        df['contains_impossible_words'] = df.apply(lambda x: check_impossible_words(x[pred_colname]), axis=1)

        # Derive and choose the selector metrics

        # Choose selector column
        if selector_column == 'kndhbleu1':
            df[selector_column] = df['KN-BLEU-1'] + df['DH-BLEU-1']
        elif selector_column == 'knunigramf1':
            df[selector_column] = df['KN-unigram-F1']
        elif selector_column == 'unigramf1':
            df[selector_column] = df['unigram-F1']
        elif selector_column == 'fed_turnlv_bsc_knunigramf1':
            
            ############ FED ############
            # Load model
            print('============ Calculating FED ============', flush=True)
            model, tokenizer = fed.load_models("microsoft/DialoGPT-large")

            df['fed_input'] = df.apply(lambda x: tokenizer.eos_token + ' ' + \
                                        x['dialogue_histories'] + ' ' + \
                                        tokenizer.eos_token + ' ' + x['preds'], axis=1)
            df['fed_scores'] = df.apply(lambda x: fed.evaluate(x['fed_input'],\
                                        model, tokenizer), axis=1)

            # get string dict value to df column
            df_fed = pd.json_normalize(df['fed_scores'])
            new_columns = ['fed_' + key for key in list(df_fed.columns)]
            df_fed.columns = new_columns
            for column in df_fed.columns:
                df[column] = df_fed[column]
            
            fed_combination = {"fed_turnlv_bsc_metrics" : ['fed_semantically appropriate', 'fed_understandable', 'fed_fluent']}

            for fed_metric in ['fed_turnlv_bsc_metrics']:
                for kn_metric in ['KN-unigram-F1']:

                    result_key = fed_metric + '+' + kn_metric
                    metric_list = fed_combination[fed_metric]
                    df[result_key] = df[metric_list[0]]
                    for metric in metric_list[1:]:
                        df[result_key] += df[metric]

            df[result_key] += df[kn_metric]
            df[selector_column] = df[result_key]
        else:
            NotImplementedError('Selection criteria: {} is not yet implemented'.\
                                          format(selector_column))
            
        df.to_csv(analysed_path, index=False, compression='gzip')
    
    # Filter and select
    filtered_path = save_path[:-5]+'_filtered_'+gen_args.exploration_filter+'.gzip'
    if os.path.isfile(filtered_path):
        df_selected = pd.read_csv(filtered_path, compression='gzip')
        df_selected.fillna('', inplace=True)
    else:
        # Filter for clean generation
        df_clean = df[(~df['is_repetitive']) & (~df['contains_impossible_words'])]

        # Sometimes model didn't generate any generation clean from repetition and impossible words
        if df_clean['sample_group'].nunique() != df['sample_group'].nunique():
            df_list = [df_clean]
            for i in set(np.arange(0, df['sample_group'].nunique(), 1)) - set(df_clean['sample_group'].unique()):
                df_list.append(df[df['sample_group']==i])
            df_clean = pd.concat(df_list)
            df_clean = df_clean.sort_values('index', ascending=True)

        # Filter and find the row with matching criterion
        groups = df_clean.groupby('sample_group', group_keys=False)
        df_selected = groups.apply(lambda x: x.sort_values(selector_column, ascending=False).head(1))
        df_selected.to_csv(filtered_path, index=False, compression='gzip')
    
    return df_selected