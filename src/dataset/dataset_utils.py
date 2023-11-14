import os
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk import word_tokenize
from nltk.tokenize import RegexpTokenizer

nltk_tokenizer = RegexpTokenizer(r'\w+')
stopwords_english = stopwords.words('english')



def count_resp_in_reference(response, reference):
    count = 0
    for i, r_word in enumerate(response.split()):
        if r_word in reference:
            count += 1
    return count


def nltk_tokenizer_tokenize(input_string):
    if input_string is not np.nan and input_string is not None:
        if len(input_string) > 0:
            output_string = " ".join(nltk_tokenizer.tokenize(input_string))
        else:
            output_string = ""
    else:
        output_string = ""
    return output_string


def nltk_word_tokenize(input_string):
    if input_string is not np.nan and input_string is not None:
        if len(input_string) > 0:
            output_string = " ".join(word_tokenize(input_string))
        else:
            output_string = ""
    else:
        output_string = ""
    return output_string


def remove_resp_if_in_reference(response, reference):
    filtered_response = ""
    if response is not np.nan and response is not None:
        if len(response) > 0:
            for i, r_word in enumerate(response.split()):
                if r_word not in reference:
                    filtered_response+=r_word+" "
            return filtered_response.strip()
        else:
            return filtered_response
    else:
        return filtered_response