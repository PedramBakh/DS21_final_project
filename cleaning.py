import numpy as np
import re
import nltk
from concurrent.futures import ProcessPoolExecutor
from cleantext import constants as cleantext_re
import pandas as pd

non_ascii_pattern = re.compile(r'\w*[^\x00-\x7f]\w*')

punctuation_pattern = re.compile(r'[^\w|]')

nltk.download('stopwords')

tokenizer = nltk.RegexpTokenizer(r'(\w+|\|\w+\|)')

stopwords = nltk.corpus.stopwords.words('english')

stemmer = nltk.stem.PorterStemmer()

executor = ProcessPoolExecutor(max_workers=4)


def make_list_atomic(source_data, key, to_split, pattern):
    source_data[to_split + '_split'] = source_data[to_split].str.lower().str.findall(pattern)
    df = source_data[[key, to_split + '_split']][source_data[to_split + '_split'].notnull()]\
        .explode(to_split + '_split', ignore_index=True)
    df['article_index'] = df[key]
    df[to_split[:-1] + '_name'] = df[to_split + '_split']
    return df[['article_index', to_split[:-1] + '_name']]


def tokenize_filter_stem_join(input):
    return ' '.join([stemmer.stem(token) for token in tokenizer.tokenize(input) if token not in stopwords])


def clean_content(series, concurrent=False):
    series = series.astype(str)
    series = series \
        .str.replace(non_ascii_pattern, " ") \
        .str.replace(cleantext_re.EMAIL_REGEX, '|email|') \
        .str.replace(cleantext_re.URL_REGEX, '|url|') \
        .str.replace(cleantext_re.NUMBERS_REGEX, '|num|') \
        .str.replace(punctuation_pattern, " ") \
        .str.lower()

    if concurrent:
        input = series.to_numpy()

        results = executor.map(tokenize_filter_stem_join, input)
        for i, result in enumerate(results):
            series[i] = result

        return series
    else:
        return series.apply(tokenize_filter_stem_join)
