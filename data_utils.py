import json
from data.uri import *

import pandas as pd
import argparse

VERSION = '0.0.1'
CONCEPTNET_VERSION = '5.7.0'
DEFAULT_DATASET_PATH = 'data/conceptnet-assertions-{}.csv.gz'.format(CONCEPTNET_VERSION)
DEFAULT_PROCESSED_DATASET_PATH = 'out/conceptnet-{}.h5'.format(CONCEPTNET_VERSION)

FILEPATH = "data/conceptnet-assertions-5.6.0.csv.gz"
NUMBERBATCH = 'weights/conceptnet-numberbatch.h5'

def load_data_iterator(filepath=FILEPATH, chunksize=20000):
    return pd.read_csv(filepath, delimiter='\t', chunksize=chunksize,
                       names=['uri', 'relation', 'subject', 'object', 'data'])


def create_input_and_target_from_df(df):
    df = filter_df(df)
    x = df['subject'] + " " + df['relation'] + " " + df['object']
    y = "<start> " + df['text'] + ' <end>'
    return x, y


def filter_df(df, include_empty_text=False):
    df = df[(df['language'] == 'en') & (df['subject_lang'] == 'en') & (df['object_lang'] == 'en')]
    df = df[(df['subject'].notna()) & (df['object'].notna())]
    if not include_empty_text:
        df = df[df['text'] != ""]
    df.reset_index()
    return df


def normalize_df(df):
    new_df = pd.DataFrame()
    new_df['relation'] = df['relation'].apply(uri_to_label)
    new_df['language'] = df['uri'].apply(get_uri_language)
    new_df['subject'] = df['subject'].apply(uri_to_label)
    new_df['subject_lang'] = df['subject'].apply(get_uri_language)
    new_df['object'] = df['object'].apply(uri_to_label)
    new_df['object_lang'] = df['object'].apply(get_uri_language)
    new_df['data'] = df['data'].apply(json.loads)
    new_df['dataset'] = new_df['data'].apply(lambda x: x['dataset'])
    new_df['weight'] = new_df['data'].apply(lambda x: x['weight'])
    new_df['text'] = new_df['data'].apply(lambda x: x.get('surfaceText', '').replace('[', '').replace(']', ''))

    return new_df

def save_normalized_and_filtered_knowledge_graph():
    data_iter = load_data_iterator()
    gen_df = pd.DataFrame()
    for chunk in data_iter:
        df = normalize_df(chunk)
        df = filter_df(df, False) #our current model doesn't know how to deal with unseen words - and was originally
                                  # trained on filtered words
        gen_df = pd.concat((gen_df, df))
        print(len(gen_df))
    gen_df.to_csv('data/normalized_knowledge_graph.csv', sep='\t')


def create_sequence_to_sentence_iterator(chunksize=50000):
    for chunk in load_data_iterator(chunksize=chunksize):
        df = normalize_df(chunk)
        inputs, outputs = create_input_and_target_from_df(df)
        yield inputs, outputs

def generate_sequence_to_sentece_dataset():
    data_it = create_sequence_to_sentence_iterator()
    total_input_sents = pd.Series()
    total_target_sents = pd.Series()
    for input_sents, target_sents in data_it:
        total_input_sents = pd.concat((total_input_sents, input_sents))
        total_target_sents = pd.concat((total_target_sents, target_sents))
        print(len(total_input_sents))
    return total_input_sents, total_target_sents

def save_sequence_to_sentence_dataset():
    tot_inputs, tot_ouputs = generate_sequence_to_sentece_dataset()
    new_df = pd.DataFrame()
    new_df['inputs'] = tot_inputs
    new_df['targets'] = tot_ouputs
    new_df.to_csv('data/filtered_concpetnet_data.csv', sep='\t')

def load_sequence_to_sentence_dataset():
    df = load_tsv('data/filtered_concpetnet_data.csv')
    return df['inputs'], df['targets']

def load_normalized_dataset():
    df = load_tsv('data/normalized_knowledge_graph.csv')
    return df

def load_tsv(filepath):
    return pd.read_csv(filepath, sep='\t')

def max_length(tensor):
    return max(len(t) for t in tensor)


def load_numberbatch_weights(filename=NUMBERBATCH, lang=None):
    df = pd.read_hdf(filename)
    if lang:
        mask = df.index.map(lambda x: get_uri_language(x) == lang)
        df = df[mask]
        df.reset_index()
    return df

def create_langauge_index(lang='en'):
    embeddings = load_numberbatch_weights(lang=lang)
    words = embeddings.index.map(uri_to_label).values
    return LanguageIndex(words)

class LanguageIndex():
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.vocab = set()
        self.max_sentence_length = 0
        self.create_index()

    def create_index(self):
        for phrase in self.lang:
            list_of_words = phrase.split(' ')
            self.max_sentence_length = max(self.max_sentence_length, len(list_of_words))
            self.vocab.update(list_of_words)
        self.vocab = sorted(self.vocab)

        self.word2idx['<pad>'] = 0
        for index, word in enumerate(self.vocab):
            self.word2idx[word] = index + 1

        for word, index in self.word2idx.items():
            self.idx2word[index] = word


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing ConceptNet')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=VERSION))
    parser.add_argument('-d', '--dataset', default=DEFAULT_DATASET_PATH, help='Path to ConceptNet dataset')
    parser.add_argument('-pd', '--processed_dataset', default=DEFAULT_PROCESSED_DATASET_PATH,
                        help='Path to processed ConceptNet dataset')
    parser.add_argument('--no_dataset', action='store_true', help='Don\'t process the dataset')
    args = parser.parse_args()
    print('Starting with args: {}'.format(vars(args)))

    #uncomment this to create the training set
    # save_sequence_to_sentence_dataset()
    #creates the dataset to generate sentences from
    save_normalized_and_filtered_knowledge_graph()