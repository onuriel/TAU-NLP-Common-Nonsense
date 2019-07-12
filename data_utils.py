import json
from data.uri import *

import sys
import logging
import pandas as pd
import argparse
import pathlib

VERSION = '0.0.1'
CONCEPTNET_VERSION = '5.7.0'
NUMBERBATCH_VERSION = '17.06'
DEFAULT_DATASET_PATH = 'data/conceptnet-assertions-{}.csv.gz'.format(CONCEPTNET_VERSION)
DEFAULT_NORMALIZED_DATASET_PATH = 'out/normalized_conceptnet-{}.h5'.format(CONCEPTNET_VERSION)
DEFAULT_SEQ2SENT_DATASET_PATH = 'out/seq2sent_conceptnet-{}.h5'.format(CONCEPTNET_VERSION)
DEFAULT_NUMBERBATCH_PATH = 'data/conceptnet-numberbatch-mini-{}.h5'.format(NUMBERBATCH_VERSION)


class _PreProcessor:
    def __init__(self, dataset, normalized_dataset_path, seq2sent_dataset_path):
        self.dataset = pathlib.Path(dataset)
        self.normalized_dataset_path = pathlib.Path(normalized_dataset_path)
        self.seq2sent_dataset_path = pathlib.Path(seq2sent_dataset_path)
        self.normalized_df = None

    def make_normalized_dataset(self, override, filter_lines):
        logging.info('Making normalized dataset {} into {}'.format(self.dataset, self.normalized_dataset_path))
        if not self.dataset.exists():
            self._fail('Input dataset {} does not exist'.format(self.dataset))
        if self.normalized_dataset_path.exists():
            if not override:
                self._fail('Output normalized dataset {} already exist, use \'--override\' to override it'.format(
                    self.normalized_dataset_path))
            else:
                logging.info('Overriding output normalized dataset {} since \'--override\' is given'.format(
                    self.normalized_dataset_path))
        self.normalized_dataset_path.parent.mkdir(parents=True, exist_ok=True)
        self.normalized_df = pd.concat(
            [self._filter_df(self._normalize_df(chunk)) for chunk in self._iterate_dataset(filter_lines)])
        self.normalized_df.to_hdf(self.normalized_dataset_path, 'data', mode='w')
        logging.info('Done making normalized dataset')

    def make_seq2sent_dataset(self, override, filter_lines):
        logging.info('Making Seq2Sent dataset {} into {}'.format(self.dataset, self.seq2sent_dataset_path))
        if not self.dataset.exists():
            self._fail('Input dataset {} does not exist'.format(self.dataset))
        if self.seq2sent_dataset_path.exists():
            if not override:
                self._fail('Output Seq2Sent dataset {} already exist, use \'--override\' to override it'.format(
                    self.seq2sent_dataset_path))
            else:
                logging.info('Overriding output Seq2Sent dataset {} since \'--override\' is given'.format(
                    self.seq2sent_dataset_path))
        self.seq2sent_dataset_path.parent.mkdir(parents=True, exist_ok=True)

        if self.normalized_df is None:
            if override or not self.normalized_dataset_path.exists():
                self.make_normalized_dataset(override, filter_lines)
            else:
                self.normalized_df = load_normalized_dataset(self.normalized_dataset_path)

        inputs, targets = self._create_input_and_target_from_df(self.normalized_df)
        inputs.to_hdf(self.seq2sent_dataset_path, 'inputs', mode='w')
        targets.to_hdf(self.seq2sent_dataset_path, 'targets', mode='a')
        logging.info('Done making Seq2Sent dataset')

    def _iterate_dataset(self, filter_lines):
        return pd.read_csv(self.dataset, delimiter='\t', chunksize=50000,
                           names=['uri', 'relation', 'subject', 'object', 'data'],
                           skiprows=lambda x: x % filter_lines != 0)

    @staticmethod
    def _normalize_df(df):
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

    @staticmethod
    def _filter_df(df, include_empty_text=False):
        df.query('language=="en" and subject_lang=="en" and object_lang=="en"', inplace=True)
        df.dropna(subset=['subject', 'object'], inplace=True)
        if not include_empty_text:
            df.query('text != ""', inplace=True)
        return df

    @staticmethod
    def _create_input_and_target_from_df(df):
        x = df['subject'] + ' ' + df['relation'] + ' ' + df['object']
        y = '<start> ' + df['text'] + ' <end>'
        return x, y

    @staticmethod
    def _fail(err_msg):
        logging.error(err_msg)
        sys.exit(1)


def load_normalized_dataset(h5_path=DEFAULT_NORMALIZED_DATASET_PATH):
    return pd.read_hdf(h5_path, 'data')


def load_sequence_to_sentence_dataset(h5_path=DEFAULT_SEQ2SENT_DATASET_PATH):
    return pd.read_hdf(h5_path, 'inputs'), pd.read_hdf(h5_path, 'targets')


def load_numberbatch(h5_path=DEFAULT_NUMBERBATCH_PATH, lang=None):
    df = pd.read_hdf(h5_path)
    if lang:
        df = df[df.index.map(lambda x: get_uri_language(x) == lang)]
    return df


def create_langauge_index(lang='en'):
    embeddings = load_numberbatch(lang=lang)
    words = embeddings.index.map(uri_to_label).values
    return LanguageIndex(words)


class LanguageIndex:
    def __init__(self, lang):
        self.lang = lang
        self.word2idx = {}
        self.idx2word = {}
        self.max_sentence_length = 0
        self._create_index()

    def _create_index(self):
        vocab = set()
        for phrase in self.lang:
            list_of_words = phrase.split(' ')
            self.max_sentence_length = max(self.max_sentence_length, len(list_of_words))
            vocab.update(list_of_words)
        vocab = sorted(vocab)

        self.word2idx['<pad>'] = 0
        self.idx2word[0] = '<pad>'
        for index, word in enumerate(vocab):
            self.word2idx[word] = index + 1
            self.idx2word[index + 1] = word


def positive_int(value):
    ivalue = int(value)
    if ivalue <= 0:
        raise argparse.ArgumentTypeError('{} is not a positive integer '.format(ivalue))
    return ivalue


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Preprocessing ConceptNet')
    parser.add_argument('--version', action='version', version='%(prog)s {version}'.format(version=VERSION))
    parser.add_argument('-d', '--dataset', default=DEFAULT_DATASET_PATH, help='Path to ConceptNet dataset')
    parser.add_argument('-nd', '--normalized_dataset', default=DEFAULT_NORMALIZED_DATASET_PATH,
                        help='Path to normalized ConceptNet dataset')
    parser.add_argument('--no_normalize', action='store_true', help='Don\'t normalize the dataset')
    parser.add_argument('-sd', '--seq2sent_dataset', default=DEFAULT_SEQ2SENT_DATASET_PATH,
                        help='Path to seq2sent ConceptNet dataset')
    parser.add_argument('--no_seq2sent', action='store_true', help='Don\'t create seq2sent dataset')
    parser.add_argument('--filter_lines', default=1, type=positive_int,
                        help='For Dev/QA purposes only. Filter lines, only read 1/X lines from the dataset')
    parser.add_argument('--override', action='store_true', help='Override outputs if already exist')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Starting with args: {}'.format(vars(args)))

    preprocessor = _PreProcessor(args.dataset, args.normalized_dataset, args.seq2sent_dataset)
    if not args.no_normalize:
        preprocessor.make_normalized_dataset(args.override, args.filter_lines)
    if not args.no_seq2sent:
        preprocessor.make_seq2sent_dataset(args.override, args.filter_lines)
