import data_constants
import data.uri as uri_helper
import pandas as pd


def load_normalized_dataset(h5_path=data_constants.DEFAULT_NORMALIZED_DATASET_PATH):
    return pd.read_hdf(h5_path, 'data')


def load_sequence_to_sentence_dataset(h5_path=data_constants.DEFAULT_SEQ2SENT_DATASET_PATH):
    return pd.read_hdf(h5_path, 'inputs'), pd.read_hdf(h5_path, 'targets')


def load_numberbatch(h5_path=data_constants.DEFAULT_NUMBERBATCH_PATH, lang=None):
    df = pd.read_hdf(h5_path)
    if lang:
        df = df[df.index.map(lambda x: uri_helper.get_uri_language(x) == lang)]
    return df