VERSION = '0.0.1'
CONCEPTNET_VERSION = '5.7.0'
NUMBERBATCH_VERSION_MINI = '17.06'
NUMBERBATCH_VERSION_FULL_ENG = '19.08'


DEFAULT_DATASET_PATH = 'data/conceptnet-assertions-{}.csv.gz'.format(CONCEPTNET_VERSION)
DEFAULT_NUMBERBATCH_MINI_PATH = 'data/conceptnet-numberbatch-mini-{}.h5'.format(NUMBERBATCH_VERSION_MINI)
DEFAULT_NUMBERBATCH_PATH = 'data/numberbatch-en-{}.txt.gz'.format(NUMBERBATCH_VERSION_FULL_ENG)
DEFAULT_CHECKPOINTS_PATH = 'training_checkpoints'
DEFAULT_NORMALIZED_DATASET_PATH = 'out/normalized_conceptnet-{}.h5'.format(CONCEPTNET_VERSION)
DEFAULT_SEQ2SENT_DATASET_PATH = 'out/seq2sent_conceptnet-{}.h5'.format(CONCEPTNET_VERSION)
DEFAULT_GENERATED_SENTENCES_PATH = 'out/generated_sentences-{}.txt'.format(CONCEPTNET_VERSION)
DEFAULT_RANKED_SENTENCES_PATH = 'out/ranked_sentences-{}.txt'.format(CONCEPTNET_VERSION)
DEFAULT_CONCEPTNET_DATASET = 'data/conceptnet_dataset1.h5'
DEFAULT_CONCEPTNET_GENERATED_DATASET = 'data/conceptnet_dataset2.h5'
DEFAULT_RANDOM_GENERATED_DATASET = 'data/generated_dataset.h5'
