VERSION = '0.0.1'
CONCEPTNET_VERSION = '5.7.0'
NUMBERBATCH_VERSION_MINI = '17.06'
NUMBERBATCH_VERSION_FULL_ENG = '19.08'


DEFAULT_DATASET_PATH = 'data/conceptnet-assertions-{}.csv.gz'.format(CONCEPTNET_VERSION)
DEFAULT_NUMBERBATCH_PATH = 'data/conceptnet-numberbatch-mini-{}.h5'.format(NUMBERBATCH_VERSION_MINI)
#DEFAULT_NUMBERBATCH_PATH = 'data/numberbatch-en-{}.txt.gz'.format(NUMBERBATCH_VERSION_FULL_ENG)
DEFAULT_CHECKPOINTS_PATH = 'training_checkpoints'
DEFAULT_NORMALIZED_DATASET_PATH = 'out/normalized_conceptnet-{}.h5'.format(CONCEPTNET_VERSION)
DEFAULT_SEQ2SENT_DATASET_PATH = 'out/seq2sent_conceptnet-{}.h5'.format(CONCEPTNET_VERSION)
DEFAULT_GENERATED_SENTENCES_PATH = 'out/generated_sentences-{}.txt'.format(CONCEPTNET_VERSION)
DEFAULT_FILTERED_SENTENCES_PATH = 'out/filtered_sentences-{}.txt'.format(CONCEPTNET_VERSION)