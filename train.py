import data_constants
import data_loader
from language_index import LanguageIndex
import Model
import utils
import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split
import logging
import absl.logging
import argparse

BATCH_SIZE = 64


def create_train_val_dataset(input_sents, target_sents, inp_lang, targ_lang):
    input_tensor = [[inp_lang.word2idx[s] for s in sent.split(' ')] for sent in input_sents]
    target_tensor = [[targ_lang.word2idx[s] for s in sent.split(' ')] for sent in target_sents]
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        input_tensor, maxlen=inp_lang.max_sentence_length, padding='post')
    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(
        target_tensor, maxlen=targ_lang.max_sentence_length, padding='post')
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val =\
        train_test_split(input_tensor, target_tensor, test_size=0.2, random_state=42)
    train_set_size = len(input_tensor_train)
    val_set_size = len(input_tensor_val)
    train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(train_set_size)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(val_set_size)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    return train_dataset, val_dataset


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Model Train')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=data_constants.VERSION))
    parser.add_argument('-sd', '--seq2sent_dataset', default=data_constants.DEFAULT_SEQ2SENT_DATASET_PATH,
                        help='Path to seq2sent ConceptNet dataset')
    parser.add_argument('-cp', '--checkpoints_dir', default=data_constants.DEFAULT_CHECKPOINTS_PATH,
                        help='Path to model checkpoints dir')
    parser.add_argument('-e', '--epochs', type=utils.positive_int, default=10, help='Number of epochs to train')
    parser.add_argument('--eval', action='store_true', help='Evaluate loss on validation dataset during training')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    # Suppressing absl logger - see https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-507420022
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Starting with args: {}'.format(vars(args)))

    logging.info('Loading sequence to sentence dataset from {}'.format(args.seq2sent_dataset))
    input_sents, target_sents = data_loader.load_sequence_to_sentence_dataset(args.seq2sent_dataset)
    logging.info('Done loading')

    logging.info('Creating Model')
    inp_lang = LanguageIndex(input_sents)
    targ_lang = LanguageIndex(target_sents)
    model = Model.make_basic_model(inp_lang, targ_lang, args.checkpoints_dir)
    logging.info('Model is ready')

    logging.info('Creating train and evaluation datasets')
    train_dataset, val_dataset = create_train_val_dataset(input_sents, target_sents, inp_lang, targ_lang)
    logging.info('Datasets are ready')

    logging.info('Start training for {} epochs'.format(args.epochs))
    model.train(train_dataset, val_dataset if args.eval else None, epochs=args.epochs, batch_size=BATCH_SIZE)
    logging.info('Done')
