import data_constants
import data_loader
from language_index import LanguageIndex
import utils
import Model
import tensorflow as tf
import numpy as np
import logging
import absl.logging
import argparse
from collections import defaultdict


class NegativeSamplesGenerator:
    def __init__(self, graph_to_text, dataset):
        self.graph2text = graph_to_text
        self.dataset = dataset

    def generate_random_sentences(self, num_of_sequences, random_state):
        sequences = self._generate_with_same_relation(num_of_sequences, random_state)
        sentences = self.graph2text.generate_sentences_from_word_sequences(sequences)
        return sequences, sentences

    def _generate_with_same_relation(self, num_of_sequences, random_state):
        # the relations might be biased - so it might make sense to sample them by unique
        relations = self.dataset['relation'].sample(num_of_sequences, random_state=random_state)
        result = []
        for relation in relations:
            rel_subject = self.dataset.query('relation=="{}"'.format(relation))['subject']\
                .sample(1, random_state=random_state).item()
            rel_object = self.dataset.query('relation=="{}" and subject!="{}"'.format(relation, rel_subject))['object']\
                .sample(1, random_state=random_state).item()
            result.append(rel_subject + " " + relation + " " + rel_object)
        return result


class GeneratedSamplesFilter:
    def __init__(self, dataset):
        self.sequences = list(dataset['subject'] + ' ' + dataset['relation'] + ' ' + dataset['object'])

    def filter_sentences(self, sequences, sentences):
        for i in reversed(range(len(sentences))):
            if self._is_sequence_in_dataset(sequences[i]):
                logging.debug('Removing sequence in original dataset \'{}\''.format(sequences[i]))
                del sentences[i]
                del sequences[i]
            elif self._is_sentence_unreasonable(sentences[i]):
                logging.debug('Removing unreasonable sentence \'{}\''.format(sentences[i]))
                del sentences[i]
                del sequences[i]

    def _is_sequence_in_dataset(self, sequence):
        return sequence in self.sequences

    @staticmethod
    def _is_sentence_unreasonable(sentence):
        word_counts = defaultdict(int)
        for word in sentence.split():
            word_counts[word] += 1
        return max(word_counts.values()) > 3


if __name__ == '__main__':
    tf.compat.v1.enable_eager_execution()
    # import os
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    parser = argparse.ArgumentParser(description='Random Sentences Generator')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=data_constants.VERSION))
    parser.add_argument('-nd', '--normalized_dataset', default=data_constants.DEFAULT_NORMALIZED_DATASET_PATH,
                        help='Path to normalized ConceptNet dataset')
    parser.add_argument('-sd', '--seq2sent_dataset', default=data_constants.DEFAULT_SEQ2SENT_DATASET_PATH,
                        help='Path to seq2sent ConceptNet dataset')
    parser.add_argument('-cp', '--checkpoints_dir', default=data_constants.DEFAULT_CHECKPOINTS_PATH,
                        help='Path to model checkpoints dir')
    parser.add_argument('-o', '--out', default=data_constants.DEFAULT_GENERATED_SENTENCES_PATH,
                        help='Path to output generated sentences file')
    parser.add_argument('-n', '--num_of_sentences', type=utils.positive_int, default=2000,
                        help='Number of sentences to generate')
    parser.add_argument('-r', '--random_seed', type=int, help='Random seed to use')
    parser.add_argument('--no_filter', action='store_true', help='Do not filter badly generated sentences')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    # Suppressing absl logger - see https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-507420022
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Starting with args: {}'.format(vars(args)))

    logging.info('Loading normalized dataset from {}'.format(args.normalized_dataset))
    dataset = data_loader.load_normalized_dataset(args.normalized_dataset)
    logging.info('Done loading')

    logging.info('Loading sequence to sentence dataset from {}'.format(args.seq2sent_dataset))
    input_sents, target_sents = data_loader.load_sequence_to_sentence_dataset(args.seq2sent_dataset)
    logging.info('Done loading')

    logging.info('Creating Model')
    inp_lang = LanguageIndex(input_sents)
    targ_lang = LanguageIndex(target_sents)
    graph2text = Model.make_basic_model(inp_lang, targ_lang, args.checkpoints_dir)
    logging.info('Model is ready')

    logging.info('Generating {} random sentences'.format(args.num_of_sentences))
    generator = NegativeSamplesGenerator(graph2text, dataset)
    gs_filter = GeneratedSamplesFilter(dataset)

    random_sequences = []
    random_sentences = []
    random_state = np.random.RandomState(args.random_seed)
    while len(random_sentences) < args.num_of_sentences:
        sequences, sentences = generator.generate_random_sentences(args.num_of_sentences - len(random_sentences),
                                                                   random_state)
        if not args.no_filter:
            gs_filter.filter_sentences(sequences, sentences)
        random_sequences.extend(sequences)
        random_sentences.extend(sentences)
    logging.info('Done generating, will write outputs to {}'.format(args.out))
    with open(args.out, 'w', encoding='utf-8') as f:
        for seq, sent in zip(random_sequences, random_sentences):
            f.write(seq + '\n' + sent + '\n')
    logging.info('Done')
