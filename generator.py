import data_constants
import data_utils
import data_loader
import utils
import Model
import tensorflow as tf
import numpy as np
import pandas as pd
import logging
import absl.logging
import argparse


class NegativeSamplesGenerator:
    def __init__(self, graph_to_text, dataset):
        self.graph2text = graph_to_text
        self.dataset = dataset

    def generate_random_sentences(self, num_of_sequences, semi, random_state):
        if semi:
            word_sequences = self._generate_semi_random_sequences(num_of_sequences, random_state)
        else:
            word_sequences = self._generate_random_sequences([['subject'], ['relation'], ['object']], num_of_sequences,
                                                             random_state)
        return word_sequences, self.graph2text.generate_sentences_from_word_sequences(word_sequences)

    def _generate_semi_random_sequences(self, num_of_sequences, random_state):
        probs = np.random.randint(0, 3, num_of_sequences)
        odd_subject = self._generate_random_sequences([['relation', 'object'], ['subject']], (probs == 0).sum(),
                                                      random_state)
        odd_relation = self._generate_random_sequences([['subject', 'object'], ['relation']], (probs == 1).sum(),
                                                       random_state + 2 if random_state else None)
        odd_object = self._generate_random_sequences([['subject', 'relation'], ['object']], (probs == 2).sum(),
                                                     random_state + 4 if random_state else None)
        return pd.concat((odd_subject, odd_relation, odd_object), ignore_index=True)

    def _generate_random_sequences(self, list_of_keys_list, num_of_sequences, random_state):
        logging.debug('Generating {} sequences with random_state={} and keys list {}'.format(
            num_of_sequences, random_state, list_of_keys_list))
        res_dict = {}
        for keys_list in list_of_keys_list:
            sampled_vals = self.dataset[keys_list].sample(num_of_sequences, random_state=random_state).reset_index()
            res_dict.update({key: sampled_vals[key] for key in keys_list})
            if random_state:
                random_state += 1
        return res_dict['subject'] + ' ' + res_dict['relation'] + ' ' + res_dict['object']


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Random Sentences Generator')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=data_constants.VERSION))
    parser.add_argument('-nd', '--normalized_dataset', default=data_constants.DEFAULT_NORMALIZED_DATASET_PATH,
                        help='Path to normalized ConceptNet dataset')
    parser.add_argument('-sd', '--seq2sent_dataset', default=data_constants.DEFAULT_SEQ2SENT_DATASET_PATH,
                        help='Path to seq2sent ConceptNet dataset')
    parser.add_argument('-o', '--out', default=data_constants.DEFAULT_GENERATED_SENTENCES_PATH,
                        help='Path to output generated sentences file')
    parser.add_argument('-n', '--num_of_sentences', type=utils.positive_int, default=2000,
                        help='Number of sentences to generate')
    parser.add_argument('-r', '--random_state', type=int, help='Random state (seed) to use')
    parser.add_argument('-s', '--semi_random', action='store_true', help='Generate semi-random sentences')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    # Suppressing absl logger - see https://github.com/tensorflow/tensorflow/issues/26691#issuecomment-507420022
    logging.root.removeHandler(absl.logging._absl_handler)
    absl.logging._warn_preinit_stderr = False
    logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Starting with args: {}'.format(vars(args)))

    tf.compat.v1.enable_eager_execution()
    logging.info('Loading normalized dataset from {}'.format(args.normalized_dataset))
    dataset = data_loader.load_normalized_dataset(args.normalized_dataset)
    logging.info('Done loading')
    logging.info('Loading sequence to sentence dataset from {}'.format(args.seq2sent_dataset))
    input_sents, target_sents = data_loader.load_sequence_to_sentence_dataset(args.seq2sent_dataset)
    logging.info('Done loading')
    inp_lang = data_utils.LanguageIndex(input_sents)
    targ_lang = data_utils.LanguageIndex(target_sents)
    embedding_dim = 256
    units = 1024
    encoder = Model.Encoder(embedding_dim, units, inp_lang)
    decoder = Model.Decoder(embedding_dim, units, targ_lang)
    optimizer = tf.compat.v1.train.AdamOptimizer()
    graph2text = Model.GraphToText(decoder, encoder, optimizer)
    generator = NegativeSamplesGenerator(graph2text, dataset)
    logging.info('Generating {} {}random sentences'.format(args.num_of_sentences, 'semi ' if args.semi_random else ''))
    random_sequences, random_sentences = generator.generate_random_sentences(args.num_of_sentences, args.semi_random,
                                                                             args.random_state)
    logging.info('Done generating, will write outputs to {}'.format(args.out))
    with open(args.out, 'w', encoding='utf-8') as f:
        for seq, sent in zip(random_sequences, random_sentences):
            f.write(seq + '\n' + sent + '\n')
    logging.info('Done')
