import data_utils
import data_loader
from Model import Encoder, Decoder, GraphToText
import tensorflow as tf
import numpy as np
import pandas as pd


class NegativeSamplesGenerator(object):
    def __init__(self, graph_to_text, dataset):
        self.graph2text = graph_to_text
        self.dataset = dataset

    def generate_random_sentences(self, num_of_sequences, random_state=None):
        word_sequences = self._generate_random_sequences([['subject'], ['relation'], ['object']], num_of_sequences,
                                                         random_state)
        return word_sequences, self.graph2text.generate_sentences_from_word_sequences(word_sequences)

    def generate_semi_random_sentences(self, num_of_sequences, random_state=None):
        word_sequences = self._generate_semi_random_sequences(num_of_sequences, random_state)
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
        res_dict = {}
        for keys_list in list_of_keys_list:
            sampled_vals = self.dataset[keys_list].sample(num_of_sequences, random_state=random_state).reset_index()
            res_dict.update({key: sampled_vals[key] for key in keys_list})
            if random_state:
                random_state += 1
        return res_dict['subject'] + ' ' + res_dict['relation'] + ' ' + res_dict['object']


def main():
    tf.compat.v1.enable_eager_execution()
    dataset = data_loader.load_normalized_dataset()
    input_sents, target_sents = data_loader.load_sequence_to_sentence_dataset()
    inp_lang = data_utils.LanguageIndex(input_sents)
    targ_lang = data_utils.LanguageIndex(target_sents)
    embedding_dim = 256
    units = 1024
    encoder = Encoder(embedding_dim, units, inp_lang)
    decoder = Decoder(embedding_dim, units, targ_lang)
    optimizer = tf.compat.v1.train.AdamOptimizer()
    graph2text = GraphToText(decoder, encoder, optimizer)
    generator = NegativeSamplesGenerator(graph2text, dataset)
    random_sequences, random_sentences = generator.generate_semi_random_sentences(2000)
    with open('generated_sentences.txt', 'w', encoding='utf-8') as f:
        for i in range(len(random_sentences)):
            f.write(random_sequences[i] + "\n" + random_sentences[i] + '\n')


if __name__ == '__main__':
    main()
