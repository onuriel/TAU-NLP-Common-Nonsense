import data_utils
from Model import Encoder, Decoder, GraphToText
from data_utils import load_normalized_dataset
import tensorflow as tf
import numpy as np
import pandas as pd

class NegativeSamplesGenerator(object):

    def __init__(self, graph_to_text, dataset):
        self.graph2text = graph_to_text
        self.dataset = dataset

    def generate_random_sequences(self, num_of_sequences, random_state=None):
        subject_random_state, objects_random_state = None, None
        if random_state:
            subject_random_state = random_state + 1
            objects_random_state = random_state - 1
        subjects = self.dataset['subject'].sample(num_of_sequences, random_state=subject_random_state).reset_index(drop=True)
        relations = self.dataset['relation'].sample(num_of_sequences, random_state=random_state).reset_index(drop=True)
        objects = self.dataset['object'].sample(num_of_sequences, random_state=objects_random_state).reset_index(drop=True)
        return subjects + " " + relations + " " + objects

    def generate_random_sentences(self, num_of_sequences, random_state=None):
        word_sequences = self.generate_random_sequences(num_of_sequences, random_state=random_state)
        return word_sequences, self.graph2text.generate_sentences_from_word_sequences(word_sequences)

    def generate_semi_random_sequences(self, num_of_sequences):
        probs = np.random.randint(0, 3, num_of_sequences)
        zeros = (probs == 0).sum()
        ones = (probs == 1).sum()
        twos = (probs == 2).sum()
        return pd.concat((self.generate_odd_object_sequences(zeros), self.generate_odd_relation_sequences(ones),
                          self.generate_odd_subject_sequences(twos)))

    def generate_odd_object_sequences(self, num_of_sequences):
        pairs = self.dataset[['subject', 'relation']].sample(num_of_sequences)
        objects = self.dataset['object'].sample(num_of_sequences)
        return pairs['subject'] + " " + pairs['relation'] + " " + objects

    def generate_odd_relation_sequences(self, num_of_sequences):
        pairs = self.dataset[['subject', 'object']].sample(num_of_sequences)
        relation = self.dataset['relation'].sample(num_of_sequences)
        return pairs['subject'] + " " + relation + " " + pairs['object']

    def generate_odd_subject_sequences(self, num_of_sequences):
        pairs = self.dataset[['object', 'relation']].sample(num_of_sequences)
        subjects = self.dataset['subject'].sample(num_of_sequences)
        return subjects + " " + pairs['relation'] + " " + pairs['object']


    def generate_semi_random_sentences(self, num_of_sequences):
        word_sequences = self.generate_semi_random_sequences(num_of_sequences)
        return word_sequences, self.graph2text.generate_sentences_from_word_sequences(word_sequences)



def main():
    tf.enable_eager_execution()
    dataset = load_normalized_dataset()
    input_sents, target_sents = data_utils.load_sequence_to_sentence_dataset()
    inp_lang = data_utils.LanguageIndex(input_sents)
    targ_lang = data_utils.LanguageIndex(target_sents)
    embedding_dim = 256
    units = 1024
    encoder = Encoder(embedding_dim, units, inp_lang)
    decoder = Decoder(embedding_dim, units, targ_lang)
    optimizer = tf.train.AdamOptimizer()
    graph2text = GraphToText(decoder, encoder, optimizer)
    generator = NegativeSamplesGenerator(graph2text, dataset)
    random_sequences, random_sentences = generator.generate_semi_random_sentences(2000)
    with open('checkpoint-4.txt', 'w', encoding='utf-8') as f:
        for i in range(len(random_sentences)):
            f.write(random_sequences[i] + "\n" + random_sentences[i] + '\n')


if __name__ == '__main__':
    main()
