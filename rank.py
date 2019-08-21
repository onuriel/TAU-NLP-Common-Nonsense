import data_constants
import data_loader
import logging
import argparse
import numpy as np

EMBEDDING_NOT_FOUND = -100


class SentencesRank:
    def __init__(self, use_mini_version=False):
        if use_mini_version:
            numberbatch_path = data_constants.DEFAULT_NUMBERBATCH_MINI_PATH
            logging.info('Loading mini numberbatch version from {}'.format(numberbatch_path))
            self.prefix = '/c/en/'
        else:
            numberbatch_path = data_constants.DEFAULT_NUMBERBATCH_PATH
            logging.info('Loading full numberbatch version from {}'.format(numberbatch_path))
            self.prefix = ''
        self.numberbatch_df = data_loader.load_numberbatch(numberbatch_path)
        logging.info('Numberbatch loaded')

    @staticmethod
    def load_sentences(input_path):
        edges = []
        sentences = []
        logging.info('Loading sentences from {}'.format(input_path))
        with open(input_path, 'r') as input_file:
            for edge, sentence in zip(input_file, input_file):
                edge = edge.strip()
                sentence = sentence.strip()
                edges.append(edge)
                sentences.append(sentence)
                logging.debug('{} -> {}'.format(edge, sentence))
        logging.info('Sentences loaded')
        return edges, sentences

    @staticmethod
    def edges_to_tuples(edges):
        logging.info('Turning edges (sequences) to tuples')
        edges_tuples = []
        for edge in edges:
            words = edge.split(' ')
            relation_idx = next(idx for idx, word in enumerate(words) if word[0].isupper())
            edges_tuples.append((words[:relation_idx], [words[relation_idx]], words[relation_idx+1:]))
        return edges_tuples
    
    def get_cosine_similarity(self, edges_tuples):
        logging.info('Calculating cosine similarity for each edge tuple')
        res = []
        for edge_tuple in edges_tuples:
            subject_embedding = self._check_for_embedding(edge_tuple[0])
            object_embedding = self._check_for_embedding(edge_tuple[2])
            if subject_embedding == EMBEDDING_NOT_FOUND or object_embedding == EMBEDDING_NOT_FOUND:
                res.append(EMBEDDING_NOT_FOUND)
                continue
            subject_embedding = np.mean(subject_embedding, axis=0)
            object_embedding = np.mean(object_embedding, axis=0)
            cos_sim = np.dot(subject_embedding, object_embedding.T) / (
                        np.linalg.norm(subject_embedding) * np.linalg.norm(object_embedding))

            res.append(cos_sim)
        return res

    def _check_for_embedding(self, words):
        embedding = []
        for word in words:
            prefixed_word = self.prefix + word
            try:
                embedding.append(self.numberbatch_df.loc[prefixed_word])
            except KeyError:
                return EMBEDDING_NOT_FOUND
        return embedding


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Ranking Generated Sentences')
    parser.add_argument('--version', action='version',
                        version='%(prog)s {version}'.format(version=data_constants.VERSION))
    parser.add_argument('-m', '--mini', action='store_true', help='Use mini numberbatch version')
    parser.add_argument('-i', '--input_sentences', default=data_constants.DEFAULT_GENERATED_SENTENCES_PATH,
                        help='Path to generated input sentences')
    parser.add_argument('-o', '--output_sentences', default=data_constants.DEFAULT_RANKED_SENTENCES_PATH,
                        help='Path to ranked output sentences')
    parser.add_argument('-v', '--verbose', action='store_true', help='Verbose logging')
    args = parser.parse_args()
    logging.basicConfig(format='[%(asctime)s] [%(name)s] [%(levelname)s] %(message)s',
                        level=logging.DEBUG if args.verbose else logging.INFO)
    logging.info('Starting with args: {}'.format(vars(args)))

    rank = SentencesRank(args.mini)
    edges, sentences = rank.load_sentences(args.input_sentences)
    edges_tuples = rank.edges_to_tuples(edges)
    edges_scores = rank.get_cosine_similarity(edges_tuples)
    logging.info('Writing outputs to file {}'.format(args.output_sentences))
    with open(args.output_sentences, 'w') as f:
        for edge_score, edge, sentence in sorted(zip(edges_scores, edges, sentences)):
            f.write('{}_{}_{}\n'.format(edge_score, edge, sentence))
    logging.info('Done')
