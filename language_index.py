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
