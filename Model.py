import time
import logging
import tensorflow as tf
import numpy as np


def make_basic_model(inp_lang, targ_lang, checkpoints_dir, embedding_dim = 256, units = 1024):
    encoder = Encoder(embedding_dim, units, inp_lang)
    decoder = Decoder(embedding_dim, units, targ_lang)
    optimizer = tf.compat.v1.train.AdamOptimizer()
    return GraphToText(decoder, encoder, optimizer, checkpoints_dir)


def make_gru(units):
    # If you have a GPU, we recommend using CuDNNGRU(provides a 3x speedup than GRU)
    # the code automatically does that.
    if tf.test.is_gpu_available():
        return tf.keras.layers.CuDNNGRU(units, return_sequences=True, return_state=True,
                                        recurrent_initializer='glorot_uniform')
    else:
        return tf.keras.layers.GRU(units, return_sequences=True, return_state=True, recurrent_activation='sigmoid',
                                   recurrent_initializer='glorot_uniform')


class Encoder(tf.keras.Model):
    def __init__(self, embedding_dim, enc_units, lang):
        super(Encoder, self).__init__()
        vocab_size = len(lang.word2idx)
        self.enc_units = enc_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = make_gru(self.enc_units)
        self.lang = lang

    def call(self, x, hidden):
        x = self.embedding(x)
        if len(x.shape) == 2:
            x = tf.expand_dims(x, 0)
        output, state = self.gru(x, initial_state=hidden)
        return output, state

    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.enc_units))


class Decoder(tf.keras.Model):
    def __init__(self, embedding_dim, dec_units, lang):
        super(Decoder, self).__init__()
        vocab_size = len(lang.word2idx)
        self.dec_units = dec_units
        self.embedding = tf.keras.layers.Embedding(vocab_size, embedding_dim)
        self.gru = make_gru(self.dec_units)
        self.fc = tf.keras.layers.Dense(vocab_size)
        self.lang = lang
        # used for attention
        self.W1 = tf.keras.layers.Dense(self.dec_units)
        self.W2 = tf.keras.layers.Dense(self.dec_units)
        self.V = tf.keras.layers.Dense(1)

    def call(self, x, hidden, enc_output):
        # enc_output shape == (batch_size, max_length, hidden_size)

        # hidden shape == (batch_size, hidden size)
        # hidden_with_time_axis shape == (batch_size, 1, hidden size)
        # we are doing this to perform addition to calculate the score
        hidden_with_time_axis = tf.expand_dims(hidden, 1)

        # score shape == (batch_size, max_length, 1)
        # we get 1 at the last axis because we are applying tanh(FC(EO) + FC(H)) to self.V
        score = self.V(tf.nn.tanh(self.W1(enc_output) + self.W2(hidden_with_time_axis)))

        # attention_weights shape == (batch_size, max_length, 1)
        attention_weights = tf.nn.softmax(score, axis=1)

        # context_vector shape after sum == (batch_size, hidden_size)
        context_vector = attention_weights * enc_output
        context_vector = tf.reduce_sum(context_vector, axis=1)

        # x shape after passing through embedding == (batch_size, 1, embedding_dim)
        x = self.embedding(x)

        # x shape after concatenation == (batch_size, 1, embedding_dim + hidden_size)
        x = tf.concat([tf.expand_dims(context_vector, 1), x], axis=-1)

        # passing the concatenated vector to the GRU
        output, state = self.gru(x)

        # output shape == (batch_size * 1, hidden_size)
        output = tf.reshape(output, (-1, output.shape[2]))

        # output shape == (batch_size * 1, vocab)
        x = self.fc(output)

        return x, state, attention_weights

    def initialize_hidden_state(self, batch_sz):
        return tf.zeros((batch_sz, self.dec_units))


class GraphToText(tf.keras.Model):
    def __init__(self, decoder, encoder, optimizer, checkpoints_dir):
        super(GraphToText, self).__init__()
        self.decoder = decoder
        self.encoder = encoder
        self.optimizer = optimizer
        self.max_length_input = self.encoder.lang.max_sentence_length
        self.max_length_targ = self.decoder.lang.max_sentence_length
        self.checkpoint = tf.train.Checkpoint(optimizer=optimizer, encoder=encoder, decoder=decoder)
        self.manager = tf.train.CheckpointManager(self.checkpoint, checkpoints_dir, max_to_keep=3)
        checkpoint_file = self.manager.latest_checkpoint
        self.checkpoint.restore(checkpoint_file)
        if checkpoint_file:
            logging.info('Model restored from {}'.format(checkpoint_file))
        else:
            logging.info('Model initialized from scratch')

    def train(self, dataset, val_dataset, epochs, batch_size):
        logging.info('Will train for {} epochs'.format(epochs))
        for epoch in range(1, epochs + 1):
            logging.info('Starting epoch {}/{}'.format(epoch, epochs))
            start = time.time()

            hidden = self.encoder.initialize_hidden_state(batch_size)
            total_loss = 0
            num_batches = 0
            for (batch, (inp, targ)) in enumerate(dataset):
                loss = 0
                num_batches += 1
                with tf.GradientTape() as tape:
                    enc_output, enc_hidden = self.encoder(inp, hidden)
                    dec_hidden = enc_hidden
                    dec_input = tf.expand_dims([self.decoder.lang.word2idx['<start>']] * batch_size, 1)

                    # Teacher forcing - feeding the target as the next input
                    for t in range(1, targ.shape[1]):
                        # passing enc_output to the decoder
                        predictions, dec_hidden, _ = self.decoder(dec_input, dec_hidden, enc_output)
                        loss += self._loss_function(targ[:, t], predictions)
                        # using teacher forcing
                        dec_input = tf.expand_dims(targ[:, t], 1)

                batch_loss = (loss / int(targ.shape[1]))
                total_loss += batch_loss
                variables = self.encoder.variables + self.decoder.variables
                gradients = tape.gradient(loss, variables)
                self.optimizer.apply_gradients(zip(gradients, variables))

                if batch % 100 == 0:
                    logging.info('Epoch {} Batch {} Loss {:.4f}'.format(epoch, batch, batch_loss.numpy()))
            logging.info('Epoch {} Loss {:.4f}'.format(epoch, total_loss / num_batches))
            logging.info('Finished epoch {}/{} in {} seconds\n'.format(epoch, epochs, time.time() - start))

            logging.info('Saving checkpoint')
            self.manager.save()
            logging.info('Saved')
            if val_dataset is not None:
                logging.info('Evaluating validation loss')
                val_loss = self.evaluate_loss_on_dataset(val_dataset.prefetch(5))
                logging.info('Epoch {} Validation Loss {:.4f}'.format(epoch, val_loss))
        logging.info('Done training')

    def generate_sentences_from_word_sequences(self, word_sequences):
        inputs = [[self.encoder.lang.word2idx.get(word) for word in word_seq.split(" ")] for word_seq in word_sequences]
        return next(self._batch_iterator(inputs, return_all=True))

    def generate_sentences_from_word_indices(self, word_indices):
        targets = self._predict_batch(word_indices)
        results = np.vectorize(lambda idx: self.decoder.lang.idx2word[idx])(targets)
        results = list(
            map(lambda row: ' '.join(row[:list(row).index('<end>') if '<end>' in row else len(row)]), results.tolist()))
        return results

    def evaluate_loss(self, word_indices, targ_indices, batch_size=64):
        inputs = tf.keras.preprocessing.sequence.pad_sequences(word_indices, maxlen=self.max_length_input,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        targets = tf.keras.preprocessing.sequence.pad_sequences(targ_indices, maxlen=self.max_length_input,
                                                                padding='post')
        targets = tf.convert_to_tensor(targets)
        dataset = tf.data.Dataset.from_tensor_slices((inputs, targets))
        dataset = dataset.batch(batch_size)
        return self.evaluate_loss_on_dataset(dataset)

    def evaluate_loss_on_dataset(self, dataset):
        total_loss = 0
        num_batches = 0
        for inp, targ in dataset:
            total_loss += self._loss_on_batch(inp, targ)
            num_batches += 1
            if num_batches % 20 == 0:
                logging.info('Batch {} Loss {:.4f}'.format(num_batches, (total_loss / num_batches).numpy()))
        return total_loss / num_batches

    @staticmethod
    def _loss_function(real, pred):
        mask = 1 - np.equal(real, 0)
        loss_ = tf.nn.sparse_softmax_cross_entropy_with_logits(labels=real, logits=pred) * mask
        return tf.reduce_mean(loss_)

    def _batch_iterator(self, word_indices, batch_size=64, return_all=False):
        inputs = tf.keras.preprocessing.sequence.pad_sequences(word_indices, maxlen=self.max_length_input,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        dataset = tf.data.Dataset.from_tensor_slices(inputs)
        dataset = dataset.batch(batch_size)
        results = []
        for inp in dataset:
            res = self.generate_sentences_from_word_indices(inp)
            if not return_all:
                yield res
            results += res
        if return_all:
            yield results

    def _predict_batch(self, word_indices, real=None):
        batch_size = len(word_indices)
        inputs = tf.keras.preprocessing.sequence.pad_sequences(word_indices, maxlen=self.max_length_input,
                                                               padding='post')
        inputs = tf.convert_to_tensor(inputs)
        targets = np.zeros(shape=(batch_size, self.max_length_targ-1), dtype=np.int32)
        hidden = tf.zeros((batch_size, self.encoder.enc_units))
        enc_out, enc_hidden = self.encoder(inputs, hidden)
        dec_hidden = enc_hidden
        dec_input = tf.expand_dims([self.decoder.lang.word2idx['<start>']] * batch_size, 1)
        loss = 0
        for t in range(1, self.max_length_targ):
            predictions, dec_hidden, attention_weights = self.decoder(dec_input, dec_hidden, enc_out)
            if real is not None:
                loss += self._loss_function(real[:, t], predictions)
            predicted_ids = tf.argmax(predictions, axis=1).numpy()

            targets[:, t-1] = predicted_ids
            # the predicted ID is fed back into the model
            dec_input = np.expand_dims(predicted_ids, 1)
        if real is not None:
            batch_loss = (loss / int(real.shape[1]))
            return targets, batch_loss
        return targets

    def _loss_on_batch(self, word_indices, real):
        predicts, loss = self._predict_batch(word_indices, real)
        return loss
