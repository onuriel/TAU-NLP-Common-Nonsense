import tensorflow as tf
from tensorflow.contrib.learn.python.learn.estimators._sklearn import train_test_split

import data_utils
import data_loader
import Model

BATCH_SIZE = 64


def create_train_val_dataset(input_sents, target_sents, inp_lang, targ_lang):
    input_tensor = [[inp_lang.word2idx[s] for s in sent.split(' ')] for sent in input_sents]
    target_tensor = [[targ_lang.word2idx[s] for s in sent.split(' ')] for sent in target_sents]
    input_tensor = tf.keras.preprocessing.sequence.pad_sequences(input_tensor,
                                                                 maxlen=inp_lang.max_sentence_length,
                                                                 padding='post')

    target_tensor = tf.keras.preprocessing.sequence.pad_sequences(target_tensor,
                                                                  maxlen=targ_lang.max_sentence_length,
                                                                  padding='post')
    input_tensor_train, input_tensor_val, target_tensor_train, target_tensor_val = train_test_split(input_tensor,
                                                                                                    target_tensor,
                                                                                                    test_size=0.2,
                                                                                                    random_state=42)
    train_set_size = len(input_tensor_train)
    val_set_size = len(input_tensor_val)
    train_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_train, target_tensor_train)).shuffle(
        train_set_size)
    train_dataset = train_dataset.batch(BATCH_SIZE, drop_remainder=True)
    val_dataset = tf.data.Dataset.from_tensor_slices((input_tensor_val, target_tensor_val)).shuffle(val_set_size)
    val_dataset = val_dataset.batch(BATCH_SIZE, drop_remainder=True)
    return train_dataset, val_dataset


def main():
    tf.compat.v1.enable_eager_execution()
    # os.environ["CUDA_DEVICE_ORDER"] = "PCI_BUS_ID"  # see issue #152
    # os.environ["CUDA_VISIBLE_DEVICES"] = "-1"
    # Creating training and validation sets using an 80-20 split
    input_sents, target_sents = data_loader.load_sequence_to_sentence_dataset()
    inp_lang = data_utils.LanguageIndex(input_sents)
    targ_lang = data_utils.LanguageIndex(target_sents)

    embedding_dim = 256
    units = 1024
    train_dataset, val_dataset = create_train_val_dataset(input_sents, target_sents, inp_lang, targ_lang)
    encoder = Model.Encoder(embedding_dim, units, inp_lang)
    decoder = Model.Decoder(embedding_dim, units, targ_lang)

    optimizer = tf.compat.v1.train.AdamOptimizer()
    model = Model.GraphToText(decoder, encoder, optimizer)
    model.train(train_dataset, None, epochs=10, batch_size=BATCH_SIZE)
    #for i in range(6):
    #    model.load_from_checkpoint('ckpt-'+str(i))
    #    result = model.evaluate_loss_on_dataset(val_dataset.take(400))
    #    print(result)


if __name__ == '__main__':
    main()
