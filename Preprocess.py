import numpy as np
import tensorflow as tf
import pickle
import os
from tqdm import tqdm
from Config import Config
from pre_train_word2vec import load_pickle, save_pickle
config = Config()


def Read_WordVec():
    if not os.path.exists(config.word_vec_path):
        with open(config.vec_file, 'r', encoding="utf-8") as fvec:
            word_voc = []
            vec_ls = []

            word_voc.append('[PAD]')
            vec_ls.append([0]*config.word_embedding_size)
            word_voc.append('[UNK]')
            vec_ls.append([0]*config.word_embedding_size)
            word_voc.append('[CLS]')
            vec_ls.append([0] * config.word_embedding_size)
            word_voc.append('[SEP]')
            vec_ls.append([0]*config.word_embedding_size)
            word_voc.append('[MASK]')
            vec_ls.append([0]*config.word_embedding_size)

            for line in fvec:
                line = line.split()
                try:
                    word = line[0]
                    vec = [float(i) for i in line[1:]]
                    assert len(vec) == config.word_embedding_size
                    word_voc.append(word)
                    vec_ls.append(vec)
                except:
                    print(line[0])

            config.vocab_size = len(word_voc)
            word_vec = np.array(vec_ls, dtype=np.float32)

            save_pickle(word_vec, config.word_vec_path)
            save_pickle(word_voc, config.word_voc_path)
    else:

        word_vec = load_pickle(config.word_vec_path)
        word_voc = load_pickle(config.word_voc_path)

    return word_voc, word_vec


print('loading the training data...')
vocab, _ = Read_WordVec()
train_data = load_pickle(config.train_pickle)
texts, a1s, a2s = list(train_data["text"]), list(train_data["attribute1"]), list(train_data["attribute2"])
train_data = list(zip(texts, a1s, a2s))

word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
data_size, _vocab_size = len(train_data), len(vocab)
print('data has %d document, size of word vocabulary: %d.' % (data_size, _vocab_size))


def data_iterator(train_data, batch_size, num_steps):
    chunk = len(train_data) // batch_size
    print("epoch_chunk:{}".format(chunk))
    for i in tqdm(range(chunk)):
        batch_data = train_data[i * batch_size: (i+1) * batch_size]

        data_x = np.zeros((batch_size, num_steps), dtype=np.int64)
        data_y = np.zeros((batch_size, num_steps), dtype=np.int64)
        key_words = []

        ids = 0
        for text, a1, a2 in batch_data:

            key_words.append([a1, config.movie-1+a2])  # !!!!!

            doc = [word_to_idx.get(wd, 1) for wd in text]
            doc = doc[:num_steps-1]

            doc_x = [4] + doc  # 开始符号
            doc_x = np.array(doc_x, dtype=np.int64)
            data_x[ids][:len(doc_x)] = doc_x

            doc_y = doc + [2]   # 结束符号
            doc_y = np.array(doc_y, dtype=np.int64)
            data_y[ids][:len(doc_y)] = doc_y

            ids += 1

        key_words = np.array(key_words, dtype=np.int64)
        mask = np.float32(data_x != 0)
        yield (data_x, data_y, mask, key_words)


writer = tf.python_io.TFRecordWriter(config.writer_path)
dataLS = []
iterator = data_iterator(train_data, config.batch_size, config.num_steps)

step = 0
for x, y, mask, key_words in tqdm(iterator):
    example = tf.train.Example(
        # Example contains a Features proto object
        features=tf.train.Features(
          # Features contains a map of string to Feature proto objects
          feature={
            # A Feature contains one of either a int64_list,
            # float_list, or bytes_list
            'input_data': tf.train.Feature(
                int64_list=tf.train.Int64List(value=x.reshape(-1).astype("int64"))),
            'target': tf.train.Feature(
                int64_list=tf.train.Int64List(value=y.reshape(-1).astype("int64"))),
            'mask': tf.train.Feature(
                float_list=tf.train.FloatList(value=mask.reshape(-1).astype("float"))),
            'key_words': tf.train.Feature(
                int64_list=tf.train.Int64List(value=key_words.reshape(-1).astype("int64")))
          }))
    # use the proto object to serialize the example to a string
    serialized = example.SerializeToString()
    # write the serialized object to disk
    writer.write(serialized)
    step += 1
    
print('total step: ', step)
