import os


class Config(object):
    debug = -1
    train_pickle = 'Data/movie/train.pickle'
    test_pickle = 'Data/movie/test.pickle'
    train_txt = 'Data/movie/train.txt'
    word2vec_model = "Data/movie/word2vec.model"
    vec_file = 'Data/movie/vocab.txt'
    word_vec_path = 'Data/movie/word_vec.pickle'
    word_voc_path = 'Data/movie/word_voc.pickle'
    word_embedding_size = 512
    hidden_size = 512
    vocab_size = -1
    writer_path = 'Data/movie/TFRecordWriter'  # epoch_size:15467  total step:  15466
    batch_size = 128
    num_steps = 140

    movie = 28
    score = 5
    num_keywords = 2

    max_epoch = 50
    keep_prob = 0.9  # The probability that each element is kept through dropout layer
    lr_decay = 1.0

    model_path = './model_1102/'  # the path of model that need to save or load
    eval_eopch = 1
    res_path = '{}epoch_{}'.format(model_path, eval_eopch)
    if not os.path.exists(res_path):
        os.makedirs(res_path)
    generate_file = '{}/generated.txt'.format(res_path)
    eval_file = '{}/eval.txt'.format(res_path)

    log = './model_1102/log.txt'
    init_scale = 0.02
    learning_rate = 0.001
    max_grad_norm = 5
    num_layers = 2
    print_steps = 10
    # parameter for generation
    len_of_generation = 140  # The number of characters by generated

    is_sample = True  # true means using sample, if not using argmax
    BeamSize = 2
