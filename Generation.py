import time
import numpy as np
import pickle
import tensorflow.nn.rnn_cell as rnn_cell
import tensorflow as tf
from Config import Config

try:
    from tensorflow.contrib.legacy_seq2seq.python.ops.seq2seq import sequence_loss_by_example
except:
    pass


def save_pickle(content, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj=content, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        content = pickle.load(handle)
    return content


config_tf = tf.ConfigProto()
config_tf.gpu_options.allow_growth = True

# total_step = 7470  # todo
# get value from output of Preprocess.py file

config = Config()

# log = open(config.log, "w")
word_vec = load_pickle(config.word_vec_path)
vocab = load_pickle(config.word_voc_path)
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
config.vocab_size = len(vocab)


class Model(object):
    def __init__(self, is_training, config):
        self.batch_size = batch_size = config.batch_size
        self.num_steps = num_steps = config.num_steps
        self.size = size = config.hidden_size
        vocab_size = config.vocab_size

        self._input_data = tf.placeholder(tf.int32, [batch_size, num_steps])
        self._mask = tf.placeholder(tf.float32, [batch_size, None])  # 注意类型是float
        self._input_word = tf.placeholder(tf.int32, [batch_size, config.num_keywords])
        self._init_output = tf.placeholder(tf.float32, [batch_size, size])

        def single_cell_fn(unit_type, num_units, dropout, mode, forget_bias=1.0):
            """Create an instance of a single RNN cell."""
            dropout = dropout if mode is True else 0.0
            if unit_type == "lstm":
                c = rnn_cell.LSTMCell(num_units, forget_bias=forget_bias, state_is_tuple=False)
            elif unit_type == "gru":
                c = rnn_cell.GRUCell(num_units)
            else:
                raise ValueError("Unknown unit type %s!" % unit_type)
            if dropout > 0.0:
                c = rnn_cell.DropoutWrapper(cell=c, input_keep_prob=(1.0 - dropout))
            return c

        cell_list = []
        for i in range(config.num_layers):
            single_cell = single_cell_fn(unit_type="lstm", num_units=size, dropout=1 - config.keep_prob,
                                         mode=is_training)
            cell_list.append(single_cell)

        cell = rnn_cell.MultiRNNCell(cell_list, state_is_tuple=False)
        self._initial_state = cell.zero_state(batch_size, tf.float32)

        with tf.device("/cpu:0"):
            embedding_keyword = tf.get_variable('keyword_embedding',
                                                [config.movie + config.score, config.word_embedding_size],
                                                trainable=True,
                                                initializer=tf.random_uniform_initializer(-config.init_scale,
                                                                                          config.init_scale))
            embedding = tf.get_variable('word_embedding', [vocab_size, config.word_embedding_size], trainable=True,
                                        initializer=tf.constant_initializer(word_vec))

            inputs = tf.nn.embedding_lookup(embedding, self._input_data)
            keyword_inputs = tf.nn.embedding_lookup(embedding_keyword, self._input_word)

        if is_training and config.keep_prob < 1:
            inputs = tf.nn.dropout(inputs, config.keep_prob)
            # keyword_inputs = tf.nn.dropout(keyword_inputs, config.keep_prob)
        self.initial_gate = tf.ones([batch_size, config.num_keywords])
        atten_sum = tf.zeros([batch_size, config.num_keywords])

        with tf.variable_scope("coverage"):
            """
            u_f 是一个变量参数，他负责与topic相乘，得到的结果再通过sigmoid归一化到0～1之间，目的是为每一个控制信息分配一个初始比例
            sen_len 是想计算每个样本的有效字数
            假设每个样本，如果有两个控制条件的话，每一个控制条件的重要程度用一个0～1之间的数表示，（其实这里应该是 softmax更加合理）
            有多少有效字，那么这句话中该控制条件就有多少的初始总分值
            """
            u_f = tf.get_variable("u_f", [config.num_keywords * config.word_embedding_size, config.num_keywords])
            res1 = tf.sigmoid(tf.matmul(tf.reshape(keyword_inputs, [batch_size, -1]), u_f))  # todo
            sen_len = tf.reduce_sum(self._mask, -1, keepdims=True)
            phi_res = sen_len * res1
            self.output1 = phi_res

        outputs = []
        output_state = self._init_output
        state = self._initial_state
        with tf.variable_scope("RNN"):
            for time_step in range(num_steps):
                # vs 里面放的是当前这个time step，上一个时刻的隐含层状态跟每一个主题的关系一个被gate消弱后的得分
                vs = []
                for kw_i in range(config.num_keywords):
                    with tf.variable_scope("RNN_attention"):
                        if time_step > 0 or kw_i > 0:
                            tf.get_variable_scope().reuse_variables()
                        u = tf.get_variable("u", [size, 1])
                        w1 = tf.get_variable("w1", [size, size])
                        w2 = tf.get_variable("w2", [config.word_embedding_size, size])
                        b = tf.get_variable("b1", [size])

                        # 加工上一次隐含层状态 线性变换一下
                        temp2 = tf.matmul(output_state, w1)
                        # 取到某一个主题的向量
                        temp3 = keyword_inputs[:, kw_i, :]
                        # 对主题的向量,线性变换一下
                        temp4 = tf.matmul(temp3, w2)
                        # 线性变换后的隐状态和主题add起来
                        temp5 = tf.add(temp2, temp4)
                        # 加上一个偏置项
                        temp6 = tf.add(temp5, b)
                        # 加上一个非线性
                        temp7 = tf.tanh(temp6)
                        # 在线性变换一下
                        vi = tf.matmul(temp7, u)
                        temp8 = self.initial_gate[:, kw_i:kw_i + 1]  # 把kw_i主题对应的gate控制变量取出来，这个gate初始值都是1
                        temp9 = vi * temp8
                        vs.append(temp9)

                self.attention_vs = tf.concat(vs, axis=1)
                prob_p = tf.nn.softmax(self.attention_vs)
                # 此处prob_p表示的是上一步的隐含层状态对每一个主题的注意力得分
                self.initial_gate = self.initial_gate - (prob_p / phi_res)
                temp10 = self._mask[:, time_step:time_step + 1]
                atten_sum += prob_p * temp10
                # （batchsize，2） * （batchsize，1）
                # 如果某一个样本的这个time step的mask是0，那么对应这个样本的所有的主题的权重都为0
                # 全部被mask掉了
                # 全部主题的词向量的加权和
                mt = tf.add_n([prob_p[:, i:i + 1] * keyword_inputs[:, i, :] for i in range(config.num_keywords)])

                with tf.variable_scope("RNN_sentence"):
                    if time_step > 0:
                        tf.get_variable_scope().reuse_variables()
                    temp11 = inputs[:, time_step, :]
                    # mt 是根据 time_step上一个时刻的 隐含层状态 和 主题 信息一起得到的
                    temp12 = tf.concat([temp11, mt], axis=1)
                    # 必须要保证 cell input 的 dims = hidden units
                    temp13 = tf.layers.dense(inputs=temp12, units=size)
                    (cell_output, state) = cell(temp13, state)
                    outputs.append(cell_output)
                    output_state = cell_output

            self._last_output = cell_output

        output = tf.reshape(tf.concat(outputs, axis=1), [-1, size])
        softmax_w = tf.get_variable("softmax_w", [size, vocab_size])
        softmax_b = tf.get_variable("softmax_b", [vocab_size])
        logits = tf.matmul(output, softmax_w) + softmax_b

        # loss = sequence_loss_by_example([logits], [tf.reshape(self._targets, [-1])], [tf.reshape(self._mask, [-1])])
        # # 得到的是一个batch里面 所有字的 loss  shape ： batch_size*seq_len
        # self.cost1 = tf.reduce_sum(loss)
        # self.cost2 = tf.reduce_sum((phi_res - atten_sum) ** 2)
        # mask_sum = tf.reduce_sum(self._mask)
        # self._cost = cost = (self.cost1 + 0.1 * self.cost2) / mask_sum
        # # self._cost = cost = (self.cost1 + 0.1 * self.cost2)

        self._final_state = state
        self._prob = tf.nn.softmax(logits)

        # if not is_training:
        #     prob = tf.nn.softmax(logits)
        #     self._sample = tf.argmax(prob, 1)
        #     return

        # self._lr = tf.Variable(0.0, trainable=False)
        # tvars = tf.trainable_variables()
        # grads, _ = tf.clip_by_global_norm(tf.gradients(cost, tvars), config.max_grad_norm)
        # optimizer = tf.train.AdamOptimizer(self.lr)
        # self._train_op = optimizer.apply_gradients(zip(grads, tvars))

    def assign_lr(self, session, lr_value):
        session.run(tf.assign(self.lr, lr_value))

    @property
    def input_data(self):
        return self._input_data

    @property
    def input_word(self):
        return self._input_word

    @property
    def init_output(self):
        return self._init_output

    @property
    def last_output(self):
        return self._last_output

    @property
    def initial_state(self):
        return self._initial_state

    @property
    def prob(self):
        return self._prob

    @property
    def final_state(self):
        return self._final_state

    @property
    def mask(self):
        return self._mask

    @property
    def lr(self):
        return self._lr


def run_epoch(session, m, x, state=None, input_words=None, last_output=None, last_gate=None, lens=None, flag=0):
    initial_output = np.zeros((m.batch_size, m.size))
    if flag is 0:
        prob, _state, _last_output = session.run([m.prob, m.final_state, m.last_output],
                                                 {m.input_data: x, m.input_word: input_words, m.initial_state: state,
                                                  m.mask: np.float32([[1]]), m.init_output: initial_output})

        return prob, _state, _last_output
    else:
        prob, _state, _last_output = session.run([m.prob, m.final_state, m.last_output],
                                                 {m.input_data: x, m.input_word: input_words, m.initial_state: state,
                                                  m.mask: np.float32([[1]]), m.init_output: last_output})
    return prob, _state, _last_output


def main(_):
    with tf.Graph().as_default(), tf.Session(config=config_tf) as session:
        config.batch_size = 1
        config.num_steps = 1
        beam_size = config.BeamSize

        initializer = tf.random_uniform_initializer(-config.init_scale, config.init_scale)
        with tf.variable_scope("model", reuse=None, initializer=initializer):
            mtest = Model(is_training=False, config=config)

        tf.global_variables_initializer().run()

        model_saver = tf.train.Saver(tf.all_variables())
        print('model loading ...')
        model_saver.restore(session, config.model_path + 'epoch_%d' % config.eval_eopch)
        print('Done!')

        test_topic = [2, 27 + 3]

        len_of_sample = config.len_of_generation
        _state = mtest.initial_state.eval()
        beams = [(0.0, [idx_to_word[1]], idx_to_word[1])]
        # 第一个数值 表示此beam的得分，第二个list表示句子的词，最后表示目前句子的最后一个词
        _input_words = np.array([test_topic], dtype=np.float32)
        test_data = np.int32([[1]])
        prob, _state, _last_output = run_epoch(session, mtest, test_data, _state, input_words=_input_words,
                                               lens=len_of_sample, flag=0)
        y1 = np.log(1e-20 + prob.reshape(-1))
        if config.is_sample:
            try:
                # 是从a中以概率p，随机选择size个, p没有指定的时候相当于是一致的分布,
                # replace是False的话，那么采样出来的结果是不放回不同的
                top_indices = np.random.choice(a=config.vocab_size, size=beam_size, replace=False, p=prob.reshape(-1))
            except:
                top_indices = np.random.choice(a=config.vocab_size, size=beam_size, replace=True, p=prob.reshape(-1))
        else:
            top_indices = np.argsort(-y1)
        b = beams[0]
        beam_candidates = []
        for i in range(beam_size):
            wordix = top_indices[i]
            beam_candidates.append((b[0] + y1[wordix], b[1] + [idx_to_word[wordix]], wordix, _state, _last_output))
        beam_candidates.sort(key=lambda x: x[0], reverse=True)  # 按照句子练成的概率排序
        beams = beam_candidates[:beam_size]  # 剪枝
        for xy in range(len_of_sample - 1):
            beam_candidates = []
            for b in beams:
                test_data = np.int32(b[2])
                test_data = np.reshape(test_data, (1, 1))
                prob, _state, _last_output = run_epoch(session=session, m=mtest, x=test_data, state=_state,
                                                       input_words=_input_words, last_output=_last_output,
                                                       lens=len_of_sample, flag=1)
                y1 = np.log(1e-20 + prob.reshape(-1))
                if config.is_sample:
                    try:
                        top_indices = np.random.choice(config.vocab_size, beam_size, replace=False, p=prob.reshape(-1))
                    except:
                        top_indices = np.random.choice(config.vocab_size, beam_size, replace=True, p=prob.reshape(-1))
                else:
                    top_indices = np.argsort(-y1)
                for i in range(beam_size):
                    wordix = top_indices[i]
                    beam_candidates.append(
                        (b[0] + y1[wordix], b[1] + [idx_to_word[wordix]], wordix, _state, _last_output))
            beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
            beams = beam_candidates[:beam_size]  # truncate to get new beams
        res_text = beams[0][1][1:]
        final_text = []
        remove_set = {'[PAD]', '[MASK]', '[CLS]', '[UNK]'}
        for w in res_text:
            if w in remove_set:
                continue
            final_text.append(w)
        print(' '.join(final_text))


if __name__ == "__main__":
    tf.app.run()
