import pickle
from random import shuffle
import math
import os
import pickle
import time
from collections import namedtuple
from random import shuffle
from nltk.translate.bleu_score import sentence_bleu
# plt.switch_backend('agg')

import matplotlib.pyplot as plt
import matplotlib.ticker as ticker
import matplotlib.font_manager as fm
import matplotlib.font_manager as font_manager
import numpy as np
import torch
import torch.nn.functional as F
from torch import nn, autograd
from torch.optim.lr_scheduler import ReduceLROnPlateau
from tqdm import tqdm
from radam import RAdam


def save_pickle(content, path):
    with open(path, 'wb') as handle:
        pickle.dump(obj=content, file=handle, protocol=pickle.HIGHEST_PROTOCOL)


def load_pickle(path):
    with open(path, 'rb') as handle:
        content = pickle.load(handle)
    return content


embedding_dim = 512
hidden_dim = 512
lr = 1e-3 * 0.5
momentum = 0.01
num_epoch = 50
clip_value = 11
use_gpu = True  # todo
num_layers = 2
bidirectional = False
batch_size = 512
num_keywords = 2
verbose = 1
check_point = 2
beam_size = 5
is_sample = True
dropout = 0.1  # 默认居然是0.5 todo
init_uniform = 0.02
version_num = 0
train_path = "./Data/movie/train.pickle"
test_path = "./Data/movie/test.pickle"

vocab = []
with open("./Data/movie/vocab.txt") as f:
    for v in f:
        v = v.strip()
        vocab.append(v)
word_to_idx = {ch: i for i, ch in enumerate(vocab)}
idx_to_word = {i: ch for i, ch in enumerate(vocab)}
vocab_size = len(vocab)
loss_function = nn.NLLLoss()
adaptive_softmax = nn.AdaptiveLogSoftmaxWithLoss(
    1000, len(vocab), cutoffs=[round(vocab_size / 20), 4 * round(vocab_size / 20)])

font_dirs = ['/fonts/', ]
font_files = font_manager.findSystemFonts(fontpaths=font_dirs)
font_list = font_manager.createFontList(font_files)
font_manager.fontManager.ttflist.extend(font_list)
path = './fonts/NotoSansCJKtc-Regular.otf'
fontprop = fm.FontProperties(fname=path)

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3"
device = 'cuda' if torch.cuda.is_available() else 'cpu'
print('using device:', device)
# print('Available cuda:', torch.cuda.device_count())
# if torch.cuda.is_available():
#     device_num = 1
#     deviceName = "cuda:%d" % device_num
#     torch.cuda.set_device(device_num)
#     print('Current device:', torch.cuda.current_device())
# else:
#     deviceName = "cpu"
# device = torch.device(deviceName)

essays = []
topics = []
train = load_pickle(train_path)  # todo
# num_lines = sum(1 for line in open(file_path+'composition_zh_tw.txt', 'r'))
# with open(file_path+'composition_zh_tw.txt') as f:
#     for line in tqdm(f, total=num_lines):
#         essay, topic = line.replace('\n', '').split(' </d> ')
#         essays.append(essay.split(' '))
#         topics.append(topic.split(' '))
#     f.close()

text = list(train["text"])
attribute1, attribute2 = list(train["attribute1"]), list(train["attribute2"])
for i in range(len(text)):
    essays.append(list(text[i]))
    topics.append([attribute1[i], 27 + attribute2[i]])

assert len(topics) == len(essays)

corpus_indice = list(map(lambda x: [word_to_idx[w] if (w in word_to_idx) else word_to_idx['[UNK]'] for w in x], essays))
topics_indice = topics

essays = []
topics = []
train = load_pickle(test_path)

text = list(train["text"])
attribute1, attribute2 = list(train["attribute1"]), list(train["attribute2"])
for i in range(len(text)):
    essays.append(list(text[i]))
    topics.append([attribute1[i], 27 + attribute2[i]])

assert len(topics) == len(essays)

corpus_test = list(map(lambda x: [word_to_idx[w] if (w in word_to_idx) else word_to_idx['[UNK]'] for w in x], essays))
topics_test = topics


def viewData(topics, X):
    #     topics = [idx_to_word[x] for x in topics]
    X = [idx_to_word[x] for x in X]
    print(topics, X)


def shuffleData(topics_indice, corpus_indice):
    ind_list = [i for i in range(len(topics_indice))]
    shuffle(ind_list)
    topics_indice = np.array(topics_indice)
    corpus_indice = np.array(corpus_indice)
    topics_indice = topics_indice[ind_list,]
    corpus_indice = corpus_indice[ind_list,]
    topics_indice = topics_indice.tolist()
    corpus_indice = corpus_indice.tolist()
    return topics_indice, corpus_indice


def params_init_uniform(m):
    if type(m) == nn.Linear:
        y = init_uniform
        nn.init.uniform_(m.weight, -y, y)

# topics_indice, corpus_indice = shuffleData(topics_indice, corpus_indice)
# viewData(topics_indice[0], corpus_indice[0])

length = list(map(lambda x: len(x), corpus_indice))


def data_iterator(corpus_indice, topics_indice, batch_size, num_steps):
    epoch_size = len(corpus_indice) // batch_size
    for i in range(epoch_size):
        raw_data = corpus_indice[i * batch_size: (i + 1) * batch_size]
        key_words = topics_indice[i * batch_size: (i + 1) * batch_size]
        data = np.zeros((len(raw_data), num_steps), dtype=np.int64)
        for i in range(batch_size):
            doc = raw_data[i][0:num_steps-2]
            tmp = [4]  # 开始
            tmp.extend(doc)
            tmp.extend([2])  # 结束
            tmp = np.array(tmp, dtype=np.int64)
            _size = tmp.shape[0]
            data[i][:_size] = tmp
        key_words = np.array(key_words, dtype=np.int64)
        x = data[:, 0:num_steps-1]
        y = data[:, 1:]
        mask = np.float32(y != 0)  # todo mask 的选取
        x = torch.tensor(x)
        y = torch.tensor(y)
        mask = torch.tensor(mask)
        key_words = torch.tensor(key_words)
        yield (x, y, mask, key_words)


class Attention(nn.Module):
    """Implements Bahdanau (MLP) attention"""

    def __init__(self, hidden_size, embed_size):
        super(Attention, self).__init__()

        self.Ua = nn.Linear(embed_size, hidden_size, bias=False)
        self.Wa = nn.Linear(hidden_size, hidden_size, bias=False)
        self.va = nn.Linear(hidden_size, 1, bias=True)
        # to store attention scores
        self.alphas = None

    def forward(self, query, topics, coverage_vector):
        scores = []
        C_t = coverage_vector.clone()
        for i in range(topics.shape[1]):
            proj_key = self.Ua(topics[:, i, :])
            query = self.Wa(query)
            scores += [self.va(torch.tanh(query + proj_key)) * C_t[:, i:i + 1]]

        # stack scores
        scores = torch.stack(scores, dim=1)
        scores = scores.squeeze(2)
        #         print(scores.shape)
        # turn scores to probabilities
        alphas = F.softmax(scores, dim=1)
        self.alphas = alphas

        # mt vector is the weighted sum of the topics
        mt = torch.bmm(alphas.unsqueeze(1), topics)
        mt = mt.squeeze(1)

        # mt shape: [batch x embed], alphas shape: [batch x num_keywords]
        return mt, alphas


class AttentionDecoder(nn.Module):
    def __init__(self, hidden_size, embed_size, num_layers, dropout=0.5):
        super(AttentionDecoder, self).__init__()

        self.hidden_size = hidden_size
        self.embed_size = embed_size
        self.num_layers = num_layers
        self.dropout = dropout

        # topic attention
        self.attention = Attention(hidden_size, embed_size)

        # lstm
        self.rnn = nn.LSTM(input_size=embed_size * 2,
                           hidden_size=hidden_size,
                           num_layers=num_layers,
                           dropout=dropout)

    def forward(self, input, output, hidden, phi, topics, coverage_vector):
        # 1. calculate attention weight and mt
        mt, score = self.attention(output.squeeze(0), topics, coverage_vector)
        mt = mt.unsqueeze(1).permute(1, 0, 2)

        # 2. update coverge vector [batch x num_keywords]
        coverage_vector = coverage_vector - score / phi

        # 3. concat input and Tt, and feed into rnn
        output, hidden = self.rnn(torch.cat([input, mt], dim=2), hidden)

        return output, hidden, score, coverage_vector


LSTMState = namedtuple('LSTMState', ['hx', 'cx'])


class MTALSTM(nn.Module):
    def __init__(self, hidden_dim, embed_dim, num_keywords, num_layers, weight,
                 num_labels, bidirectional, dropout=0.5, **kwargs):
        super(MTALSTM, self).__init__(**kwargs)
        self.hidden_dim = hidden_dim
        self.embed_dim = embed_dim
        self.num_layers = num_layers
        self.num_labels = num_labels
        self.bidirectional = bidirectional

        if num_layers <= 1:
            self.dropout = 0
        else:
            self.dropout = dropout

        self.embedding = nn.Embedding(vocab_size, embed_dim)
        self.embedding_topic = nn.Embedding(28+5, embed_dim)  # todo
        # self.embedding = nn.Embedding.from_pretrained(weight)
        # self.embedding.weight.requires_grad = False

        self.Uf = nn.Linear(embed_dim * num_keywords, num_keywords, bias=False)

        # attention decoder
        self.decoder = AttentionDecoder(hidden_size=hidden_dim,
                                        embed_size=embed_dim,
                                        num_layers=num_layers,
                                        dropout=dropout)

        # adaptive softmax
        self.adaptiveSoftmax = nn.AdaptiveLogSoftmaxWithLoss(hidden_dim,
                                                             num_labels,
                                                             cutoffs=[round(num_labels / 20),
                                                                      4 * round(num_labels / 20)])

    def forward(self, inputs, topics, output, hidden=None, mask=None, target=None, coverage_vector=None,
                seq_length=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding_topic(topics)
        ''' calculate phi [batch x num_keywords] '''
        phi = None
        phi = torch.sum(mask, dim=1, keepdim=True) * torch.sigmoid(
            self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))

        # loop through sequence
        inputs = embeddings.permute([1, 0, 2]).unbind(0)
        output_states = []
        attn_weight = []
        for i in range(len(inputs)):
            output, hidden, score, coverage_vector = self.decoder(input=inputs[i].unsqueeze(0),
                                                                  output=output,
                                                                  hidden=hidden,
                                                                  phi=phi,
                                                                  topics=topics_embed,
                                                                  coverage_vector=coverage_vector)  # [seq_len x batch x embed_size]
            output_states += [output]
            attn_weight += [score]

        output_states = torch.stack(output_states)
        attn_weight = torch.stack(attn_weight)

        # calculate loss py adaptiveSoftmax
        input = output_states.reshape(-1, output_states.shape[-1])
        target = target.t().reshape((-1,))
        outputs = self.adaptiveSoftmax(input, target)

        return outputs, output_states, hidden, attn_weight, coverage_vector

    def inference(self, inputs, topics, output, hidden=None, mask=None, coverage_vector=None, seq_length=None):
        embeddings = self.embedding(inputs)
        topics_embed = self.embedding_topic(topics)

        phi = None
        phi = seq_length.float() * torch.sigmoid(self.Uf(topics_embed.reshape(topics_embed.shape[0], -1).float()))

        queries = embeddings.permute([1, 0, 2])[-1].unsqueeze(0)

        inputs = queries.permute([1, 0, 2]).unbind(0)
        output_states = []
        attn_weight = []
        for i in range(len(inputs)):
            output, hidden, score, coverage_vector = self.decoder(input=inputs[i].unsqueeze(0),
                                                                  output=output,
                                                                  hidden=hidden,
                                                                  phi=phi,
                                                                  topics=topics_embed,
                                                                  coverage_vector=coverage_vector)  # [seq_len x batch x embed_size]
            output_states += [output]
            attn_weight += [score]

        output_states = torch.stack(output_states)
        attn_weight = torch.stack(attn_weight)

        outputs = self.adaptiveSoftmax.log_prob(output_states.reshape(-1, output_states.shape[-1]))
        return outputs, output_states, hidden, attn_weight, coverage_vector

    def init_hidden(self, batch_size):
        #         hidden = torch.zeros(num_layers, batch_size, hidden_dim)
        #         hidden = LSTMState(torch.zeros(batch_size, hidden_dim).to(device), torch.zeros(batch_size, hidden_dim).to(device))
        hidden = (torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device),
                  torch.zeros(self.num_layers, batch_size, self.hidden_dim).to(device))
        return hidden

    def init_coverage_vector(self, batch_size, num_keywords):
        #         self.coverage_vector = torch.ones([batch_size, num_keywords]).to(device)
        return torch.ones([batch_size, num_keywords]).to(device)


#         print(self.coverage_vector)
def pad_topic(topics):
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    print(topics)
    max_num = 5
    size = 1
    ans = np.zeros((size, max_num), dtype=int)
    for i in range(size):
        true_len = min(len(topics), max_num)
        for j in range(true_len):
            print(topics[i])
            ans[i][j] = topics[i][j]
    return ans


def predict_rnn(topics, num_chars, model, idx_to_word, word_to_idx):
    output_idx = [1]
    topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    topics = topics.reshape((1, topics.shape[0]))
    #     hidden = torch.zeros(num_layers, 1, hidden_dim)
    #     hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device), torch.zeros(num_layers, 1, hidden_dim).to(device))
    hidden = model.init_hidden(batch_size=1)
    if use_gpu:
        #         hidden = hidden.cuda()
        adaptive_softmax.to(device)
        topics = topics.to(device)
    coverage_vector = model.init_coverage_vector(topics.shape[0], topics.shape[1])
    attentions = torch.zeros(num_chars, topics.shape[1])
    for t in range(num_chars):
        X = torch.tensor(output_idx[-1]).reshape((1, 1))
        #         X = torch.tensor(output).reshape((1, len(output)))
        if use_gpu:
            X = X.to(device)
        if t == 0:
            output = torch.zeros(1, hidden_dim).to(device)
        else:
            output = output.squeeze(0)
        pred, output, hidden, attn_weight, coverage_vector = model.inference(inputs=X, topics=topics, output=output,
                                                                             hidden=hidden,
                                                                             coverage_vector=coverage_vector,
                                                                             seq_length=torch.tensor(50).reshape(1,
                                                                                                                 1).to(
                                                                                 device))
        #         print(coverage_vector)
        pred = pred.argmax(dim=1)  # greedy strategy
        attentions[t] = attn_weight[0].data
        #         pred = adaptive_softmax.predict(pred)
        if pred[-1] == 2:
            #         if pred.argmax(dim=1)[-1] == 2:
            break
        else:
            output_idx.append(int(pred[-1]))
    #             output.append(int(pred.argmax(dim=1)[-1]))
    return (
        ''.join([idx_to_word[i] for i in output_idx[1:]]), [idx_to_word[i] for i in output_idx[1:]],
        attentions[:t + 1].t(),
        output_idx[1:])


def beam_search(topics, num_chars, model, idx_to_word, word_to_idx, is_sample=False):
    output_idx = [1]
    # topics = [word_to_idx[x] for x in topics]
    topics = torch.tensor(topics)
    topics = topics.reshape((1, topics.shape[0]))
    #     hidden = torch.zeros(num_layers, 1, hidden_dim)
    #     hidden = (torch.zeros(num_layers, 1, hidden_dim).to(device), torch.zeros(num_layers, 1, hidden_dim).to(device))
    hidden = model.init_hidden(batch_size=1)
    seq_length = torch.tensor(140).reshape(1, 1)  # todo
    if use_gpu:
        #         hidden = hidden.cuda()
        adaptive_softmax.to(device)
        topics = topics.to(device)
        seq_length = seq_length.to(device)
    """1"""
    coverage_vector = model.init_coverage_vector(topics.shape[0], topics.shape[1])
    attentions = torch.zeros(num_chars, topics.shape[1])
    X = torch.tensor(output_idx[-1]).reshape((1, 1)).to(device)
    output = torch.zeros(1, hidden_dim).to(device)
    log_prob, output, hidden, attn_weight, coverage_vector = model.inference(inputs=X,
                                                                             topics=topics,
                                                                             output=output,
                                                                             hidden=hidden,
                                                                             coverage_vector=coverage_vector,
                                                                             seq_length=seq_length)
    log_prob = log_prob.cpu().detach().reshape(-1).numpy()
    #     print(log_prob[10])
    """2"""
    if is_sample:
        top_indices = np.random.choice(vocab_size, beam_size, replace=False, p=np.exp(log_prob))
    else:
        top_indices = np.argsort(-log_prob)
    """3"""
    beams = [(0.0, [idx_to_word[1]], idx_to_word[1], torch.zeros(1, topics.shape[1]), torch.ones(1, topics.shape[1]))]
    b = beams[0]
    beam_candidates = []
    #     print(attn_weight[0].cpu().data, coverage_vector)
    #     assert False
    for i in range(beam_size):
        word_idx = top_indices[i]
        beam_candidates.append((b[0] + log_prob[word_idx], b[1] + [idx_to_word[word_idx]], word_idx,
                                torch.cat((b[3], attn_weight[0].cpu().data), 0),
                                torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0),
                                coverage_vector))
    """4"""
    beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
    beams = beam_candidates[:beam_size]  # truncate to get new beams

    for xy in range(num_chars - 1):
        beam_candidates = []
        for b in beams:
            """5"""
            X = torch.tensor(b[2]).reshape((1, 1)).to(device)
            """6"""
            log_prob, output, hidden, attn_weight, coverage_vector = model.inference(inputs=X,
                                                                                     topics=topics,
                                                                                     output=b[6],
                                                                                     hidden=b[5],
                                                                                     coverage_vector=b[7],
                                                                                     seq_length=seq_length)
            log_prob = log_prob.cpu().detach().reshape(-1).numpy()
            """8"""
            if is_sample:
                top_indices = np.random.choice(vocab_size, beam_size, replace=False, p=np.exp(log_prob))
            else:
                top_indices = np.argsort(-log_prob)
            """9"""
            for i in range(beam_size):
                word_idx = top_indices[i]
                beam_candidates.append((b[0] + log_prob[word_idx], b[1] + [idx_to_word[word_idx]], word_idx,
                                        torch.cat((b[3], attn_weight[0].cpu().data), 0),
                                        torch.cat((b[4], coverage_vector.cpu().data), 0), hidden, output.squeeze(0),
                                        coverage_vector))
        """10"""
        beam_candidates.sort(key=lambda x: x[0], reverse=True)  # decreasing order
        beams = beam_candidates[:beam_size]  # truncate to get new beams

    """11"""
    if '<EOS>' in beams[0][1]:
        first_eos = beams[0][1].index('<EOS>')
    else:
        first_eos = num_chars - 1
    return (
        ''.join(beams[0][1][:first_eos]), beams[0][1][:first_eos], beams[0][3][:first_eos].t(), beams[0][4][:first_eos])


def showAttention(input_sentence, output_words, attentions):
    # Set up figure with colorbar
    fig = plt.figure()
    ax = fig.subplots(1)
    #     cmap = 'bone'
    cmap = 'viridis'
    cax = ax.matshow(attentions.numpy(), cmap=cmap)
    fig.colorbar(cax)

    # Set up axes
    ax.set_yticklabels([''] + input_sentence.split(' '), fontproperties=fontprop, fontsize=10)
    ax.set_xticklabels([''] + output_words, fontproperties=fontprop, fontsize=10, rotation=45)

    # Show label at every tick
    ax.xaxis.set_major_locator(ticker.MultipleLocator(1))
    ax.yaxis.set_major_locator(ticker.MultipleLocator(1))

    word_size = 0.5
    fig.set_figheight(word_size * len(input_sentence.split(' ')))
    fig.set_figwidth(word_size * len(output_words))
    plt.show()


def evaluateAndShowAttention(input_sentence, method='beam_search', is_sample=False):
    if method == 'beam_search':
        _, output_words, attentions, coverage_vector = beam_search(input_sentence, 100, model, idx_to_word, word_to_idx,
                                                                   is_sample=is_sample)
    else:
        _, output_words, attentions, _ = predict_rnn(input_sentence, 100, model, idx_to_word, word_to_idx)
    input_sentence = map(str, input_sentence)
    print('input =', ' '.join(input_sentence))
    print('output =', ' '.join(output_words))
    #     n_digits = 3
    #     coverage_vector = torch.round(coverage_vector * 10**n_digits) / (10**n_digits)
    #     coverage_vector=np.round(coverage_vector, n_digits)
    #     print(coverage_vector.numpy())
    showAttention(' '.join(input_sentence), output_words, attentions)


def evaluate_bleu(model, topics_test, corpus_test, num_test, method='beam_search', is_sample=False):
    bleu_2_score = 0
    for i in tqdm(range(len(corpus_test[:num_test]))):
        if method == 'beam_search':
            _, output_words, _, _ = beam_search([idx_to_word[x] for x in topics_test[i]], 100, model, idx_to_word,
                                                word_to_idx, False)
        else:
            _, output_words, _, _ = predict_rnn([idx_to_word[x] for x in topics_test[i]], 100, model, idx_to_word,
                                                word_to_idx)
        bleu_2_score += sentence_bleu([[idx_to_word[x] for x in corpus_test[i] if x not in [0, 2]]], output_words,
                                      weights=(0, 1, 0, 0))

    bleu_2_score = bleu_2_score / num_test * 100
    return bleu_2_score


model = MTALSTM(hidden_dim=hidden_dim, embed_dim=embedding_dim, num_keywords=num_keywords,
                num_layers=num_layers, num_labels=len(vocab), weight=None, bidirectional=bidirectional, dropout=dropout)
# optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
# optimizer = optim.Adam(model.parameters(), lr=lr)


optimizer = RAdam(model.parameters(), lr=lr)
lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=1e-7, verbose=True)
# optimizer = optim.Adadelta(model.parameters(), lr=lr)
if use_gpu:
    #     model = nn.DataParallel(model)
    #     model = model.to(device)
    model = model.to(device)
    print("Dump to cuda")

model.apply(params_init_uniform)

# Type = 'best'
save_folder = 'model_result_multi_layer'
Type = 'trainable'
model_check_point = '%s/model_%s_%d.pk' % (save_folder, Type, version_num)
optim_check_point = '%s/optim_%s_%d.pkl' % (save_folder, Type, version_num)
loss_check_point = '%s/loss_%s_%d.pkl' % (save_folder, Type, version_num)
epoch_check_point = '%s/epoch_%s_%d.pkl' % (save_folder, Type, version_num)
bleu_check_point = '%s/bleu_%s_%d.pkl' % (save_folder, Type, version_num)
loss_values = []
epoch_values = []
bleu_values = []
if os.path.isfile(model_check_point):
    print('Loading previous status (ver.%d)...' % version_num)
    model.load_state_dict(torch.load(model_check_point, map_location='cpu'))
    model = model.to(device)
    optimizer.load_state_dict(torch.load(optim_check_point))
    lr_scheduler = ReduceLROnPlateau(optimizer, mode='min', factor=0.4, patience=2, min_lr=1e-7, verbose=True)
    loss_values = torch.load(loss_check_point)
    epoch_values = torch.load(epoch_check_point)
    bleu_values = torch.load(bleu_check_point)
    print('Load successfully')
else:
    print("ver.%d doesn't exist" % version_num)

# evaluateAndShowAttention(['現在', '未來', '夢想', '科學', '文化'], method='beam_search', is_sample=True)


def isnan(x):
    return x != x


# for name, p in model.named_parameters():
#     #     if p.grad is None:
#     #         continue
#     if p.requires_grad:
#         print(name, p)


#         p.register_hook(lambda grad: torch.clamp(grad, -clip_value, clip_value))

def decay_lr(optimizer, epoch, factor=0.1, lr_decay_epoch=60):
    if epoch % lr_decay_epoch == 0:
        for param_group in optimizer.param_groups:
            param_group['lr'] = param_group['lr'] * factor
        print('lr decayed to %.4f' % optimizer.param_group[0]['lr'])
    return optimizer


since = time.time()
autograd.set_detect_anomaly(False)
prev_epoch = 0 if not epoch_values else epoch_values[-1]
best_bleu = 0 if not bleu_values else max(bleu_values)

for epoch in range(num_epoch - prev_epoch):
    epoch += prev_epoch
    start = time.time()
    num, total_loss = 0, 0
    #     optimizer = decay_lr(optimizer=optimizer, epoch=epoch+1)
    topics_indice, corpus_indice = shuffleData(topics_indice, corpus_indice)  # shuffle data at every epoch
    # max_length = max(length) + 1
    max_length = 140  # todo
    data = data_iterator(corpus_indice, topics_indice, batch_size, max_length)
    hidden = model.init_hidden(batch_size=batch_size)
    weight = torch.ones(len(vocab))
    weight[0] = 0
    num_iter = len(corpus_indice) // batch_size
    for X, Y, mask, topics in tqdm(data, total=num_iter):
        num += 1
        #         hidden.detach_()
        if use_gpu:
            X = X.to(device)
            Y = Y.to(device)
            mask = mask.to(device)
            topics = topics.to(device)
            #             hidden = hidden.to(device)
            #             hidden[0].to(device)
            #             hidden[1].to(device)
            loss_function = loss_function.to(device)
            weight = weight.to(device)
        optimizer.zero_grad()
        # init hidden layer
        #         hidden = model.init_hidden(num_layers, batch_size, hidden_dim)
        coverage_vector = model.init_coverage_vector(batch_size, num_keywords)
        init_output = torch.zeros(batch_size, hidden_dim).to(device)
        # inputs, topics, output, hidden=None, mask=None, target=None, coverage_vector=None, seq_length=None):
        output, _, hidden, _, _ = model(inputs=X, topics=topics, output=init_output, hidden=hidden, mask=mask, target=Y,
                                        coverage_vector=coverage_vector)
        #         output, hidden = model(X, topics)
        hidden[0].detach_()
        hidden[1].detach_()

        loss = (-output.output).reshape((-1, batch_size)).t() * mask
        #         loss = loss.sum(dim=1)
        loss = loss.sum(dim=1) / mask.sum(dim=1)
        loss = loss.mean()
        loss.backward()

        norm = 0.0
        #         norm = nn.utils.clip_grad_norm_(model.parameters(), 10)
        nn.utils.clip_grad_value_(model.parameters(), 1)

        optimizer.step()
        total_loss += float(loss.item())

        if np.isnan(total_loss):
            for name, p in model.named_parameters():
                if p.grad is None:
                    continue
                print(name, p)
            assert False, "Gradient explode"

    one_iter_loss = np.mean(total_loss)
    lr_scheduler.step(one_iter_loss)
    #     print("One iteration loss {:.3f}".format(one_iter_loss))

    # validation
    bleu_score = 0
    num_test = len(topics_test)
    bleu_score = evaluate_bleu(model, topics_test, corpus_test, num_test=num_test, method='predict_rnn',
                               is_sample=False)

    bleu_values.append(bleu_score)
    loss_values.append(total_loss / num)
    epoch_values.append(epoch + 1)

    # save checkpoint
    if ((epoch + 1) % check_point == 0) or (epoch == (num_epoch - 1)) or epoch + 1 > 90 or bleu_score > 4:
        model_check_point = '%s/model_trainable_%d.pk' % (save_folder, epoch + 1)
        optim_check_point = '%s/optim_trainable_%d.pkl' % (save_folder, epoch + 1)
        loss_check_point = '%s/loss_trainable_%d.pkl' % (save_folder, epoch + 1)
        epoch_check_point = '%s/epoch_trainable_%d.pkl' % (save_folder, epoch + 1)
        bleu_check_point = '%s/bleu_trainable_%d.pkl' % (save_folder, epoch + 1)
        torch.save(model.state_dict(), model_check_point)
        torch.save(optimizer.state_dict(), optim_check_point)
        torch.save(loss_values, loss_check_point)
        torch.save(epoch_values, epoch_check_point)
        torch.save(bleu_values, bleu_check_point)

    # save current best result
    if bleu_score > best_bleu:
        best_bleu = bleu_score
        print('current best bleu: %.4f' % best_bleu)
        model_check_point = '%s/model_best_%d.pk' % (save_folder, epoch + 1)
        optim_check_point = '%s/optim_best_%d.pkl' % (save_folder, epoch + 1)
        loss_check_point = '%s/loss_best_%d.pkl' % (save_folder, epoch + 1)
        epoch_check_point = '%s/epoch_best_%d.pkl' % (save_folder, epoch + 1)
        bleu_check_point = '%s/bleu_best_%d.pkl' % (save_folder, epoch + 1)
        torch.save(model.state_dict(), model_check_point)
        torch.save(optimizer.state_dict(), optim_check_point)
        torch.save(loss_values, loss_check_point)
        torch.save(epoch_values, epoch_check_point)
        torch.save(bleu_values, bleu_check_point)

    # calculate time
    end = time.time()
    s = end - since
    h = math.floor(s / 3600)
    m = s - h * 3600
    m = math.floor(m / 60)
    s -= (m * 60 + h * 3600)

    # verbose
    if ((epoch + 1) % verbose == 0) or (epoch == (num_epoch - 1)):
        print('epoch %d/%d, loss %.4f, norm %.4f, predict bleu: %.4f, time %.3fs, since %dh %dm %ds'
              % (epoch + 1, num_epoch, total_loss / num, norm, bleu_score, end - start, h, m, s))
        evaluateAndShowAttention([0, 32], method='beam_search', is_sample=False)
        # evaluateAndShowAttention(['現在', '未來', '夢想', '科學', '文化'], method='beam_search', is_sample=False)
