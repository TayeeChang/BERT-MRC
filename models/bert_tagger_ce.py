#! -*- coding: utf-8 -*-
# 用CRF做中文命名实体识别
# 数据集 http://s3.bmio.net/kashgari/china-people-daily-ner-corpus.tar.gz
# 实测验证集的F1可以到96.48%，测试集的F1可以到95.38%

import numpy as np
from bert4keras.backend import keras, K, batch_gather
from bert4keras.models import build_transformer_model
from bert4keras.tokenizers import Tokenizer
from bert4keras.optimizers import Adam, extend_with_gradient_accumulation, extend_with_weight_decay, extend_with_piecewise_linear_lr
from bert4keras.snippets import sequence_padding, DataGenerator
from bert4keras.snippets import open, to_array
from bert4keras.layers import Loss, Input
from keras.layers import Dense
from keras.models import Model
from tqdm import tqdm
import json
from collections import OrderedDict

# 模型超参数
maxlen = 128
epochs = 10
batch_size = 4
bert_layers = 12
learning_rate = 2e-5  # bert_layers越小，学习率应该要越大
categories = set()


# bert配置
config_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单条格式：[text, (start, end, label), (start, end, label), ...]，
              意味着text[start:end + 1]是类型为label的实体。
    """
    D = []
    with open(filename, encoding='utf-8') as f:
        f = f.read()
        for l in f.split('\n\n'):
            if not l:
                continue
            d = ['']
            for i, c in enumerate(l.split('\n')):
                char, flag = c.split(' ')
                d[0] += char
                if flag[0] == 'B':
                    d.append([i, i, flag[2:]])
                    categories.add(flag[2:])
                elif flag[0] == 'I':
                    d[-1][1] = i
            D.append(d)
    return D


# 标注数据
train_data = load_data('E:\\kg\\corpus\\china-people-daily-ner-corpus/example.train')
valid_data = load_data('E:\\kg\\corpus\\china-people-daily-ner-corpus/example.dev')
test_data = load_data('E:\\kg\\corpus\\china-people-daily-ner-corpus/example.test')
categories = list(sorted(categories))

# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    """数据生成器
    """
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_labels = [], [], []
        for is_end, d in self.sample(random):
            tokens = tokenizer.tokenize(d[0], maxlen=maxlen)
            mapping = tokenizer.rematch(d[0], tokens)
            start_mapping = {j[0]: i for i, j in enumerate(mapping) if j}
            end_mapping = {j[-1]: i for i, j in enumerate(mapping) if j}
            token_ids = tokenizer.tokens_to_ids(tokens)
            segment_ids = [0] * len(token_ids)
            labels = np.zeros(len(token_ids))
            for start, end, label in d[1:]:
                if start in start_mapping and end in end_mapping:
                    start = start_mapping[start]
                    end = end_mapping[end]
                    labels[start] = categories.index(label) * 2 + 1
                    labels[start + 1:end + 1] = categories.index(label) * 2 + 2
            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_labels.append(labels)
            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids)
                batch_segment_ids = sequence_padding(batch_segment_ids)
                batch_labels = sequence_padding(batch_labels)
                yield [batch_token_ids, batch_segment_ids, batch_labels], None
                batch_token_ids, batch_segment_ids, batch_labels = [], [], []


class TotalLoss(Loss):
    def compute_loss(self, inputs, mask=None):
        y_true, y_pred = inputs
        if mask[1] is None:
            mask = 1.0
        else:
            mask = K.cast(mask[1], K.floatx())
        loss = K.sparse_categorical_crossentropy(y_true, y_pred)
        loss = K.sum(loss * mask) / K.sum(mask)
        return loss


labels = Input(shape=(None,), name='labels')

model = build_transformer_model(
    config_path,
    checkpoint_path,
)

output_layer = 'Transformer-%s-FeedForward-Norm' % (bert_layers - 1)
output = model.get_layer(output_layer).output

# 分类器
output = Dense(1536, activation='gelu')(output)
output = Dense(len(categories) * 2 + 1, activation='softmax')(output)
output = TotalLoss([1])([labels, output])
#
model = Model(model.input + [labels], output)

model.summary()

adamW = extend_with_weight_decay(Adam, 'adamW')
adamWLR = extend_with_piecewise_linear_lr(adamW, 'adamWLR')
optimizer = adamWLR(lr=learning_rate,
                    weight_decay_rate=0.01,
                    lr_schedule={1000: 1})

model.compile(optimizer=optimizer)

# 解码器
def recognize(text):
    tokens = tokenizer.tokenize(text, maxlen=512)
    mapping = tokenizer.rematch(text, tokens)
    token_ids = tokenizer.tokens_to_ids(tokens)
    segment_ids = [0] * len(token_ids)
    token_ids, segment_ids = to_array([token_ids], [segment_ids])
    nodes = model.predict([token_ids, segment_ids, np.zeros_like(token_ids)])[0]
    labels = nodes.argmax(axis=-1)
    entities, starting = [], False
    for i, label in enumerate(labels):
        if label > 0:
            if label % 2 == 1:
                starting = True
                entities.append([[i], categories[(label - 1) // 2]])
            elif starting:
                entities[-1][0].append(i)
            else:
                starting = False
        else:
            starting = False

    return [(mapping[w[0]][0], mapping[w[-1]][-1], l) for w, l in entities if mapping[w[0]] and mapping[w[-1]]]


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        R = set(recognize(d[0]))
        T = set([tuple(i) for i in d[1:]])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    """评估与保存
    """
    def __init__(self):
        self.best_val_f1 = 0

    def on_epoch_end(self, epoch, logs=None):

        f1, precision, recall = evaluate(valid_data)
        # 保存最优
        if f1 >= self.best_val_f1:
            self.best_val_f1 = f1
            model.save_weights('../weights/ner_crf_ce.best_model.weights')
        print(
            'valid:  f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

        row_dict = OrderedDict()
        row_dict['epoch'] = epoch
        row_dict.update(logs)

        row_dict['best_f1_val'] = self.best_val_f1
        row_dict['f1_valid'] = f1
        row_dict['precision_valid'] = precision
        row_dict['recall_valid'] = recall

        f1, precision, recall = evaluate(test_data)
        print(
            'test:  f1: %.5f, precision: %.5f, recall: %.5f\n' %
            (f1, precision, recall)
        )
        row_dict['f1_test'] = f1
        row_dict['precision_test'] = precision
        row_dict['recall_test'] = recall

        with open(dir, 'a', encoding='utf-8') as f:
            f.write(json.dumps(row_dict) + '\n')



dir = '../log/bert_tagger_ce.log'

if __name__ == '__main__':

    evaluator = Evaluator()
    train_generator = data_generator(train_data, batch_size)

    model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator])

else:
    model.load_weights('../weights/ner_crf_ce.best_model.weights')
