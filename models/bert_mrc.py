# -*- coding: utf-8 -*-
# @Time : 2021/8/16 14:32
# @Author : haojie zhang

import numpy as np
import tensorflow as tf
from bert4keras.backend import keras, K
from bert4keras.models import build_transformer_model

from bert4keras.layers import Loss
from bert4keras.snippets import sequence_padding, DataGenerator, to_array
from bert4keras.tokenizers import Tokenizer
from keras.layers import Input, Dense, Lambda, Dropout
from keras.models import Model
import json
from tqdm import tqdm
from bert4keras.optimizers import *


# 超参数
maxlen = 128
batch_size = 4
learning_rate = 8e-6
epochs = 10

weight_span = 0.1 # default = 0.1
weight_start = 1.0
weight_end = 1.0

intermediate_hidden_size = 1536

# bert配置
config_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/bert_config.json'
checkpoint_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/bert_model.ckpt'
dict_path = 'E:/kg/bert/chinese_L-12_H-768_A-12/vocab.txt'


def load_data(filename):
    """加载数据
    单挑格式：[[text, query, label, (start_pos, end_pos, span_pos), ...]

    """

    with open(filename, 'r', encoding='utf-8') as infile:
        lines = json.load(infile)
        D = []
        for line in lines:
            text = line['context'].replace(' ', '')
            query = line['query']
            start_position = line['start_position']
            end_position = line['end_position']
            D.append([text, query, start_position, end_position])
        return D


def load_query(filename):
    with open(filename, 'r', encoding='utf-8') as infile:
        lines = json.load(infile)
        label2query = lines['default']
        query2label = {j:i for i, j in label2query.items()}
    return query2label


query2label = load_query('../query/zh_msra.json')


train_data = load_data('../data/mrc-ner.train')
valid_data = load_data('../data/mrc-ner.dev')

categories = list(sorted(query2label))


# 建立分词器
tokenizer = Tokenizer(dict_path, do_lower_case=True)


class data_generator(DataGenerator):
    def __iter__(self, random=False):
        batch_token_ids, batch_segment_ids, batch_start_labels, batch_end_labels, batch_start_label_mask,\
        batch_end_label_mask, batch_match_labels = [], [], [], [], [], [], []
        cnt = 0
        for is_end, d in self.sample(random):
            text, query, start_positions, end_positions = d
            token_ids, segment_ids = tokenizer.encode(query, text, maxlen=maxlen)
            tokens = tokenizer.tokenize(text)
            mapping = tokenizer.rematch(text, tokens)

            origin_offset2token_idx_start = {}
            origin_offset2token_idx_end = {}
            len_query = 0
            for token_idx in range(len(token_ids)-1):
                if segment_ids[token_idx] == 0:
                    len_query += 1
                    continue

                token_start = mapping[token_idx - len_query + 1][0]
                token_end = mapping[token_idx - len_query + 1][-1]
                origin_offset2token_idx_start[token_start] = token_idx
                origin_offset2token_idx_end[token_end] = token_idx

            new_start_positions = []
            new_end_positions = []

            for start, end in zip(start_positions, end_positions):
                if start in origin_offset2token_idx_start and end in origin_offset2token_idx_end:
                    new_start_positions.append(origin_offset2token_idx_start[start])
                    new_end_positions.append(origin_offset2token_idx_end[end])
                else:
                    cnt += 1

            label_mask = [0] * len_query + [1] * (len(token_ids) - len_query - 1) + [0]
            start_label_mask = label_mask
            end_label_mask = label_mask

            assert all(start_label_mask[p] != 0 for p in new_start_positions)
            assert all(end_label_mask[p] != 0 for p in new_end_positions)

            assert len(new_start_positions) == len(new_end_positions)
            assert len(label_mask) == len(token_ids)

            start_labels = [(1 if idx in new_start_positions else 0)
                            for idx in range(len(token_ids))]
            end_labels = [(1 if idx in new_end_positions else 0)
                          for idx in range(len(token_ids))]

            seq_len = len(token_ids)
            match_labels = np.zeros((maxlen, maxlen))
            for start, end in zip(new_start_positions, new_end_positions):
                if start >= seq_len or end >= seq_len:
                    continue
                match_labels[start, end] = 1

            batch_token_ids.append(token_ids)
            batch_segment_ids.append(segment_ids)
            batch_start_labels.append(start_labels)
            batch_end_labels.append(end_labels)
            batch_start_label_mask.append(start_label_mask)
            batch_end_label_mask.append(end_label_mask)
            batch_match_labels.append(match_labels)

            if len(batch_token_ids) == self.batch_size or is_end:
                batch_token_ids = sequence_padding(batch_token_ids, maxlen)
                batch_segment_ids = sequence_padding(batch_segment_ids, maxlen, value=1)
                batch_start_labels = sequence_padding(batch_start_labels, maxlen)
                batch_end_labels = sequence_padding(batch_end_labels, maxlen)
                batch_start_label_mask = sequence_padding(batch_start_label_mask, maxlen)
                batch_end_label_mask = sequence_padding(batch_end_label_mask, maxlen)
                batch_match_labels = sequence_padding(batch_match_labels, maxlen)
                yield [batch_token_ids, batch_segment_ids, batch_start_labels, batch_end_labels,
                       batch_start_label_mask, batch_end_label_mask, batch_match_labels], None
                batch_token_ids, batch_segment_ids, batch_start_labels, batch_end_labels, batch_start_label_mask, \
                batch_end_label_mask, batch_match_labels = [], [], [], [], [], [], []


class TotalLoss(Loss):
    def __init__(self, *args, **kwargs):
        super(TotalLoss, self).__init__(*args, **kwargs)
        self.weight_start = weight_start
        self.weight_end = weight_end
        self.weight_span = weight_span
        self.weight_sum = self.weight_start + self.weight_end + self.weight_span
        self.weight_start = self.weight_start / self.weight_sum
        self.weight_end = self.weight_end / self.weight_sum
        self.weight_span = self.weight_span / self.weight_sum

    def compute_loss(self, inputs, mask=None):

        start_outputs, end_outputs, match_outputs, \
        start_labels, end_labels, match_labels, start_label_mask, end_label_mask = inputs

        match_label_row_mask = K.cast(
            K.repeat_elements(K.expand_dims(start_label_mask, axis=-1), maxlen, axis=-1), 'bool')
        match_label_col_mask = K.cast(
            K.repeat_elements(K.expand_dims(end_label_mask, axis=-2), maxlen, axis=-2), 'bool')

        match_label_mask = match_label_row_mask & match_label_col_mask
        match_label_mask = tf.linalg.band_part(match_label_mask, 0, -1)

        # use only pred or golden start/end to compute match loss
        start_labels_candidates = K.cast(
            K.repeat_elements(K.expand_dims(start_labels, axis=-1), maxlen, axis=-1), 'bool')
        end_labels_candidates = K.cast(
            K.repeat_elements(K.expand_dims(end_labels, axis=-2), maxlen, axis=-2), 'bool')
        match_candidates = start_labels_candidates & end_labels_candidates

        match_label_mask = match_label_mask & match_candidates
        match_label_mask = K.cast(match_label_mask, K.floatx())

        # 计算损失
        start_loss = K.binary_crossentropy(start_labels, start_outputs[:, :, 0])
        start_loss = K.sum(start_loss * start_label_mask) / K.sum(start_label_mask)

        end_loss = K.binary_crossentropy(end_labels, end_outputs[:, :, 0])
        end_loss = K.sum(end_loss * end_label_mask) / K.sum(end_label_mask)

        match_loss = K.binary_crossentropy(match_labels, match_outputs[:, :, :, 0])
        match_loss = K.sum(match_loss * match_label_mask) / (K.sum(match_label_mask) + 1e-12)

        return self.weight_start * start_loss + self.weight_end * end_loss + self.weight_span * match_loss


def getSpanMatrix(inputs):
    """
    # for every position $i$ in sequence, should concate $j$ to
    # predict if $i$ and $j$ are start_pos and end_pos for an entity.
    # [batch, seq_len, seq_len, hidden]
    """
    sequence_heatmap = inputs
    start_extend = K.repeat_elements(K.expand_dims(sequence_heatmap, axis=2), maxlen, axis=2)  # [batch, seq_len, seq_len, hidden]
    end_extend = K.repeat_elements(K.expand_dims(sequence_heatmap, axis=1), maxlen, axis=1)  # [batch, seq_len, seq_len, hidden]
    match_matrix = K.concatenate([start_extend, end_extend], axis=-1)  # [batch, seq_len, seq_len, hidden*2]
    return match_matrix


# 加载预训练模型
start_labels = Input(shape=(maxlen, ), name='start_labels')
end_labels = Input(shape=(maxlen, ), name='end_labels')
start_label_mask = Input(shape=(maxlen, ), name='start_label_mask')
end_label_mask = Input(shape=(maxlen, ), name='end_label_mask')
match_labels = Input(shape=(None, maxlen), name='match_labels')


bert = build_transformer_model(
    config_path,
    checkpoint_path,
    return_keras_model=False,
)

start_outputs = Dense(
    units=1,
    kernel_initializer=bert.initializer,
    activation='sigmoid'
)(bert.model.output)

end_outputs = Dense(
    units=1,
    kernel_initializer=bert.initializer,
    activation='sigmoid'
)(bert.model.output)


dense1 = Dense(
    units=intermediate_hidden_size,
    kernel_initializer=bert.initializer,
    activation='gelu',
)
# dropout = Dropout(rate=0.1)

dense2 = Dense(
    units=1,
    kernel_initializer=bert.initializer,
    activation='sigmoid'
)


match_matrix = Lambda(getSpanMatrix)(bert.model.output)
match_outputs = dense2(dense1(match_matrix))

totalloss = TotalLoss([0, 1, 2])
start_outputs_, end_outputs_, match_outputs_ = \
    totalloss([start_outputs, end_outputs, match_outputs,
                         start_labels, end_labels, match_labels, start_label_mask, end_label_mask])

# Model()里馈入张量顺序需与data_generator顺序保持一致
train_model = Model(
            bert.model.inputs + [start_labels, end_labels, start_label_mask, end_label_mask, match_labels],
            [start_outputs_, end_outputs_, match_outputs_]
)
train_model.summary()

mrc_model = Model(
    bert.model.inputs,
    [start_outputs, end_outputs, match_outputs]
)


# optimizer = AccumOptimizer(Adam(lr=learning_rate), 2)
# optimizer = Adam(lr=learning_rate)
AdamW = extend_with_weight_decay(Adam, 'AdamW')
AdamGA = extend_with_gradient_accumulation(AdamW, 'AdamGA')
AdamWLR = extend_with_piecewise_linear_lr(AdamGA, 'AdamWLR')
optimizer = AdamWLR(lr=learning_rate,
                  weight_decay_rate=0.01,
                  grad_accum_steps=2,
                  lr_schedule={2000: 1}
                   )

train_model.compile(optimizer=optimizer)


class NameEntityRecognizer(object):
    def recognize(self, data, threshold=0.5):
        text, query, start_positions, end_positions = data
        tokens = tokenizer.tokenize(text, maxlen=maxlen)
        mapping = tokenizer.rematch(text, tokens)
        span_triple_lst = []
        token_ids, segment_ids = tokenizer.encode(query, text, maxlen=maxlen)
        token_ids, segment_ids = to_array([token_ids], [segment_ids])
        token_ids = sequence_padding(token_ids, maxlen)  # 需要填充到maxlen
        segment_ids = sequence_padding(segment_ids, maxlen, value=1)  # 需要填充到maxlen
        text_start_idx = np.where(token_ids[0] == 102)[0][0]
        start_pred, end_pred, match_pred = mrc_model.predict([token_ids, segment_ids])
        start_pred = np.where(start_pred[0, :, 0] > threshold)[0]
        end_pred = np.where(end_pred[0, :, 0] > threshold)[0]
        label = query2label[query]
        for s in start_pred:
            for e in end_pred:
                if s <= e and match_pred[0][s][e] > threshold:
                    ts = s - text_start_idx
                    te = e - text_start_idx
                    if ts >= len(mapping) or te >= len(mapping) or ts < 0 or te < 0:
                        continue
                    if mapping[ts] and mapping[te]:
                        span_triple_lst.append((mapping[ts][0], mapping[te][-1], label))

        return [(text[i:j+1], label) for i, j, label in span_triple_lst]

    def query_span_f1(self, valid_data, steps):
        tp, fp, fn = 1e-10, 1e-10, 1e-10
        for _ in tqdm(range(steps), ncols=100):
            data = next(valid_data)
            data = data[0]

            start_preds, end_preds, match_preds = mrc_model.predict(data[:2])
            start_label_mask, end_label_mask, match_labels = data[-3:]
            start_label_mask = start_label_mask.astype('bool')
            end_label_mask = end_label_mask.astype('bool')
            match_labels = match_labels.astype('bool')
            bsz, seq_len = start_label_mask.shape
            # [bsz, seq_len, seq_len]
            match_preds = match_preds.squeeze() > 0.5
            # [bsz, seq_len]
            start_preds = start_preds.squeeze() > 0.5
            # [bsz, seq_len]
            end_preds = end_preds.squeeze() > 0.5

            match_preds = (match_preds
                           & np.repeat(np.expand_dims(start_preds, axis=-1), seq_len, -1)
                           & np.repeat(np.expand_dims(end_preds, axis=1), seq_len, 1))
            match_label_mask = (np.repeat(np.expand_dims(start_label_mask, axis=-1), seq_len, -1)
                                & np.repeat(np.expand_dims(end_label_mask, axis=1), seq_len, 1))
            match_label_mask = np.triu(match_label_mask)
            match_preds = match_label_mask & match_preds

            tp += np.sum(match_labels & match_preds)
            fp += np.sum(~match_labels & match_preds)
            fn += np.sum(match_labels & ~match_preds)

        precision = tp / (tp + fp)
        recall = tp / (tp + fn)
        f1 = 2 * precision * recall / (precision + recall)
        return f1, precision, recall


NER = NameEntityRecognizer()


def evaluate(data):
    """评测函数
    """
    X, Y, Z = 1e-10, 1e-10, 1e-10
    for d in tqdm(data, ncols=100):
        text = d[0]
        label = query2label[d[1]]
        R = set(NER.recognize(d))
        T = set([(text[i:j+1], label) for i, j in zip(d[2], d[3])])
        X += len(R & T)
        Y += len(R)
        Z += len(T)
    f1, precision, recall = 2 * X / (Y + Z), X / Y, X / Z
    return f1, precision, recall


class Evaluator(keras.callbacks.Callback):
    def __init__(self):
        self.best_val_f1 = 0

    def on_train_begin(self, logs=None):
        # f1, precision, recall = evaluate(valid_data) # 备选评估
        f1, precision, recall = NER.query_span_f1(valid_generator.forfit(), len(valid_generator))
        self.best_val_f1 = f1
        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )
        with open(file_dir, 'a', encoding='utf-8', ) as f:
            f.write(
                'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                (f1, precision, recall, self.best_val_f1)
            )

    def on_epoch_end(self, epoch, logs=None):
        # f1, precision, recall = evaluate(valid_data) # 备选评估
        f1, precision, recall = NER.query_span_f1(valid_generator.forfit(), len(valid_generator))
        if f1 > self.best_val_f1:
            self.best_val_f1 = f1
            mrc_model.save_weights('best_model.weights')

        print(
            'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
            (f1, precision, recall, self.best_val_f1)
        )

        with open(file_dir, 'a', encoding='utf-8', ) as f:
            f.write(
                'f1: %.5f, precision: %.5f, recall: %.5f, best f1: %.5f\n' %
                (f1, precision, recall, self.best_val_f1)
            )


file_dir = '../log'

if __name__ == "__main__":
    train_generator = data_generator(train_data, batch_size)
    valid_generator = data_generator(valid_data, batch_size=batch_size)
    evaluator = Evaluator()

    # train_model.load_weights('best_model.weights')

    train_model.fit(
        train_generator.forfit(),
        steps_per_epoch=len(train_generator),
        epochs=epochs,
        callbacks=[evaluator],
    )


