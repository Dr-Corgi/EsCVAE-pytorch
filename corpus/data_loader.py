# -*- coding:utf-8 -*-
"""
Created on 18/11/27 下午12:10.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import numpy as np


class DataLoader(object):

    def __init__(self, name, data, config):

        self.name = name
        self.data = data
        self.data_size = len(data)
        self.max_utt_len = config.max_seq_len
        self.vocab_size = config.vocab_size
        self.data_lens = all_lens = [len(line) for (line, _), _ in self.data]

        if self.data_size > 0:
            print("Max len %d and min len %d and avg len %f" % (np.max(all_lens),
                                                                np.min(all_lens),
                                                                float(np.mean(all_lens))))
        else:
            print("data_size == 0.")

        self.indexes = list(np.argsort(all_lens))[::-1]

        self.num_batch = None

    def pad_to(self, tokens, do_pad=True):
        if len(tokens) >= self.max_utt_len:
            return tokens[0:self.max_utt_len-1] + [tokens[-1]]
        elif do_pad:
            return tokens + [0] * (self.max_utt_len - len(tokens))
        else:
            return tokens

    def _shuffle_batch_indexes(self):
        np.random.shuffle(self.batch_indexes)

    def epoch_init(self, batch_size, shuffle=True, intra_shuffle=True):
        assert len(self.indexes) == self.data_size and len(self.data_lens) == self.data_size

        self.ptr = 0
        self.batch_size = batch_size

        # create batch indexes
        temp_num_batch = self.data_size // batch_size
        self.batch_indexes = []
        for i in range(temp_num_batch):
            tmp_batch_indexes = self.indexes[i * batch_size: (i+1) * batch_size]
            if intra_shuffle and shuffle:
                np.random.shuffle(tmp_batch_indexes)
            self.batch_indexes.append(tmp_batch_indexes)

        left_over = self.data_size - temp_num_batch * batch_size

        # shuffle batch indexes
        if shuffle:
            self._shuffle_batch_indexes()

        self.num_batch = len(self.batch_indexes)
        print("%s begins with %d batches with %d left for samples" % (self.name, self.num_batch, left_over))

    def next_batch(self):
        if self.ptr < self.num_batch:
            current_batch = self.batch_indexes[self.ptr]
            self.ptr += 1
            return self._prepare_batch(cur_batch=current_batch)
        else:
            return None

    def set_data(self, data):
        self.data = data
        self.data_size = len(data)
        self.data_lens = all_lens = [len(line) for (line, _), _ in self.data]
        self.indexes = list(np.argsort(all_lens))[::-1]
        self.num_batch = None

    def _prepare_batch(self, cur_batch):

        sentence_dat = [self.data[idx] for idx in cur_batch]

        context_lens, context_utts, context_emotions, out_utts, out_lens, out_emotions = [],[],[],[],[],[]

        for (post, post_e), (response, response_e) in sentence_dat:
            context_utts.append(self.pad_to(post))
            context_lens.append(len(post))
            context_emotions.append(post_e)

            out_utt = self.pad_to(response, do_pad=False)
            out_utts.append(out_utt)
            out_lens.append(len(out_utt))
            out_emotions.append(response_e)

        vec_contexts = np.zeros((self.batch_size, self.max_utt_len), dtype=np.int64)
        vec_context_lens = np.array(context_lens, dtype=np.int64)
        vec_context_emotions = np.array(context_emotions, dtype=np.int64)

        vec_outs = np.zeros((self.batch_size, np.max(out_lens)), dtype=np.int64)
        vec_out_lens = np.array(out_lens, dtype=np.int64)
        vec_out_emotions = np.array(out_emotions, dtype=np.int64)

        for idx in range(self.batch_size):
            vec_outs[idx, 0:vec_out_lens[idx]] = out_utts[idx]
            vec_contexts[idx] = context_utts[idx]

        vec_outs_bows = np.zeros((self.batch_size, self.vocab_size), dtype=np.float32)
        for idx in range(self.batch_size):
            for widx in out_utts[idx]:
                vec_outs_bows[idx, widx] = 1.

        return vec_contexts, vec_context_lens, vec_context_emotions, vec_outs, vec_out_lens, vec_out_emotions, vec_outs_bows
