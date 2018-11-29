# -*- coding:utf-8 -*-
"""
Created on 18/11/27 下午12:10.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
import numpy as np
import json
import codecs


class Corpus(object):

    def __init__(self, corpus_path, max_vocab_cnt=3000, word2vec=None, word2vec_dim=None, add_start_tok=True):

        self._path = corpus_path
        self.word2vec_path = word2vec
        self.word2vec_dim = word2vec_dim
        self.word2vec = None

        self.max_vocab_cnt = max_vocab_cnt

        data = self.load_data(add_start_tok=add_start_tok)
        self.train_corpus = data['train_data']
        self.valid_corpus = data['valid_data']
        self.test_corpus = data['test_data']

        self.load_word2vec()
        print("Done loading corpus.")

    def load_data(self, add_start_tok=True):

        words = {}
        data = {}

        special_token = ["<pad>", "<unk>", "<s>", "</s>"]

        self.pad_id = 0
        self.unk_id = 1
        self.go_id = 2
        self.eos_id = 3

        # load data
        dat = json.load(codecs.open(self._path, 'r', 'utf8'))

        train_data_raw = dat['train']
        valid_data_raw = dat['valid']
        test_data_raw = dat['test']

        train_data = []
        valid_data = []
        test_data = []

        for (post, post_e), (response, response_e) in train_data_raw:
            if add_start_tok:
                new_post = ["<s>"] + post.strip().split(" ") + ["</s>"]
                new_response = ["<s>"] + response.strip().split(" ") + ["</s>"]
            else:
                new_post = post.strip().split(" ") + ["</s>"]
                new_response = response.strip().split(" ") + ["</s>"]

            train_data.append(((new_post, post_e), (new_response, response_e)))

            for word in new_post:
                if word not in special_token:
                    words[word] = words.get(word, 0) + 1
            for word in new_response:
                if word not in special_token:
                    words[word] = words.get(word, 0) + 1

        for (post, post_e), (response, response_e) in valid_data_raw:
            if add_start_tok:
                new_post = ["<s>"] + post.strip().split(" ") + ["</s>"]
                new_response = ["<s>"] + response.strip().split(" ") + ["</s>"]
            else:
                new_post = post.strip().split(" ") + ["</s>"]
                new_response = response.strip().split(" ") + ["</s>"]

            valid_data.append(((new_post, post_e), (new_response, response_e)))

            for word in new_post:
                if word not in special_token:
                    words[word] = words.get(word, 0) + 1
            for word in new_response:
                if word not in special_token:
                    words[word] = words.get(word, 0) + 1

        for (post, post_e), (response, response_e) in test_data_raw:
            if add_start_tok:
                new_post = ["<s>"] + post.strip().split(" ") + ["</s>"]
                new_response = ["<s>"] + response.strip().split(" ") + ["</s>"]
            else:
                new_post = post.strip().split(" ") + ["</s>"]
                new_response = response.strip().split(" ") + ["</s>"]

            test_data.append(((new_post, post_e), (new_response, response_e)))

            for word in new_post:
                if word not in special_token:
                    words[word] = words.get(word, 0) + 1
            for word in new_response:
                if word not in special_token:
                    words[word] = words.get(word, 0) + 1

        # create vocab
        sorted_words = special_token + [w for w, _ in sorted(words.items(), key=lambda x: x[1], reverse=True)]
        self.vocab = sorted_words[:self.max_vocab_cnt]
        self.rev_vocab = {t: idx for idx, t in enumerate(self.vocab)}

        # create emotion vocab
        self.emotion_vocab = {0:0, 1:1, 2:2, 3:3, 4:4, 5:5}
        self.emotion_show_vocab = {0: "其他", 1: "喜爱", 2: "悲伤", 3: "厌恶", 4: "愤怒", 5: "快乐"}
        self.rev_emotion_vocab = {t: idx for idx, t in enumerate(self.emotion_vocab)}

        data['train_data'] = train_data
        data['valid_data'] = valid_data
        data['test_data'] = test_data

        return data

    def load_word2vec(self):
        if self.word2vec_path is None:
            return
        with codecs.open(self.word2vec_path, 'r', 'utf8') as f:
            lines = f.readlines()
        raw_word2vec = {}
        for l in lines[1:]:
            splits_ = l.strip().split()
            w = splits_[0]
            vec = splits_[1:]
            raw_word2vec[w] = vec

        self.word2vec = []
        oov_cnt = 0

        for v in self.vocab:
            str_vec = raw_word2vec.get(v, None)
            if str_vec is None:
                oov_cnt += 1
                vec = np.random.randn(self.word2vec_dim) * 0.1
            else:
                vec = np.array([float(s_) for s_ in str_vec])
            self.word2vec.append(vec)

        print("word2vec cannot cover %f vocab" % (float(oov_cnt)/len(self.vocab)))

    def get_corpus(self):

        id_train = self.to_id_corpus(self.train_corpus)
        id_valid = self.to_id_corpus(self.valid_corpus)
        id_test = self.to_id_corpus(self.test_corpus)

        return {"train": id_train, "valid": id_valid, "test": id_test, "test_text": self.test_corpus}

    def to_id_corpus(self, data):
        results = []
        for (post, post_e), (response, response_e) in data:
            id_post = [self.rev_vocab.get(t, self.unk_id) for t in post]
            id_post_e = self.rev_emotion_vocab.get(post_e)
            id_response = [self.rev_vocab.get(t, self.unk_id) for t in response]
            id_response_e = self.rev_emotion_vocab.get(response_e)

            results.append(((id_post, id_post_e), (id_response, id_response_e)))

        return results

if __name__ == '__main__':
    corpus = Corpus('../data/nlpcc.json', max_vocab_cnt=30000, word2vec=None,
                    word2vec_dim=400, add_start_tok=False)
    id_corpus = corpus.get_corpus()

    for i in range(20):
        print(corpus.vocab[i])