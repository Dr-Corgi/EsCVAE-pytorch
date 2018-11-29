# -*- coding:utf-8 -*-
"""
Created on 18/11/27 下午12:12.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
# -*- coding: utf8 -*-
import torch

class Config(object):

    # Global Setting
    update_limit = 3000
    embedding_size = 200
    cell_type = "gru"
    vocab_size = 20000

    # Encoder
    # sent_type = "bi-rnn"
    d_encoder = 300
    max_seq_len = 40
    n_encoder_layers = 4
    dropout = 0.2

    # EsCVAE
    d_model = 400
    full_kl_step = 10000
    min_prior_size = 100
    emotion_embedding_size = 30

    # Decoder
    # dec_dropout = 0.
    d_decoder = 300
    n_decoder_layers = 4
    decode_type = "greedy"

    # Training
    op = "adam"
    grad_clip = 5.0
    init_w = 0.08
    batch_size = 32
    init_lr = 0.001
    lr_hold = 1
    lr_decay = 0.6
    improve_threshold = 0.998
    patience = 20
    patient_increase = 3.0
    early_stop = True
    max_epoch = 200
    grad_noise = 0.

    # GPU
    use_cuda = torch.cuda.is_available()


"""
FLAGS(object):
    面向语料的配置参数。具体如下：
    - word2vec_path: 词向量位置。不使用时可设置为None。默认为"./word2vec/wiki.zh.vector"。
    - data_path: 训练语料路径。
    - work_dir: 模型存储路径。
    - resume: 从训练一半的模型中恢复。
    - forward_only: 只运行不训练
    - save_model: 是否保存模型。
    - test_path: 模型路径。
"""
class FLAGS(object):
    word2vec_path = None
    data_path = ""
    work_dir = "./runs/"
    resume = False
    forward_only = False
    save_model = True
    test_path = ""