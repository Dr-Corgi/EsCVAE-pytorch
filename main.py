# -*- coding:utf-8 -*-
"""
Created on 18/11/27 下午12:11.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
# -*- coding: utf8 -*-
import codecs
import glob
import os
import time

import numpy as np
import torch
from beeprint import pp

from utils import Config, FLAGS
from corpus.corpus import Corpus
from corpus.data_loader import DataLoader

from seq2seq import Seq2Seq
from escvae import EsCVAE

flags = FLAGS()

"""
从保存的模型中恢复。
"""


def get_checkpoint_state(ckpt_dir):
    files = os.path.join(ckpt_dir, "*.pth")
    files = glob.glob(files)
    files.sort(key=os.path.getmtime)
    return len(files) > 0 and files[-1] or None


def main():
    # 设置训练、校验、测试阶段的配置参数
    config = Config()

    valid_config = Config()
    valid_config.dropout = valid_config.dec_dropout = 0.
    valid_config.batch_size = 60

    test_config = Config()
    test_config.dropout = test_config.dec_dropout = 0.
    test_config.batch_size = 1

    # 语料处理，包括读取语料、ID化
    corpus = Corpus(flags.data_path, max_vocab_cnt=config.vocab_size, word2vec=flags.word2vec_path,
                    word2vec_dim=config.embedding_size)
    id_corpus = corpus.get_corpus()

    train_data, valid_data, test_data = id_corpus.get("train"), id_corpus.get("valid"), id_corpus.get("test")
    # test_text = id_corpus.get("test_text")

    # convert to numeric input outputs that fits into models
    train_feed = DataLoader("Train", train_data, config)
    valid_feed = DataLoader("Valid", valid_data, valid_config)
    test_feed = DataLoader("Test", test_data, test_config)

    # 从训练过的模型中读取
    if flags.forward_only or flags.resume:
        log_dir = os.path.join(flags.work_dir, flags.test_path)
    else:
        log_dir = os.path.join(flags.work_dir, "run" + str(int(time.time())))
    if not os.path.exists(log_dir):
        os.mkdir(log_dir)

    # 构建模型
    model = EsCVAE(config, corpus, log_dir=None if flags.forward_only else log_dir)

    print("\nOutput Configs")

    # 将模型参数写入日志
    if not flags.forward_only:
        with open(os.path.join(log_dir, "run.log"), 'w') as f:
            f.write(pp(config, output=False))
            f.write("======== Corpus ========")
            f.write(pp(flags, output=False))

    # 确认当前路径有无checkpoints，如果没有则强制增加路径
    ckpt_dir = os.path.join(log_dir, "checkpoints")
    if not os.path.exists(ckpt_dir):
        os.mkdir(ckpt_dir)

    # 从模型保存路径中读取已经训练的模型，如无CKPT则重新初始化模型
    # 如使用预训练的w2v则载入数据，否则使用0初始化
    ckpt = get_checkpoint_state(ckpt_dir)
    if ckpt:
        print("Reading dm models parameters from %s" % ckpt)
        model.load_state_dict(torch.load(ckpt))
    else:
        print("Create models with fresh parameter")
        model.apply(
            lambda m: [torch.nn.init.uniform_(p.data, -1.0 * config.init_w, config.init_w) for p in m.parameters()])

        if corpus.word2vec is not None and not flags.forward_only:
            print("load word2vec")
            model.word_embedding.weight.data.copy_(torch.from_numpy(np.array(corpus.word2vec)))
        else:
            model.word_embedding.weight.data[0].fill_(0)

    if config.use_cuda:
        model.cuda()

    """
    训练模型。具体流程如下：
        - 定义模型保存路径checkpoint_path
        - 定义全局训练轮次global_t（似乎用处不大）
        - 定义初始耐心patience
        - 定义校验的LOSS阈值和最佳校验LOSS

        - 遍历最多max_epoch次的训练：
            如果当前feed还没初始化或者快跑完了，重新初始化feed；
            使用train_model函数训练模型；
            使用valid_model校验模型，使用test_model测试模型（5个）;
            如果校验LOSS低于最优校验LOSS，且校验LOSS低于原有最佳校验LOSS乘上阈值，则更新耐心值和校验LOSS阈值
            如果校验LOSS低于最优校验LOSS，保存模型；
            如果超出耐心，提前结束；    

    """
    if not flags.forward_only:
        # if False:
        checkpoint_path = os.path.join(ckpt_dir, model.__class__.__name__ + "-%d.pth")
        global_t = 1
        patience = config.patience
        dev_loss_threshold = np.inf
        best_dev_loss = np.inf

        for epoch in range(config.max_epoch):
        # for epoch in range(1):
            print(">> Epoch %d with lr %f" % (epoch, model.learning_rate))
            # begin training
            if train_feed.num_batch is None or train_feed.ptr >= train_feed.num_batch:
                train_feed.epoch_init(config.batch_size, shuffle=True)

            global_t, train_loss = model.train_model(global_t, train_feed, update_limit=config.update_limit)

            # valid model
            valid_feed.epoch_init(valid_config.batch_size, shuffle=False, intra_shuffle=False)
            model.eval()
            valid_loss = model.valid_model("ELBO_VALID", valid_feed)

            # test model
            test_feed.epoch_init(test_config.batch_size, shuffle=True, intra_shuffle=False)
            model.test_model(test_feed, num_batch=5)

            model.train()

            done_epoch = epoch + 1

            if config.op == "sgd" and done_epoch > config.lr_hold:
                model.learning_rate_decay()

            if valid_loss < best_dev_loss:
                if valid_loss <= dev_loss_threshold * config.improve_threshold:
                    patience = max(patience, done_epoch * config.patient_increase)
                    dev_loss_threshold = valid_loss

                if flags.save_model:
                    print("Save model!")
                    torch.save(model.state_dict(), checkpoint_path % (epoch))

                best_dev_loss = valid_loss

            if config.early_stop and patience <= done_epoch:
                print("Early stop due to run out of patience!")
                break

        print("Best validation loss %f" % best_dev_loss)
        print("Done training")

        print("Save model!")
        torch.save(model.state_dict(), checkpoint_path % (epoch))

    """
    校验模型，测试模型。
    """

    valid_feed.epoch_init(valid_config.batch_size, shuffle=False, intra_shuffle=False)

    model.eval()
    model.valid_model("ELBO_VALID", valid_feed)

    test_feed.epoch_init(test_config.batch_size, shuffle=False, intra_shuffle=False)
    model.valid_model("ELBO_TEST", test_feed)

    model.train()


if __name__ == "__main__":

    if flags.forward_only:
        if flags.test_path is None:
            print("Set test_path before forward only")
            exit(1)

    main()
