# -*- coding:utf-8 -*-
"""
Created on 18/11/28 下午2:43.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
import constants as constants

from modules.fn_lib import get_bi_rnn_encode, get_bleu_stats, print_loss, get_rnncell, sample_gaussian, gaussian_kld, norm_log_liklihood
from modules.decoder_fn_lib import train_loop, inference_loop

import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import sys
import time
import re


class EsCVAE(nn.Module):
    def __init__(self, config, corpus, log_dir=None):
        super(EsCVAE, self).__init__()

        self.config = config

        self.idx2word = corpus.vocab
        self.word2idx = corpus.rev_vocab
        self.emotion2idx = corpus.emotion_vocab
        self.idx2emotion = corpus.emotion_show_vocab

        self.word_embedding_size = config.embedding_size
        self.emotion_embedding_size = config.emotion_embedding_size

        self.n_encoder_layers = config.n_encoder_layers
        self.dropout = config.dropout
        self.max_sequence_length = config.max_seq_len
        self.d_encoder = config.d_encoder
        self.d_model = config.d_model
        self.n_decoder_layers = config.n_decoder_layers
        self.d_decoder = config.d_decoder
        self.decode_type = config.decode_type

        self.full_kl_step = config.full_kl_step
        self.grad_clip = config.grad_clip
        self.grad_noise = config.grad_noise

        self.train_ops = None

        self.word_embedding = nn.Embedding(
            len(self.idx2word),
            self.word_embedding_size,
            padding_idx=constants.PAD)

        self.emotion_embedding = nn.Embedding(
            len(self.idx2emotion),
            self.emotion_embedding_size)

        self.bi_sent_cell = get_rnncell(
            'gru',
            self.word_embedding_size,
            self.d_encoder,
            dropout=self.dropout,
            num_layer=self.n_encoder_layers,
            bidirectional=True)

        input_sentence_size = output_embedding_size = self.d_encoder * 2

        cond_embedding_size = input_sentence_size + self.emotion_embedding_size

        # recognitionNetwork
        recog_input_size = cond_embedding_size + output_embedding_size + self.emotion_embedding_size
        self.recogNet_mulogvar = nn.Linear(recog_input_size, self.d_model * 2)

        # priorNetwork
        prior_hidden_size = int(np.maximum(self.d_model*2, 300))
        self.priorNet_mulogvar = nn.Sequential(
            nn.Linear(cond_embedding_size, prior_hidden_size),
            nn.Tanh(),
            nn.Linear(prior_hidden_size, self.d_model * 2))

        decoder_init_size = cond_embedding_size + self.d_model

        # decoder init state projection
        if self.n_encoder_layers > 1:
            self.dec_init_state_net = nn.ModuleList(
                [nn.Linear(decoder_init_size, self.d_decoder) for i in range(config.n_decoder_layers)])
        else:
            self.dec_init_state_net = nn.Linear(decoder_init_size, self.d_decoder)

        # decoder
        decoder_input_size = self.word_embedding_size + self.emotion_embedding_size
        self.decoder_cell = get_rnncell(
            'gru',
            decoder_input_size,
            self.d_decoder,
            dropout=0.,
            num_layer=self.n_decoder_layers)

        self.decoder_projection = nn.Linear(
            self.d_decoder,
            len(self.idx2word))

        # BOW Loss
        self.bow_projection = nn.Sequential(
            nn.Linear(decoder_init_size, self.d_model),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, len(self.idx2word)),
            nn.Sigmoid())

        # Y Loss
        self.emotion_projection = nn.Sequential(
            nn.Linear(decoder_init_size, self.d_model),
            nn.Tanh(),
            nn.Dropout(self.dropout),
            nn.Linear(self.d_model, len(self.idx2emotion)))

        # optimizer
        tvars = self.parameters()
        if config.op == 'adam':
            print('Use Adam')
            optimizer = optim.Adam(tvars)
            self.learning_rate = 1e-3
        elif config.op == 'rmsprop':
            print('Use RMSProp')
            optimizer = optim.RMSprop(tvars)
            self.learning_rate = 1e-2
        else:
            print('Use SGD')
            optimizer = optim.SGD(tvars, lr=0.5)
            self.learning_rate = 0.5
        self.train_ops = optimizer

    def forward(self, feed_dict, mode='train'):

        for k, v in feed_dict.items():
            setattr(self, k, v)

        # self.input_sentence -> B x T
        # inp_word_emb -> B x T x emb
        inp_word_emb = self.word_embedding(self.input_sentences)
        inp_emotion_emb = self.emotion_embedding(self.input_emotions)

        outp_word_emb = self.word_embedding(self.output_tokens)
        outp_emotion_emb = self.emotion_embedding(self.output_emotions)

        # inp_sequence_emb -> B x 2d
        inp_sequence_emb, d_input_embed = get_bi_rnn_encode(
            inp_word_emb,
            self.bi_sent_cell)

        outp_sequence_emb, _ = get_bi_rnn_encode(
            outp_word_emb,
            self.bi_sent_cell,
            self.output_lens)

        if self.dropout > 0.:
            inp_sequence_emb = F.dropout(inp_sequence_emb, self.dropout, self.training)

        cond_embedding = torch.cat([inp_sequence_emb, inp_emotion_emb], 1)

        # recognitionNetwork
        recog_input = torch.cat([cond_embedding, outp_sequence_emb, outp_emotion_emb], 1)
        self.recog_mulogvar = recog_mulogvar = self.recogNet_mulogvar(recog_input)
        recog_mu, recog_logvar = torch.chunk(recog_mulogvar, 2 , 1)

        # priorNetwork
        prior_mulogvar = self.priorNet_mulogvar(cond_embedding)
        prior_mu, prior_logvar = torch.chunk(prior_mulogvar, 2, 1)

        # use sampled Z of posterior Z
        if self.use_prior:
            latent_sample = sample_gaussian(prior_mu, prior_logvar)
        else:
            latent_sample = sample_gaussian(recog_mu, recog_logvar)

        decoder_initializer = torch.cat([cond_embedding, latent_sample], 1)

        # predict bow
        # self.bow_logits -> B x W
        self.bow_logits = self.bow_projection(decoder_initializer)

        # predict emotion
        self.emotion_logits = self.emotion_projection(decoder_initializer)
        emotion_prob = F.softmax(self.emotion_logits, dim=-1)
        pred_emotion_emb = torch.matmul(emotion_prob, self.emotion_embedding.weight)
        if mode == 'train':
            selected_emotion_emb = outp_emotion_emb
        else:
            selected_emotion_emb = pred_emotion_emb

        # decoder init
        # dec_init_state -> Layer x B x d
        if self.n_decoder_layers > 1:
            dec_init_state = [self.dec_init_state_net[i](decoder_initializer) for i in range(self.n_decoder_layers)]
            dec_init_state = torch.stack(dec_init_state)
        else:
            dec_init_state = self.dec_init_state_net(decoder_initializer).unsqueeze(0)

        if mode == 'test':
            dec_outs, _, final_context_state = inference_loop(
                self.decoder_cell,
                self.decoder_projection,
                self.word_embedding,
                encoder_state=dec_init_state,
                start_of_sequence_id=constants.BOS,
                end_of_sequence_id=constants.EOS,
                maximum_length=self.max_sequence_length,
                target_emotion_vector=selected_emotion_emb,
                decode_type=self.decode_type)
        else:
            # self.output_tokens -> B x T
            # input_tokens -> B x (T-1)
            input_tokens = self.output_tokens[:, :-1]

            if self.dropout > 0.:
                keep_mask = Variable(input_tokens.data.new(input_tokens.size()).bernoulli_(1 - self.dropout))
                input_tokens = input_tokens * keep_mask

            decoder_input_embed = self.word_embedding(input_tokens)
            dec_seq_lens = self.output_lens - 1

            decoder_input_embed = F.dropout(decoder_input_embed, self.dropout, self.training)

            dec_outs, _, final_context_state = train_loop(
                self.decoder_cell,
                self.decoder_projection,
                decoder_input_embed,
                init_state=dec_init_state,
                target_emotion_vector=selected_emotion_emb,
                sequence_length=dec_seq_lens)

        if final_context_state is not None:
            self.dec_out_words = final_context_state
        else:
            self.dec_out_words = torch.max(dec_outs, 2)[1]

        # ========= Loss ==========
        if not mode == 'test':
            labels = self.output_tokens[:, 1:]
            label_mask = torch.sign(labels).detach().float()

            # rc_loss
            rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)), labels.reshape(-1), reduction='none').view(
                dec_outs.size()[:-1])
            rc_loss = torch.sum(rc_loss * label_mask, 1)
            self.avg_rc_loss = rc_loss.mean()
            self.rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

            # bow_loss
            bow_loss = F.binary_cross_entropy(self.bow_logits, self.output_bows, reduction='none')
            bow_loss = torch.sum(bow_loss, 1)
            self.avg_bow_loss = torch.mean(bow_loss)

            # meta info loss
            self.avg_emotion_loss = F.cross_entropy(self.emotion_logits, self.output_emotions)

            # kl loss
            kld = gaussian_kld(recog_mu, recog_logvar, prior_mu, prior_logvar)
            self.avg_kld = torch.mean(kld)
            if mode == 'train':
                kl_weights = min(self.global_t / self.full_kl_step, 1.0)
            else:
                kl_weights = 1.0

            self.kl_w = kl_weights
            # calc loss
            self.elbo = self.avg_rc_loss + kl_weights * self.avg_kld
            self.aug_elbo = self.avg_bow_loss + self.avg_emotion_loss + self.elbo

            self.log_p_z = norm_log_liklihood(latent_sample, prior_mu, prior_logvar)
            self.log_q_z_xy = norm_log_liklihood(latent_sample, recog_mu, recog_logvar)
            self.est_marginal = torch.mean(rc_loss + bow_loss - self.log_p_z + self.log_q_z_xy)

    def train_model(self, global_t, train_feed, update_limit=5000):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        kl_losses = []
        bow_losses = []
        local_t = 0
        start_time = time.time()
        loss_names = ['elbo_loss', 'bow_loss', 'rc_loss', 'rc_peplexity', 'kl_loss']
        while True:
            batch = train_feed.next_batch()
            if batch is None:
                break
            if update_limit is not None and local_t >= update_limit:
                break
            feed_dict = self.batch_2_feed(batch, global_t, use_prior=False)
            self.forward(feed_dict, mode='train')
            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = self.elbo.item(), \
                                                            self.avg_bow_loss.item(), \
                                                            self.avg_rc_loss.item(), \
                                                            self.rc_ppl.item(), \
                                                            self.avg_kld.item()
            self.optimize(self.aug_elbo)

            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            global_t += 1
            local_t += 1
            if local_t % (min(train_feed.num_batch, update_limit) // 10) == 0:
                kl_w = self.kl_w
                print_loss("%.1f%%-%.2f%%" % ((local_t / min(train_feed.num_batch, update_limit) * 100),
                                              (train_feed.ptr / train_feed.num_batch * 100)),
                           loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                           "kl_w %f" % kl_w)

        if torch.cuda.is_available():
            torch.cuda.synchronize()
        epoch_time = time.time() - start_time
        avg_losses = print_loss("Epoch Done ", loss_names,
                                [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                "step time %.4f" % (epoch_time / train_feed.num_batch))

        return global_t, avg_losses[0]

    def valid_model(self, name, valid_feed):
        elbo_losses = []
        rc_losses = []
        rc_ppls = []
        bow_losses = []
        kl_losses = []
        loss_names = ["elbo_loss", "bow_loss", "rc_loss", "rc_perplexity", "kl_loss"]

        while True:
            batch = valid_feed.next_batch()
            if batch is None:
                break
            feed_dict = self.batch_2_feed(batch, None, use_prior=False, repeat=1)
            with torch.no_grad():
                self.forward(feed_dict, mode='valid')
            elbo_loss, bow_loss, rc_loss, rc_ppl, kl_loss = self.elbo.item(), \
                                                            self.avg_bow_loss.item(), \
                                                            self.avg_rc_loss.item(), \
                                                            self.rc_ppl.item(), \
                                                            self.avg_kld.item()

            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_losses.append(rc_loss)
            rc_ppls.append(rc_ppl)
            kl_losses.append(kl_loss)

        avg_losses = print_loss(name, loss_names,
                                [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                                "")

        return avg_losses[0]

    def test_model(self, test_feed, num_batch=None, repeat=1, dest=sys.stdout):
        local_t = 0
        recall_bleus = []
        prec_bleus = []

        while True:
            batch = test_feed.next_batch()
            if batch is None or (num_batch is not None and local_t > num_batch):
                break

            feed_dict = self.batch_2_feed(batch, None, use_prior=True, repeat=repeat)
            with torch.no_grad():
                self.forward(feed_dict=feed_dict, mode='test')
            word_outs = self.dec_out_words.cpu().numpy()
            print("WORD_OUTS:", word_outs)
            sample_words = np.split(word_outs, repeat, axis=0)

            true_srcs = feed_dict['input_sentences'].cpu().numpy()
            true_srcs_lens = feed_dict['input_sentence_lens'].cpu().numpy()
            true_outs = feed_dict['output_tokens'].cpu().numpy()
            local_t += 1

            # print("TRUE_SRC:", true_srcs)
            # print("TRUE_OUT:", true_outs)

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch // 10) == 0:
                    print("Testing ............. %.2f%%" % (test_feed.ptr / float(test_feed.num_batch) * 100))

            for b_id in range(test_feed.batch_size):
                dest.write("Batch %d index %d\n" % (local_t, b_id))
                src_str = " ".join([self.idx2word[e] for e in true_srcs[b_id].tolist() if e != 0])
                dest.write("Src %d: %s\n" % (b_id, src_str))

                true_tokens = [self.idx2word[e] for e in true_outs[b_id].tolist() if
                               e not in [0, constants.BOS, constants.EOS]]
                true_str = " ".join(true_tokens)
                # emotion_str = self.emotion_labels[true_outp_emotions[b_id]]
                dest.write("Target >> %s\n" % (true_str))

                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    # pred_emotion = np.argmax(sample_emotions[r_id], axis=1)[0]
                    pred_tokens = [self.idx2word[e] for e in pred_outs[b_id].tolist()]
                    pred_str = " ".join(pred_tokens).replace(" ' ", "'")
                    dest.write("Sample %d >> %s\n" % (r_id, pred_str))
                    local_tokens.append(pred_tokens)

                max_bleu, avg_bleu = get_bleu_stats(true_tokens, local_tokens)
                recall_bleus.append(max_bleu)
                prec_bleus.append(avg_bleu)

                dest.write("\n")

        avg_recall_bleu = float(np.mean(recall_bleus))
        avg_prec_bleu = float(np.mean(prec_bleus))
        avg_f1 = 2 * (avg_prec_bleu * avg_recall_bleu) / (avg_prec_bleu + avg_recall_bleu + 10e-12)
        report = "Avg recall BLEU %f, avg precision BLEU %f and F1 %f (only 1 reference response. Not final result)" \
                 % (avg_recall_bleu, avg_prec_bleu, avg_f1)
        print(report)
        dest.write(report + "\n")

        print("Done testing")

    def batch_2_feed(self, batch, global_t, use_prior, repeat=1):

        input_sentences, input_sentence_lens, input_emotions, output_tokens, output_lens, output_emotions, output_bows = batch

        feed_dict = {"input_sentences": input_sentences,
                     "input_sentence_lens": input_sentence_lens,
                     "input_emotions": input_emotions,
                     "output_tokens": output_tokens,
                     "output_lens": output_lens,
                     "output_emotions": output_emotions,
                     "output_bows": output_bows,
                     "use_prior": use_prior}

        if repeat > 1:
            tiled_feed_dict = {}
            for key, val in feed_dict.items():
                if key == "use_prior":
                    tiled_feed_dict[key] = val
                    continue
                multipliers = [1] * len(val.shape)
                multipliers[0] = repeat
                tiled_feed_dict[key] = np.tile(val, multipliers)
            feed_dict = tiled_feed_dict

        if global_t is not None:
            feed_dict["global_t"] = global_t

        if torch.cuda.is_available():
            feed_dict = {k: Variable(torch.from_numpy(v).cuda()) if isinstance(v, np.ndarray) else v
                         for k, v in feed_dict.items()}
        else:
            feed_dict = {k: Variable(torch.from_numpy(v)) if isinstance(v, np.ndarray) else v for
                         k, v in feed_dict.items()}

        return feed_dict

    def optimize(self, loss):
        self.train_ops.zero_grad()
        loss.backward()

        if self.grad_clip is not None:
            torch.nn.utils.clip_grad_norm_(self.parameters(), self.grad_clip)

        if self.grad_noise > 0:
            grad_std = (self.grad_noise / (1.0 + self.global_t) ** 0.55) ** 0.5
            for name, param in self.parameters():
                param.grad.data.add_(torch.truncated_normal(param.shape, mean=0.0, stddev=grad_std))

        self.train_ops.step()