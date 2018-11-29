# -*- coding:utf-8 -*-
"""
Created on 18/11/27 上午10:25.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
from __future__ import print_function
from __future__ import division
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torch.nn.functional as F
from torch.nn.utils.rnn import pack_padded_sequence, pad_packed_sequence
import constants as constants

from modules.fn_lib import get_bi_rnn_encode, get_bleu_stats, print_loss, get_rnncell
from modules.decoder_fn_lib import train_loop, inference_loop

import numpy as np

from nltk.translate.bleu_score import sentence_bleu, SmoothingFunction

import sys
import time
import re

class Seq2Seq(nn.Module):

    def __init__(self, config, corpus, log_dir=None):
        super(Seq2Seq, self).__init__()

        self.config = config

        self.vocab = corpus.vocab
        self.rev_vocab = corpus.rev_vocab
        self.word_embedding_size = config.embedding_size
        self.n_encoder_layers = config.n_encoder_layers
        self.dropout = config.dropout
        self.max_sequence_length = config.max_seq_len
        self.d_encoder = config.d_encoder
        self.d_model = config.d_model
        self.n_decoder_layers = config.n_decoder_layers
        self.d_decoder = config.d_decoder

        self.full_kl_step = config.full_kl_step
        self.grad_clip = config.grad_clip
        self.grad_noise = config.grad_noise

        self.train_ops = None

        self.word_embedding = nn.Embedding(
            len(self.vocab),
            self.word_embedding_size,
            padding_idx=constants.PAD)

        self.bi_sent_cell = get_rnncell(
            'gru',
            self.word_embedding_size,
            self.d_encoder,
            dropout=self.dropout,
            num_layer=self.n_encoder_layers,
            bidirectional=True)

        decoder_init_size = input_sentence_size = self.d_encoder * 2

        # decoder init state projection
        if self.n_encoder_layers > 1:
            self.dec_init_state_net = nn.ModuleList(
                [nn.Linear(decoder_init_size, self.d_decoder) for i in range(config.n_decoder_layers)])
        else:
            self.dec_init_state_net = nn.Linear(decoder_init_size, self.d_decoder)

        # decoder
        decoder_input_size = self.word_embedding_size # + self.emotion_embedding_size
        self.decoder_cell = get_rnncell(
            'gru',
            decoder_input_size,
            self.d_decoder,
            dropout=0.,
            num_layer=self.n_decoder_layers)

        self.decoder_projection = nn.Linear(
            self.d_decoder,
            len(self.vocab))

        # BOW Loss
        self.bow_projection = nn.Sequential(
            nn.Linear(input_sentence_size, self.d_model),
            nn.Tanh(),
            nn.Linear(self.d_model, len(self.vocab)),
            nn.Sigmoid()
        )

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

        for k,v in feed_dict.items():
            setattr(self, k, v)

        # self.input_sentence -> B x T
        # inp_word_emb -> B x T x emb
        inp_word_emb = self.word_embedding(self.input_sentences)

        # inp_sequence_emb -> B x 2d
        inp_sequence_emb, d_input_embed = get_bi_rnn_encode(
            inp_word_emb,
            self.bi_sent_cell)


        if self.dropout > 0.:
            inp_sequence_emb = F.dropout(inp_sequence_emb, self.dropout, self.training)

        decoder_initializer = inp_sequence_emb

        # self.bow_logits -> B x W
        self.bow_logits = self.bow_projection(inp_sequence_emb)

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
                target_emotion_vector=None,
                decode_type='greedy')
        else:
            # self.output_tokens -> B x T
            # input_tokens -> B x (T-1)
            input_tokens = self.output_tokens[:, :-1]

            if self.dropout > 0.:
                keep_mask = Variable(input_tokens.data.new(input_tokens.size()).bernoulli_(1-self.dropout))
                input_tokens = input_tokens * keep_mask

            decoder_input_embed = self.word_embedding(input_tokens)
            dec_seq_lens = self.output_lens - 1

            decoder_input_embed = F.dropout(decoder_input_embed, self.dropout, self.training)

            dec_outs, _, final_context_state = train_loop(
                self.decoder_cell,
                self.decoder_projection,
                decoder_input_embed,
                init_state=dec_init_state,
                target_emotion_vector=None,
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
            rc_loss = F.cross_entropy(dec_outs.view(-1, dec_outs.size(-1)), labels.reshape(-1), reduction='none').view(dec_outs.size()[:-1])
            rc_loss = torch.sum(rc_loss * label_mask, 1)
            self.avg_rc_loss = rc_loss.mean()
            self.rc_ppl = torch.exp(torch.sum(rc_loss) / torch.sum(label_mask))

            # bow_loss

            bow_loss = F.binary_cross_entropy(self.bow_logits, self.output_bows, reduction='none')
            bow_loss = torch.sum(bow_loss, 1)
            self.avg_bow_loss = torch.mean(bow_loss)


            # calc loss
            self.elbo = self.avg_rc_loss + self.rc_ppl
            self.aug_elbo = self.avg_rc_loss + self.avg_bow_loss
            self.est_marginal = torch.mean(rc_loss + bow_loss)

    '''
    
    def inference_loop2(self, cell, output_fn, embeddings,
                       encoder_state,
                       start_of_sequence_id, end_of_sequence_id,
                       maximum_length, target_emotion_vector,
                       decode_type="beams",
                       is_seq2seq=False):

        batch_size = encoder_state.size(1)

        if decode_type == "beams":
            if not is_seq2seq:
                beam_size = 10
                cur_time = 0
                beam_states = []  # element : (outputs[], last_hidden_state, sum_score)
                results = []

                for idx_ in range(maximum_length):

                    if len(beam_states) == 0:  # invariant that this is time == 0
                        # next_input_id = Variable(
                        #     encoder_state.data.new(batch_size).long().fill_(start_of_sequence_id))  # start with <s>
                        next_input_id_candidates = \
                        torch.sort(start_of_sequence_id, descending=True)[1].data.cpu().numpy()[0][:2]
                        # print("next_input_id_candidates", next_input_id_candidates)
                        if next_input_id_candidates[0] == 1:
                            # next_input_id = next_input_id_candidates[1]
                            next_input_id = Variable(
                                encoder_state.data.new(batch_size).long().fill_(int(next_input_id_candidates[1])))
                        else:
                            # next_input_id = next_input_id_candidates[0]
                            next_input_id = Variable(
                                encoder_state.data.new(batch_size).long().fill_(int(next_input_id_candidates[0])))
                        cell_state = encoder_state

                        next_input = embeddings(next_input_id)

                        if target_emotion_vector is not None:
                            next_input = torch.cat([next_input, target_emotion_vector], 1)

                        cell_output, cell_state = cell(next_input.unsqueeze(0), cell_state)
                        cell_output = output_fn(cell_output.squeeze(1))

                        # TODO 不知道怎么获取到output的概率，这块需要改
                        topk_candidates = torch.topk(F.log_softmax(cell_output), beam_size + 1)
                        topk_candidates_words = topk_candidates[1].data.cpu().numpy()[0]  # 词
                        topk_candidates_scores = topk_candidates[0].data.cpu().numpy()[0]  # 概率

                        if 1 in topk_candidates_words[:beam_size]:
                            tmp_candidates = []
                            tmp_scores = []
                            for iter_, w_idx in enumerate(topk_candidates_words):
                                if w_idx != 1:
                                    tmp_candidates.append(topk_candidates_words[iter_])
                                    tmp_scores.append(topk_candidates_scores[iter_])
                            topk_candidates_words = tmp_candidates
                            topk_candidates_scores = tmp_scores
                        else:
                            topk_candidates_words = topk_candidates_words[:beam_size]
                            topk_candidates_scores = topk_candidates_scores[:beam_size]

                        for word, score in zip(topk_candidates_words, topk_candidates_scores):
                            # print(topk_candidates_words)
                            # print(word)
                            beam_states.append(
                                ([next_input_id.data.cpu().numpy()[0], int(word)], cell_state, score, [cell_output]))

                    else:
                        new_beams = []
                        for prev_words, prev_state, prev_score, prev_output in beam_states:
                            next_input_id = Variable(
                                encoder_state.data.new(batch_size).long().fill_(prev_words[-1]))  # last word

                            next_input = embeddings(next_input_id)

                            if target_emotion_vector is not None:
                                next_input = torch.cat([next_input, target_emotion_vector], 1)

                            cell_output, cell_state = cell(next_input.unsqueeze(0), prev_state)
                            cell_output = output_fn(cell_output.squeeze(1))

                            # TODO 不知道怎么获取到output的概率，这块需要改
                            topk_candidates = torch.topk(F.log_softmax(cell_output), beam_size + 1)
                            topk_candidates_words = topk_candidates[1].data.cpu().numpy()[0]  # 词
                            topk_candidates_scores = topk_candidates[0].data.cpu().numpy()[0]  # 概率

                            if 1 in topk_candidates_words[:beam_size]:
                                tmp_candidates = []
                                tmp_scores = []
                                for iter_, w_idx in enumerate(topk_candidates_words):
                                    if w_idx != 1:
                                        tmp_candidates.append(topk_candidates_words[iter_])
                                        tmp_scores.append(topk_candidates_scores[iter_])
                                topk_candidates_words = tmp_candidates
                                topk_candidates_scores = tmp_scores
                            else:
                                topk_candidates_words = topk_candidates_words[:beam_size]
                                topk_candidates_scores = topk_candidates_scores[:beam_size]

                            for word, score in zip(topk_candidates_words, topk_candidates_scores):
                                new_words = prev_words + [int(word)]  # concat
                                new_score = prev_score + score  # calculate new score

                                if int(word) == end_of_sequence_id:  # reach the boarder
                                    results.append(
                                        (new_words, cell_state, new_score,
                                         prev_output + [cell_output]))  # return (outputs)
                                    if len(results) == beam_size:
                                        break
                                else:
                                    new_beams.append((new_words, cell_state, new_score, prev_output + [cell_output]))

                        beam_states = sorted(new_beams, key=lambda s: s[2], reverse=True)
                        beam_states = beam_states[:beam_size]  # reserve better beams

                        if idx_ == maximum_length - 1:
                            rlen = beam_size - len(results)
                            for ridx_ in range(rlen):
                                results.append(beam_states[ridx_])

                sorted_result = sorted(results, key=lambda s: s[2], reverse=True)[0]
                final_result = (Variable(torch.from_numpy(np.array([sorted_result[0]]))),
                                sorted_result[1],
                                sorted_result[2],
                                torch.cat([_.unsqueeze(1) for _ in sorted_result[3]], 1))

                return final_result[3], final_result[1], final_result[0]

            else:
                beam_size = 10
                cur_time = 0
                beam_states = []
                results = []

                for idx_ in range(maximum_length):

                    if len(beam_states) == 0:
                        # next_input_id = Variable(
                        #     encoder_state.data.new(batch_size).long().fill_(start_of_sequence_id))
                        # ======== TEMP REMOVED ========
                        # next_input_id = torch.max(start_of_sequence_id, 1)[1]
                        # print("start_of_sequence_id", start_of_sequence_id)
                        # print("sorted id", torch.sort(start_of_sequence_id, descending=True)[1][:2])
                        # print("next_input_id", next_input_id)

                        next_input_id_candidates = \
                        torch.sort(start_of_sequence_id, descending=True)[1].data.cpu().numpy()[0][:2]
                        # print("next_input_id_candidates", next_input_id_candidates)
                        if next_input_id_candidates[0] == 1:
                            # next_input_id = next_input_id_candidates[1]
                            next_input_id = Variable(
                                encoder_state.data.new(batch_size).long().fill_(int(next_input_id_candidates[1])))
                        else:
                            # next_input_id = next_input_id_candidates[0]
                            next_input_id = Variable(
                                encoder_state.data.new(batch_size).long().fill_(int(next_input_id_candidates[0])))
                        cell_state = encoder_state

                        next_input = embeddings(next_input_id)

                        if target_emotion_vector is not None:
                            next_input = torch.cat([next_input, target_emotion_vector], 1)

                        cell_output, cell_state = cell(next_input.unsqueeze(0), cell_state)
                        cell_output = output_fn(cell_output.squeeze(1))

                        topk_candidates = torch.topk(F.log_softmax(cell_output), beam_size + 1)
                        topk_candidates_words = topk_candidates[1].data.cpu().numpy()[0]
                        topk_candidates_scores = topk_candidates[0].data.cpu().numpy()[0]

                        if 1 in topk_candidates_words[:beam_size]:
                            tmp_candidates = []
                            tmp_scores = []
                            for iter_, w_idx in enumerate(topk_candidates_words):
                                if w_idx != 1:
                                    tmp_candidates.append(topk_candidates_words[iter_])
                                    tmp_scores.append(topk_candidates_scores[iter_])
                            topk_candidates_words = tmp_candidates
                            topk_candidates_scores = tmp_scores
                        else:
                            topk_candidates_words = topk_candidates_words[:beam_size]
                            topk_candidates_scores = topk_candidates_scores[:beam_size]

                        for word, score in zip(topk_candidates_words, topk_candidates_scores):
                            # beam_states.append(([int(word)], cell_state, score, [cell_output]))
                            beam_states.append(
                                ([next_input_id.data.cpu().numpy()[0], int(word)], cell_state, score, [cell_output]))

                    else:
                        new_beams = []
                        for prev_words, prev_state, prev_score, prev_output in beam_states:
                            next_input_id = Variable(
                                encoder_state.data.new(batch_size).long().fill_(prev_words[-1]))  # last word

                            next_input = embeddings(next_input_id)

                            if target_emotion_vector is not None:
                                next_input = torch.cat([next_input, target_emotion_vector], 1)

                            cell_output, cell_state = cell(next_input.unsqueeze(0), prev_state)
                            cell_output = output_fn(cell_output.squeeze(1))

                            # TODO 不知道怎么获取到output的概率，这块需要改
                            topk_candidates = torch.topk(F.log_softmax(cell_output), beam_size + 1)
                            topk_candidates_words = topk_candidates[1].data.cpu().numpy()[0]  # 词
                            topk_candidates_scores = topk_candidates[0].data.cpu().numpy()[0]  # 概率

                            if 1 in topk_candidates_words[:beam_size]:
                                tmp_candidates = []
                                tmp_scores = []
                                for iter_, w_idx in enumerate(topk_candidates_words):
                                    if w_idx != 1:
                                        tmp_candidates.append(topk_candidates_words[iter_])
                                        tmp_scores.append(topk_candidates_scores[iter_])
                                topk_candidates_words = tmp_candidates
                                topk_candidates_scores = tmp_scores
                            else:
                                topk_candidates_words = topk_candidates_words[:beam_size]
                                topk_candidates_scores = topk_candidates_scores[:beam_size]

                            for word, score in zip(topk_candidates_words, topk_candidates_scores):
                                new_words = prev_words + [int(word)]  # concat
                                new_score = prev_score + score  # calculate new score

                                if int(word) == end_of_sequence_id:  # reach the boarder
                                    results.append(
                                        (new_words, cell_state, new_score,
                                         prev_output + [cell_output]))  # return (outputs)
                                    if len(results) == beam_size:
                                        break
                                else:
                                    new_beams.append((new_words, cell_state, new_score, prev_output + [cell_output]))

                        beam_states = sorted(new_beams, key=lambda s: s[2], reverse=True)
                        beam_states = beam_states[:beam_size]  # reserve better beams

                        if idx_ == maximum_length - 1:
                            rlen = beam_size - len(results)
                            for ridx_ in range(rlen):
                                results.append(beam_states[ridx_])

                sorted_result = sorted(results, key=lambda s: s[2], reverse=True)
                final_text, final_state, final_word = [], [], []
                for res in sorted_result:
                    final_result = (Variable(torch.from_numpy(np.array([res[0]]))),
                                    res[1],
                                    res[2],
                                    torch.cat([_.unsqueeze(1) for _ in res[3]], 1))
                    final_text.append(final_result[3])
                    final_state.append(final_result[1])
                    final_word.append(final_result[0])

                return final_text, final_state, final_word

        else:
            outputs = []
            context_state = []
            cell_state, cell_input, cell_output = encoder_state, None, None

            for time in range(maximum_length + 1):
                if cell_output is None:
                    # invariant that this is time == 0
                    next_input_id = Variable(encoder_state.data.new(batch_size).long().fill_(start_of_sequence_id))
                    # done: indicate which sentences reaches eos. used for early stopping
                    done = encoder_state.data.new(batch_size).zero_().byte()
                    cell_state = encoder_state
                else:
                    cell_output = output_fn(cell_output)
                    outputs.append(cell_output)

                    # print("CELL_OUTPUT_SIZE", cell_output.size)

                    if decode_type == 'sample':
                        matrix_U = -1.0 * torch.log(
                            -1.0 * torch.log(cell_output.data.new(cell_output.size()).uniform_()))
                        # next_input_id = torch.max(cell_output.data - matrix_U, 1)[1]
                        next_input_ids = torch.topk(cell_output.data - matrix_U, k=2, dim=1)[1]
                        next_input_id = []
                        for _id in next_input_ids[0]:
                            candidate_ids = _id.cpu().numpy()
                            if candidate_ids[0] == 1:  # if c_id == unk_id
                                next_input_id.append(candidate_ids[1])
                            else:
                                next_input_id.append(candidate_ids[0])
                        next_input_id = Variable(torch.from_numpy(np.array(next_input_id, dtype=np.int64)))
                    elif decode_type == 'greedy':
                        next_input_ids = torch.topk(cell_output, 2, 2)[1]
                        next_input_id = []
                        for _id in next_input_ids[0]:

                            candidate_ids = _id.data.cpu().numpy()
                            if candidate_ids[0] == 1:  # if c_id == unk_id
                                next_input_id.append(candidate_ids[1])
                            else:
                                next_input_id.append(candidate_ids[0])
                        next_input_id = Variable(torch.from_numpy(np.array(next_input_id, dtype=np.int64)))
                    else:
                        raise ValueError("unknown decode type")

                    if torch.cuda.is_available():
                        next_input_id = next_input_id.squeeze(0).cuda()
                    else:
                        next_input_id = next_input_id.squeeze(0)

                    next_input_id = next_input_id * Variable(
                        (~done).long())  # make sure the next_input_id to be 0 if done
                    done = (next_input_id == end_of_sequence_id).data | done

                    # save the decoding results into context state
                    context_state.append(next_input_id)

                next_input = embeddings(next_input_id)
                if target_emotion_vector is not None:
                    next_input = torch.cat([next_input, target_emotion_vector], 1)
                if done.long().sum() == batch_size:
                    break

                cell_output, cell_state = cell(next_input.unsqueeze(0), cell_state)

                # Squeeze the time dimension
                cell_output = cell_output.squeeze(1)

                # zero out done sequences
                cell_output = cell_output * Variable((~done).float()).unsqueeze(1)

            return torch.cat([_.unsqueeze(1) for _ in outputs], 1), cell_state, torch.cat(
                [_.unsqueeze(1) for _ in context_state], 1)

    def inference_loop(self,
                       cell,
                       output_fn,
                       embeddings,
                       encoder_state,
                       start_of_sequence_id,
                       end_of_sequence_id,
                       maximum_length,
                       target_emotion_vector,
                       decode_type='samples',
                       is_seq2seq=False):

        maximum_length_int = maximum_length + 1
        batch_size = encoder_state.size(1)

        time = 0
        outputs = []
        context_state = []
        cell_state, cell_input, cell_output = encoder_state, None, None

        for time in range(maximum_length + 1):
            if cell_output is None:
                next_input_id = encoder_state.new_full((batch_size,), start_of_sequence_id, dtype=torch.long)
                done = encoder_state.new_zeros(batch_size, dtype=torch.uint8)
                cell_state = encoder_state
            else:
                cell_output = output_fn(cell_output)
                outputs.append(cell_output)

                if decode_type == 'sample':
                    matrix_U = -1.0 * torch.log(
                        -1.0 * torch.log(cell_output.new_empty(cell_output.size()).uniform_()))
                    next_input_id = torch.max(cell_output - matrix_U, 1)[1]
                elif decode_type == 'greedy':
                    next_input_id = torch.max(cell_output, 1)[1]
                else:
                    raise ValueError("unknown decode type")

                next_input_id = next_input_id * (~done).long()  # make sure the next_input_id to be 0 if done
                done = (next_input_id == end_of_sequence_id) | done
                # save the decoding results into context state
                context_state.append(next_input_id)

            next_input = embeddings(next_input_id)
            if target_emotion_vector is not None:
                next_input = torch.cat([next_input, target_emotion_vector], 1)
            if done.long().sum() == batch_size:
                break

            cell_output, cell_state = cell(next_input.unsqueeze(1), cell_state)
            # Squeeze the time dimension
            cell_output = cell_output.squeeze(1)

            # zero out done sequences
            cell_output = cell_output * (~done).float().unsqueeze(1)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), cell_state, torch.cat(
            [_.unsqueeze(1) for _ in context_state], 1)
    '''

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
                                                            0.0
            self.optimize(self.aug_elbo)

            elbo_losses.append(elbo_loss)
            bow_losses.append(bow_loss)
            rc_ppls.append(rc_ppl)
            rc_losses.append(rc_loss)
            kl_losses.append(kl_loss)

            global_t += 1
            local_t += 1
            if local_t % (min(train_feed.num_batch, update_limit) // 10) == 0:
                # kl_w = self.kl_w
                print_loss("%.1f%%-%.2f%%" % ((local_t / min(train_feed.num_batch, update_limit) * 100),
                                                     (train_feed.ptr / train_feed.num_batch * 100)),
                           loss_names, [elbo_losses, bow_losses, rc_losses, rc_ppls, kl_losses],
                           "")

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
                                                            0.0

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
            sample_words = np.split(word_outs, repeat, axis=0)

            true_srcs = feed_dict['input_sentences'].cpu().numpy()
            true_srcs_lens = feed_dict['input_sentence_lens'].cpu().numpy()
            true_outs = feed_dict['output_tokens'].cpu().numpy()
            local_t += 1

            if dest != sys.stdout:
                if local_t % (test_feed.num_batch // 10) == 0:
                    print("Testing ............. %.2f%%" % (test_feed.ptr / float(test_feed.num_batch) * 100))

            for b_id in range(test_feed.batch_size):
                dest.write("Batch %d index %d\n" % (local_t, b_id))
                src_str = " ".join([self.vocab[e] for e in true_srcs[b_id].tolist() if e != 0])
                dest.write("Src %d: %s\n" % (b_id, src_str))

                true_tokens = [self.vocab[e] for e in true_outs[b_id].tolist() if
                               e not in [0, constants.BOS, constants.EOS]]
                true_str = " ".join(true_tokens)
                dest.write("Target >> %s\n" % (true_str))

                local_tokens = []
                for r_id in range(repeat):
                    pred_outs = sample_words[r_id]
                    pred_tokens = [self.vocab[e] for e in pred_outs[b_id].tolist()]
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

        input_sentences, input_sentence_lens, _, output_tokens, output_lens, _, output_bows = batch

        feed_dict = {"input_sentences": input_sentences,
                     "input_sentence_lens": input_sentence_lens,
                     "output_tokens": output_tokens,
                     "output_lens": output_lens,
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
