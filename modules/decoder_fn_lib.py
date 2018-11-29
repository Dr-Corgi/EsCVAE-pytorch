# -*- coding:utf-8 -*-
"""
Created on 18/11/27 下午5:26.

Author: Ruizhang1993 (zhang1rui4@foxmail.com)
"""
from __future__ import division
import torch
import torch.nn
from modules.fn_lib import dynamic_rnn

def inference_loop(cell,
                   output_fn,
                   embeddings,
                   encoder_state,
                   start_of_sequence_id,
                   end_of_sequence_id,
                   maximum_length,
                   target_emotion_vector,
                   decode_type='greedy',
                   beams_size=3):

    '''
    cell  ->  self.decoder_cell  ->  gru_cell
    output_fn  ->  self.decoder_projection  ->
    embedding  ->  self.word_embedding  ->
    encoder_state  ->  dec_init_state  ->  Layer x B x d_decoder
    start_od_sequence_id -> int
    end_of_sequence_id -> int
    maximum_length -> int
    '''

    batch_size = encoder_state.size(1)

    if decode_type == 'beams':

        raise NotImplementedError("Beam Search not implemented!")

    else:
        outputs = []
        context_state = []
        # cell_state -> L x B x d_decoder
        cell_state, cell_input, cell_output = encoder_state, None, None

        for time in range(maximum_length + 1):
            if cell_output is None:
                next_input_id = encoder_state.new_full((batch_size,), start_of_sequence_id, dtype=torch.long)
                done = encoder_state.new_zeros(batch_size, dtype=torch.uint8)
                cell_state = encoder_state
                # context_state.append(next_input_id)
            else:
                # cell_output -> B x emb
                cell_output = output_fn(cell_output)
                # cell_output -> B x W
                outputs.append(cell_output)

                if decode_type == 'sample':
                    matrix_U = -1.0 * torch.log(
                        -1.0 * torch.log(cell_output.new_empty(cell_output.size()).uniform_()))
                    next_input_id = torch.max(cell_output - matrix_U, 1)[1]
                elif decode_type == 'greedy':
                    next_input_id = torch.max(cell_output, 1)[1]
                else:
                    raise ValueError('unknown decode type')

                next_input_id = next_input_id * (~done).long()
                done = (next_input_id == end_of_sequence_id) | done
                context_state.append(next_input_id)

            next_input = embeddings(next_input_id)
            if target_emotion_vector is not None:
                next_input = torch.cat([next_input, target_emotion_vector], 1)
            if done.long().sum() == batch_size:
                break

            # next_input -> B x emb
            # next_input.unsqueeze(1) -> B x 1 x emb
            # cell_output -> B x 1 x emb
            # cell_state -> L x B x d_decoder
            cell_output, cell_state = cell(next_input.unsqueeze(1), cell_state)


            # cell_output.squeeze(1) -> B x emb
            cell_output = cell_output.squeeze(1)

            cell_output = cell_output * (~done).float().unsqueeze(1)

        return torch.cat([_.unsqueeze(1) for _ in outputs], 1), cell_state, torch.cat([_.unsqueeze(1) for _ in context_state], 1)


def train_loop(cell, output_fn, inputs, init_state, target_emotion_vector, sequence_length):
    if target_emotion_vector is not None:
        inputs = torch.cat([inputs, target_emotion_vector.unsqueeze(1).expand(inputs.size(0), inputs.size(1), target_emotion_vector.size(1))], 2)
    return dynamic_rnn(cell, inputs, sequence_length, init_state, output_fn) + (None,)