# coding:utf-8

# Copyright 2018 The Google AI Language Team Authors and The HugginFace Inc. team.
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""PyTorch Roberta model."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math
import torch
import torch.nn as nn
from typing import Optional, Tuple
from torch.nn import CrossEntropyLoss, BCEWithLogitsLoss
from transformers import BertConfig
from transformers.file_utils import ModelOutput
from transformers.modeling_utils import PreTrainedModel
import torch.distributed as dist
import torch.nn.functional as F
import transformers.models.bert.modeling_bert as BERT
import transformers.models.roberta.modeling_roberta as RoBerta


class DotProductSimilarity(torch.nn.Module):
    def __init__(self, scale_output=False):
        super(DotProductSimilarity, self).__init__()
        self.scale_output = scale_output

    def forward(self, tensor_1, tensor_2):
        result = (tensor_1 * tensor_2).sum(dim=-1)
        if self.scale_output:
            # TODO why allennlp do multiplication at here ?
            result /= math.sqrt(tensor_1.size(-1))
        return result.mean()


class ContrastiveSimilarity(torch.nn.Module):
    def __init__(self, tau=1.0, reduction="mean", eps=1e-6, cos_sim=False):
        super(ContrastiveSimilarity, self).__init__()
        self.tau = tau
        self.eps = eps
        self.reduction = reduction
        self.cos_sim = cos_sim

    def forward(self, tensor_1, tensor_2):
        bsz = tensor_1.size(0)
        if self.cos_sim:
            tensor_1 = tensor_1 / tensor_1.norm(dim=1).clamp(min=self.eps)[:, None]
            tensor_2 = tensor_2 / tensor_2.norm(dim=1).clamp(min=self.eps)[:, None]
        logits = torch.einsum("nc,mc->nm", tensor_1, tensor_2) / self.tau
        labels = torch.arange(bsz, device=tensor_1.device, dtype=torch.long)
        loss = F.cross_entropy(logits, labels, reduction=self.reduction)
        return loss


# cos_loss_fn = torch.nn.CosineSimilarity(dim=1, eps=1e-08)
# con_loss_fn = ContrastiveSimilarity(scale_output=False)


class BaseModelOutputWithPoolingAndCrossAttentionsAndCode(ModelOutput):
    """
    Base class for model's outputs that also contains a pooling of the last hidden states.
    Args:
        last_hidden_state (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, hidden_size)`):
            Sequence of hidden-states at the output of the last layer of the model.
        pooler_output (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, hidden_size)`):
            Last layer hidden-state of the first token of the sequence (classification token) after further processing
            through the layers used for the auxiliary pretraining task. E.g. for BERT-family of models, this returns
            the classification token after processing through a linear layer and a tanh activation function. The linear
            layer weights are trained from the next sentence prediction (classification) objective during pretraining.
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.
            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
        cross_attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` and ``config.add_cross_attention=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.
            Attentions weights of the decoder's cross-attention layer, after the attention softmax, used to compute the
            weighted average in the cross-attention heads.
        past_key_values (:obj:`tuple(tuple(torch.FloatTensor))`, `optional`, returned when ``use_cache=True`` is passed or when ``config.use_cache=True``):
            Tuple of :obj:`tuple(torch.FloatTensor)` of length :obj:`config.n_layers`, with each tuple having 2 tensors
            of shape :obj:`(batch_size, num_heads, sequence_length, embed_size_per_head)`) and optionally if
            ``config.is_encoder_decoder=True`` 2 additional tensors of shape :obj:`(batch_size, num_heads,
            encoder_sequence_length, embed_size_per_head)`.
            Contains pre-computed hidden-states (key and values in the self-attention blocks and optionally if
            ``config.is_encoder_decoder=True`` in the cross-attention blocks) that can be used (see
            :obj:`past_key_values` input) to speed up sequential decoding.
    """

    last_hidden_state: torch.FloatTensor = None
    pooler_output: torch.FloatTensor = None
    code_states: Optional[torch.FloatTensor] = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    past_key_values: Optional[Tuple[Tuple[torch.FloatTensor]]] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None
    cross_attentions: Optional[Tuple[torch.FloatTensor]] = None


class MaskedLMOutputWithPool(ModelOutput):
    """
    Output type of :class:`~transformers.BertForPreTraining`.

    Args:
        loss (`optional`, returned when ``labels`` is provided, ``torch.FloatTensor`` of shape :obj:`(1,)`):
            Total loss as the sum of the masked language modeling loss and the next sequence prediction
            (classification) loss.
        prediction_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, sequence_length, config.vocab_size)`):
            Prediction scores of the language modeling head (scores for each vocabulary token before SoftMax).
        seq_relationship_logits (:obj:`torch.FloatTensor` of shape :obj:`(batch_size, 2)`):
            Prediction scores of the next sequence prediction (classification) head (scores of True/False continuation
            before SoftMax).
        hidden_states (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_hidden_states=True`` is passed or when ``config.output_hidden_states=True``):
            Tuple of :obj:`torch.FloatTensor` (one for the output of the embeddings + one for the output of each layer)
            of shape :obj:`(batch_size, sequence_length, hidden_size)`.

            Hidden-states of the model at the output of each layer plus the initial embedding outputs.
        attentions (:obj:`tuple(torch.FloatTensor)`, `optional`, returned when ``output_attentions=True`` is passed or when ``config.output_attentions=True``):
            Tuple of :obj:`torch.FloatTensor` (one for each layer) of shape :obj:`(batch_size, num_heads,
            sequence_length, sequence_length)`.

            Attentions weights after the attention softmax, used to compute the weighted average in the self-attention
            heads.
    """

    loss: Optional[torch.FloatTensor] = None
    logits: torch.FloatTensor = None
    hidden_states: Optional[Tuple[torch.FloatTensor]] = None
    pooled_state: Optional[torch.FloatTensor] = None
    attentions: Optional[Tuple[torch.FloatTensor]] = None


class MultiheadAttention(nn.Module):
    def __init__(self, nheads=1, hidden_size=768, drop=0.1):
        super().__init__()
        self.output_attentions = False
        self.num_attention_heads = nheads
        self.attention_head_size = int(hidden_size / nheads)
        self.all_head_size = self.num_attention_heads * self.attention_head_size

        self.query = nn.Linear(hidden_size, self.all_head_size,)
        self.key = nn.Linear(hidden_size, self.all_head_size,)
        self.value = nn.Linear(hidden_size, self.all_head_size,)
        self.dropout = nn.Dropout(drop)

    def transpose_for_scores(self, x):
        new_x_shape = x.size()[:-1] + (self.num_attention_heads, self.attention_head_size)
        x = x.view(*new_x_shape)
        return x.permute(0, 2, 1, 3)

    def forward(
        self,
        hidden_states,
        attention_mask=None,
        head_mask=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        threshold=None,
    ):
        mixed_query_layer = self.query(hidden_states)
        # If this is instantiated as a cross-attention module, the keys
        # and values come from an encoder; the attention mask needs to be
        # such that the encoder's padding tokens are not attended to.
        if encoder_hidden_states is not None:
            mixed_key_layer = self.key(encoder_hidden_states)
            mixed_value_layer = self.value(encoder_hidden_states)
            attention_mask = encoder_attention_mask
        else:
            mixed_key_layer = self.key(hidden_states)
            mixed_value_layer = self.value(hidden_states)

        query_layer = self.transpose_for_scores(mixed_query_layer)
        key_layer = self.transpose_for_scores(mixed_key_layer)
        value_layer = self.transpose_for_scores(mixed_value_layer)

        # Take the dot product between "query" and "key" to get the raw attention scores.
        attention_scores = torch.matmul(query_layer, key_layer.transpose(-1, -2))
        attention_scores = attention_scores / math.sqrt(self.attention_head_size)
        if attention_mask is not None:
            # Apply the attention mask is (precomputed for all layers in BertModel forward() function)
            attention_scores = attention_scores + attention_mask

        # Normalize the attention scores to probabilities.
        attention_probs = nn.Softmax(dim=-1)(attention_scores)

        # This is actually dropping out entire tokens to attend to, which might
        # seem a bit unusual, but is taken from the original Transformer paper.
        attention_probs = self.dropout(attention_probs)

        # Mask heads if we want to
        if head_mask is not None:
            attention_probs = attention_probs * head_mask

        context_layer = torch.matmul(attention_probs, value_layer)

        context_layer = context_layer.permute(0, 2, 1, 3).contiguous()
        new_context_layer_shape = context_layer.size()[:-2] + (self.all_head_size,)
        context_layer = context_layer.view(*new_context_layer_shape)

        outputs = (context_layer, attention_probs) if self.output_attentions else (context_layer,)
        return outputs


def dot_attention(q, k, v, enc_mask=None):
    # q: [bs, poly_m, dim] or [bs, res_cnt, dim]
    # k=v: [bs, length, dim] or [bs, poly_m, dim]
    attn_weights = torch.matmul(q, k.transpose(2, 1))  # [bs, poly_m, length]
    if enc_mask is not None:  # [bsz, 1, key_len]
        attn_weights += enc_mask
    # print("attn_weights:", attn_weights.size())
    attn_weights = F.softmax(attn_weights, -1)
    output = torch.matmul(attn_weights, v)  # [bs, poly_m, dim]
    return output


class DualPLMRobertaModel(RoBerta.RobertaPreTrainedModel):
    def __init__(self, config):
        super().__init__(config)
        self.config = config
        self.sent_model = RobertaForMaskedLMWithPool(config)
        self.amr_model = RoBerta.RobertaModel(config, add_pooling_layer=True)
        self.recon_ratio = config.recon_ratio
        self.rel_ratio = config.rel_ratio
        
    def forward(
        self,
        sent_input_ids=None,
        sent_attention_mask=None,
        sent_token_type_ids=None,
        sent_position_ids=None,
        ori_sent_input_ids=None,
        ori_sent_attention_mask=None,
        ori_sent_token_type_ids=None,
        ori_sent_position_ids=None,
        amr_input_ids=None,
        amr_attention_mask=None,
        amr_token_type_ids=None,
        amr_position_ids=None,
        joint_input_ids=None,
        joint_attention_mask=None,
        labels=None,
        joint_labels=None,
        rel_mask=None,
        rel_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        mlm_sent_out = self.sent_model(
            input_ids=sent_input_ids,
            attention_mask=sent_attention_mask,
            token_type_ids=sent_token_type_ids,
            position_ids=sent_position_ids,
            labels=labels,
            rel_mask=None,
            rel_labels=None,
            output_attentions=False,
            output_hidden_states=False,
            return_dict=return_dict,
        )
        mlm_loss = mlm_sent_out[0]
        if amr_input_ids is not None or rel_mask is not None:
            std_sent_out = self.sent_model(
                input_ids=ori_sent_input_ids,
                attention_mask=ori_sent_attention_mask,
                token_type_ids=ori_sent_token_type_ids,
                position_ids=ori_sent_position_ids,
                labels=None,
                rel_mask=rel_mask,
                rel_labels=rel_labels,
                output_attentions=output_attentions,
                output_hidden_states=True,
                return_dict=return_dict,
            )
            rel_loss = std_sent_out[0]
            sent_pool = std_sent_out[3]
        else:
            rel_loss = 0
        
        if joint_input_ids is not None:
            assert joint_attention_mask is not None and joint_labels is not None
            text2amr_out = self.sent_model(
                input_ids=joint_input_ids,
                attention_mask=joint_attention_mask,
                labels=joint_labels,
                rel_mask=None,
                rel_labels=None,
                output_attentions=False,
                output_hidden_states=False,
                return_dict=return_dict,
            )
            text2amr_loss = text2amr_out[0]
        else:
            text2amr_loss = 0
                    
        sent_loss = mlm_loss + self.rel_ratio * rel_loss + self.recon_ratio * text2amr_loss

        if amr_input_ids is not None:
            amr_out = self.amr_model(
                amr_input_ids,
                amr_attention_mask,
                amr_token_type_ids,
                amr_position_ids,
                output_attentions=output_attentions,
                output_hidden_states=output_hidden_states,
                return_dict=return_dict,
            )
            return sent_loss, sent_pool, amr_out[1]

            # amr_poly_mem = amr_out[2]                   # AMR poly Memory
            # context = dot_attention(
            #     sent_pool.unsqueeze(1),                 # [bsz, 1, hidden_dim]
            #     amr_poly_mem,                           # [bsz, M, hidden_dim]
            #     amr_poly_mem,                           # [bsz, M, hidden_dim]
            # )
            # return sent_loss, sent_pool, context.squeeze(1)
        else:
            return (sent_loss,)


class RobertaForMaskedLMWithPool(RoBerta.RobertaPreTrainedModel):
    _keys_to_ignore_on_save = [r"lm_head.decoder.weight", r"lm_head.decoder.bias"]
    _keys_to_ignore_on_load_missing = [
        r"position_ids",
        r"lm_head.decoder.weight",
        r"lm_head.decoder.bias",
    ]

    def __init__(self, config):
        super().__init__(config)

        self.roberta = RoBerta.RobertaModel(config, add_pooling_layer=True)
        self.lm_head = RoBerta.RobertaLMHead(config)
        if config.use_rel:
            self.rel_head = Biaffine(config.hidden_size, 200, 0.1, config.rel_vocab_size)

        # The LM head weights require special treatment only when they are tied with the word embeddings
        self.update_keys_to_ignore(config, ["lm_head.decoder.weight"])

        self.init_weights()

    def get_output_embeddings(self):
        return self.lm_head.decoder

    def set_output_embeddings(self, new_embeddings):
        self.lm_head.decoder = new_embeddings

    def forward(
        self,
        input_ids=None,
        attention_mask=None,
        token_type_ids=None,
        position_ids=None,
        head_mask=None,
        inputs_embeds=None,
        encoder_hidden_states=None,
        encoder_attention_mask=None,
        labels=None,
        rel_mask=None,
        rel_labels=None,
        output_attentions=None,
        output_hidden_states=None,
        return_dict=None,
    ):
        r"""
        labels (:obj:`torch.LongTensor` of shape :obj:`(batch_size, sequence_length)`, `optional`):
            Labels for computing the masked language modeling loss. Indices should be in ``[-100, 0, ...,
            config.vocab_size]`` (see ``input_ids`` docstring) Tokens with indices set to ``-100`` are ignored
            (masked), the loss is only computed for the tokens with labels in ``[0, ..., config.vocab_size]``
        kwargs (:obj:`Dict[str, any]`, optional, defaults to `{}`):
            Used to hide legacy arguments that have been deprecated.
        """
        return_dict = return_dict if return_dict is not None else self.config.use_return_dict

        outputs = self.roberta(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            encoder_hidden_states=encoder_hidden_states,
            encoder_attention_mask=encoder_attention_mask,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )
        sequence_output = outputs[0]  # [bsz, seq_len, hidden_size]
        if rel_mask is not None and rel_labels is not None:
            arc_out, rel_out = self.rel_head(sequence_output)
            a_out = F.log_softmax(arc_out, 2)  # [bsz, seq_len, seq_len]
            # print('ori_a_out', a_out.size())
            # print('ori_rel_out', rel_out.size())
            a_out = a_out[rel_mask.bool()]
            # print("indexed_a_out", a_out.size())
            a_loss = -a_out.sum()
            r_out = rel_out[rel_mask.bool()]  # [K, rel_vocab]
            r_out = F.log_softmax(r_out, -1)  # [K, rel_vocab]
            rel_loss_func = nn.NLLLoss(ignore_index=0, reduction="sum")
            # print("r_out", r_out.size())
            if len(rel_labels[0]):
                r_loss = rel_loss_func(r_out, rel_labels[0].long())
                rel_loss = (r_loss + a_loss) / len(rel_labels[0])
            else:
                rel_loss = 0
            # print(rel_loss.item())
        else:
            rel_loss = 0

        prediction_scores = self.lm_head(sequence_output)

        masked_lm_loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            masked_lm_loss = loss_fct(
                prediction_scores.view(-1, self.config.vocab_size), labels.view(-1)
            )
        else:
            masked_lm_loss = 0
            
        loss = rel_loss + masked_lm_loss

        if not return_dict:
            output = (prediction_scores,) + outputs[2:]
            return ((loss,) + output) if loss is not None else output

        return MaskedLMOutputWithPool(
            loss=loss,
            logits=prediction_scores,
            hidden_states=outputs.hidden_states,
            pooled_state=outputs.pooler_output,
            attentions=outputs.attentions,
        )


class GatherLayer(torch.autograd.Function):
    """Gather tensors from all process, supporting backward propagation."""

    @staticmethod
    def forward(ctx, input):
        ctx.save_for_backward(input)
        output = [torch.zeros_like(input) for _ in range(dist.get_world_size())]
        dist.all_gather(output, input)
        return tuple(output)

    @staticmethod
    def backward(ctx, *grads):
        (input,) = ctx.saved_tensors
        grad_out = torch.zeros_like(input)
        grad_out[:] = grads[dist.get_rank()]
        return grad_out


class Biaffine(nn.Module):
    def __init__(self, ori_size, in_size=200, dropout=0.1, out_size=32):
        super(Biaffine, self).__init__()
        self.in_size = in_size
        self.out_size = out_size
        self.dense = nn.Linear(ori_size, in_size)
        self.head_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
        self.dep_mlp = nn.Sequential(nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU())
        self.label_head_mlp = nn.Sequential(
            nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU()
        )
        self.label_dep_mlp = nn.Sequential(
            nn.Linear(in_size, in_size), nn.Dropout(dropout), nn.ELU()
        )

        self.arc_attn = BiLinear(n_in=self.in_size, bias_x=True, bias_y=False)
        self.rel_attn = BiLinear(n_in=self.in_size, n_out=self.out_size, bias_x=True, bias_y=True)
        self.reset_parameters()

    def reset_parameters(self):
        nn.init.xavier_uniform_(self.dense.weight)
        nn.init.constant_(self.dense.bias, 0.0)
        nn.init.xavier_uniform_(self.head_mlp[0].weight)
        nn.init.constant_(self.head_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.dep_mlp[0].weight)
        nn.init.constant_(self.dep_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.label_head_mlp[0].weight)
        nn.init.constant_(self.label_head_mlp[0].bias, 0.0)
        nn.init.xavier_uniform_(self.label_dep_mlp[0].weight)
        nn.init.constant_(self.label_dep_mlp[0].bias, 0.0)

    def forward(self, x):
        """
        :param input: output of decoder [batch_size, seq_len, H_dim]
        :param mask: dependency matrix of target sentence
        :return: masked arc attn, masked label attn
        """
        x = self.dense(x)
        arc_h = self.head_mlp(x)
        arc_d = self.dep_mlp(x)
        rel_h = self.label_head_mlp(x)
        rel_d = self.label_dep_mlp(x)

        # get arc and rel scores from the bilinear attention
        # [batch_size, seq_len, seq_len]
        s_arc = self.arc_attn(arc_d, arc_h)
        # [batch_size, seq_len, seq_len, n_rels]
        s_rel = self.rel_attn(rel_d, rel_h).permute(0, 2, 3, 1)

        # out = F.log_softmax(s_arc, 2)                     # [batch_size, seq_len, seq_len]
        # out=out[mask]                                     # [k]
        # out=out.sum()                                     # [1]
        # l_out = s_rel[mask]                               # [k, Label_vocab]
        # l_out = F.log_softmax(l_out, -1)                  # [k, Label_vocab]

        # out = F.softmax(s_arc, 2)
        # l_out = F.softmax(s_rel, -1)                      # [batch_size, seq_len, seq_len, n_rels]
        return s_arc, s_rel


class BiLinear(nn.Module):
    def __init__(self, n_in, n_out=1, bias_x=True, bias_y=True):
        super(BiLinear, self).__init__()
        self.n_in = n_in
        self.n_out = n_out
        self.bias_x = bias_x
        self.bias_y = bias_y
        self.weight = nn.Parameter(torch.Tensor(n_out, n_in + bias_x, n_in + bias_y))
        self.reset_parameters()

    def extra_repr(self):
        s = f"n_in={self.n_in}, n_out={self.n_out}"
        if self.bias_x:
            s += f", bias_x={self.bias_x}"
        if self.bias_y:
            s += f", bias_y={self.bias_y}"

        return s

    def reset_parameters(self):
        nn.init.zeros_(self.weight)

    def forward(self, x, y):
        if self.bias_x:
            x = torch.cat((x, torch.ones_like(x[..., :1])), -1)
        if self.bias_y:
            y = torch.cat((y, torch.ones_like(y[..., :1])), -1)
        # [batch_size, n_out, seq_len, seq_len]
        s = torch.einsum("bxi,oij,byj->boxy", x, self.weight, y)
        # remove dim 1 if n_out == 1
        s = s.squeeze(1)

        return s

