# coding:utf-8
import torch
import json
import random
import numpy as np
import os
from typing import Dict, List, Tuple
from transformers import PreTrainedTokenizer


def set_seed(args):
    # print(f"Setting seed to {seed}")
    random.seed(args.seed)
    np.random.seed(args.seed)
    torch.manual_seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.cuda.manual_seed_all(args.seed)
    os.environ["PYTHONHASHSEED"] = str(args.seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True


def mask_tokens(
    ori_inputs: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability, start_speaker_token=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    inputs = ori_inputs.clone()
    labels = ori_inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    #probability_matrix[labels >= start_speaker_token] = 0.3					# Speaker Mask Rate 0.3
    #probability_matrix[labels >= start_speaker_token] = 0.25					# Speaker Mask Rate 0.25
    #probability_matrix[labels >= start_speaker_token] = 0.2                     # Speaker Mask Rate 0.2
    '''
    if speaker_token_ids is not None:
        for r in range(len(probability_matrix)):
            for l in range(len(probability_matrix[0])):
                if labels[r][l] in speaker_token_ids:
                    probability_matrix[r][l] = 0.9							# Speaker Mask Rate 0.3
    '''
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5).to(labels.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def mask_tokens_sem(
    ori_inputs: torch.Tensor, inputs_sem: torch.Tensor, tokenizer: PreTrainedTokenizer, mlm_probability, start_speaker_token=None
) -> Tuple[torch.Tensor, torch.Tensor]:
    """ Prepare masked tokens inputs/labels for masked language modeling: 80% MASK, 10% random, 10% original. """

    if tokenizer.mask_token is None:
        raise ValueError(
            "This tokenizer does not have a mask token which is necessary for masked language modeling. Remove the --mlm flag if you want to use this tokenizer."
        )

    inputs = ori_inputs.clone()
    labels = ori_inputs.clone()
    # We sample a few tokens in each sequence for masked-LM training (with probability args.mlm_probability defaults to 0.15 in Bert/RoBERTa)
    assert inputs_sem.size() == ori_inputs.size(), f"Inconsistent Size input:{ori_inputs.size()} VS Inputs_Sem:{inputs_sem.size()}"
    probability_matrix = torch.full(labels.shape, mlm_probability).to(labels.device)
    probability_matrix += inputs_sem.float().to(labels.device) * 0.05			# Semantic-aware masking rate  = std masking rate + 0.05
    special_tokens_mask = [
        tokenizer.get_special_tokens_mask(val, already_has_special_tokens=True)
        for val in labels.tolist()
    ]
    probability_matrix.masked_fill_(torch.tensor(special_tokens_mask, dtype=torch.bool, device=labels.device), value=0.0)

    if tokenizer._pad_token is not None:
        padding_mask = labels.eq(tokenizer.pad_token_id)
        probability_matrix.masked_fill_(padding_mask, value=0.0)

    #probability_matrix[labels >= start_speaker_token] = 0.3					# Speaker Mask Rate 0.3
    #probability_matrix[labels >= start_speaker_token] = 0.25					# Speaker Mask Rate 0.25
    #probability_matrix[labels >= start_speaker_token] = 0.2                     # Speaker Mask Rate 0.2
    '''
    if speaker_token_ids is not None:
        for r in range(len(probability_matrix)):
            for l in range(len(probability_matrix[0])):
                if labels[r][l] in speaker_token_ids:
                    probability_matrix[r][l] = 0.9							# Speaker Mask Rate 0.3
    '''
    masked_indices = torch.bernoulli(probability_matrix).bool()
    labels[~masked_indices] = -100  # We only compute loss on masked tokens

    # 80% of the time, we replace masked input tokens with tokenizer.mask_token ([MASK])
    indices_replaced = torch.bernoulli(torch.full(labels.shape, 0.8).to(labels.device)).bool() & masked_indices
    inputs[indices_replaced] = tokenizer.convert_tokens_to_ids(tokenizer.mask_token)

    # 10% of the time, we replace masked input tokens with random word
    indices_random = (
        torch.bernoulli(torch.full(labels.shape, 0.5).to(labels.device)).bool() & masked_indices & ~indices_replaced
    )
    random_words = torch.randint(len(tokenizer), labels.shape, dtype=torch.long, device=labels.device)
    inputs[indices_random] = random_words[indices_random]

    # The rest of the time (10% of the time) we keep the masked input tokens unchanged
    return inputs, labels


def save_dummy_batch(batch, masked_ids, tokenizer, output_dir):
    print("Saving dummy inputs...")
    json_out_path = open(output_dir + "/dummy_input.json", "w", encoding="utf-8")
    ith_dict = {}
    # print('Input Id Size:', batch["input_ids"].size())
    ith_dict["input_ids"] = str(batch["input_ids"].size()) + str(batch["input_ids"].tolist())
    ith_dict["masked_input_ids"] = str(masked_ids.size()) + str(masked_ids.tolist())
    ith_dict["input_tokens"] = tokenizer.batch_decode(batch["input_ids"].tolist())
    ith_dict["masked_input_tokens"] = tokenizer.batch_decode(masked_ids.tolist())
    ith_dict["amr_ids"] = str(batch["amr_ids"].size()) + str(batch["amr_ids"].tolist())
    ith_dict["amr_tokens"] = tokenizer.batch_decode(batch["amr_ids"].tolist())
    ith_dict["joint_ids"] = str(batch["joint_ids"].size()) + str(batch["joint_ids"].tolist())
    ith_dict["joint_tokens"] = tokenizer.batch_decode(batch["joint_ids"].tolist())
    ith_dict["joint_labels"] = str(batch["joint_labels"].size()) + str(batch["joint_labels"].tolist())
    ith_dict["joint_label_tokens"] = tokenizer.batch_decode(batch["joint_labels"].tolist())
    json.dump(ith_dict, json_out_path, indent=4)


def save_json(content, path, indent=4, **json_dump_kwargs):
    with open(path, "w") as f:
        json.dump(content, f, indent=indent, **json_dump_kwargs)


def load_json(path):
    with open(path) as f:
        return json.load(f)


def trim_batch(input_ids, pad_token_id, attention_mask=None):
    """Remove columns that are populated exclusively by pad_token_id"""
    keep_column_mask = input_ids.ne(pad_token_id).any(dim=0)
    if attention_mask is None:
        return input_ids[:, keep_column_mask]
    else:
        return (input_ids[:, keep_column_mask], attention_mask[:, keep_column_mask])


def shift_tokens_right(input_ids: torch.Tensor, pad_token_id: int, decoder_start_token_id: int):
    """
    Shift input ids one token to the right.
    """
    shifted_input_ids = input_ids.new_zeros(input_ids.shape)
    shifted_input_ids[:, 1:] = input_ids[:, :-1].clone()
    shifted_input_ids[:, 0] = decoder_start_token_id

    assert pad_token_id is not None, "self.model.config.pad_token_id has to be defined."
    # replace possible -100 values in labels by `pad_token_id`
    shifted_input_ids.masked_fill_(shifted_input_ids == -100, pad_token_id)

    return shifted_input_ids
