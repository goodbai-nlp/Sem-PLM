# coding:utf-8
import os
import torch
import inspect
import importlib
import torch.distributed
import pytorch_lightning as pl
from datasets import load_from_disk
from collections import Counter
from contextlib import contextmanager
from datasets import load_dataset
from dataclasses import dataclass
from transformers.file_utils import PaddingStrategy
from transformers.modeling_utils import PreTrainedModel
from transformers.tokenization_utils_base import PreTrainedTokenizerBase
from typing import Optional, Union
from torch.utils.data.dataloader import DataLoader
from transformers import BertTokenizer, RobertaTokenizer
from common.constant import amr_tokens, rel2idx

max_length = 512


def adjlst_to_adjmat(inp_str):
    inp_lst = [itm.split(" ") for itm in inp_str.split("\t")]
    lenh = len(inp_lst)
    adj_mat = [[0 for _ in range(lenh)] for _ in range(lenh)]
    for ridx, itm in enumerate(inp_lst):
        if len(itm):
            for cidx in itm:
                if len(cidx):
                    adj_mat[ridx][int(cidx)] = 1
    return adj_mat


def gen_casual_mask(features, key, idx):
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        dec_start_idx = feature[key].index(idx)
        ith_mask = []
        for iidx in range(max_label_length):
            if iidx < dec_start_idx:
                mask_row = [1 for _ in range(dec_start_idx)] + [
                    0 for _ in range(dec_start_idx, max_label_length, 1)
                ]
            else:
                mask_row = [1 for _ in range(iidx + 1)] + [
                    0 for _ in range(iidx + 1, max_label_length, 1)
                ]
            ith_mask.append(mask_row)
        feature["casual_mask"] = ith_mask
    return


@contextmanager
def torch_distributed_zero_first(local_rank: int):
    """
    Decorator to make all processes in distributed training wait for each local_master to do something.
    """
    if local_rank not in [-1, 0]:
        torch.distributed.barrier()
    yield                       #中断后执行上下文代码，然后返回到此处继续往下执行
    if local_rank == 0:
        torch.distributed.barrier()


class AMRDataModule(pl.LightningDataModule):
    def __init__(
        self, **args,
    ):
        super().__init__()
        self.train_file = args["train_data_file"]
        self.validation_file = args["eval_data_file"]
        self.test_file = args["test_data_file"]
        self.src_prefix = args["src_prefix"]
        self.tgt_prefix = args["tgt_prefix"]
        self.pad_to_max_length = False
        self.ignore_pad_token_for_loss = True
        self.cache_dir = args["cache_dir"]
        self.train_batchsize = args["per_gpu_train_batch_size"]
        self.val_batchsize = args["per_gpu_eval_batch_size"]
        self.train_num_worker = args["train_num_workers"]
        self.val_num_worker = args["eval_num_workers"]
        self.preprocess_worker = args["process_num_workers"]
        self.model_arch = args["model_type"]
        self.rel2idx = rel2idx
        self.shuffle_train = args["shuffle"]

        if self.model_arch.lower() == 'bert':
            tokenizer_type = BertTokenizer
            special_toks = ["[unused1]", "[unused2]"]
            assert 'bert' in args["tokenizer_name"] and 'roberta' not in args["tokenizer_name"]
            assert 'bert' in args["model_name_or_path"] and 'roberta' not in args["model_name_or_path"]
        elif self.model_arch.lower() == 'roberta':
            tokenizer_type = RobertaTokenizer
            special_toks = ["madeupword0001", "madeupword0002"]
            assert 'roberta' in args["tokenizer_name"]
            assert 'roberta' in args["model_name_or_path"]
        else:
            print(f'The model_type provided:{self.model_arch} is not supported!!, exit...')

        if args["tokenizer_name"]:
            print(f"Loading tokenizer from: {args['tokenizer_name']}")
            self.tokenizer = tokenizer_type.from_pretrained(
                args["tokenizer_name"], cache_dir=self.cache_dir
            )
        elif args["model_name_or_path"]:
            self.tokenizer = tokenizer_type.from_pretrained(
                args["model_name_or_path"], cache_dir=self.cache_dir
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )
        self.tokenizer.add_special_tokens(
            {"additional_special_tokens": special_toks + amr_tokens}
        )
        self.tokenizer.amr_bos_token = "<AMR>"
        self.tokenizer.amr_bos_token_id = self.tokenizer.convert_tokens_to_ids(
            self.tokenizer.amr_bos_token
        )

        self.max_src_length = min(args["src_block_size"], self.tokenizer.model_max_length)
        self.max_tgt_length = min(args["tgt_block_size"], self.tokenizer.model_max_length)

        self.collate_fn = DataCollatorForDialogAMR(
            self.tokenizer, label_pad_token_id=-100, pad_to_multiple_of=1 if args["fp16"] else None,
        )
        
        data_files = {}
        data_files["train"] = self.train_file
        data_files["validation"] = self.validation_file
        data_files["test"] = self.test_file

        # print("datafiles:", data_files)
        # print("Dataset cache dir:", self.cache_dir)
        self.datasets = load_dataset(
            f"{os.path.dirname(__file__)}/amrdata.py",
            data_files=data_files,
        )
        print("datasets:", self.datasets)
        self.train_dataset_len = len(self.datasets["train"])


    def setup(self, stage="fit"):
        
        def tokenize_function(examples):
            # Remove empty lines
            src = examples["src"]  # text tokens
            src_sem = [[int(iitm) for iitm in itm.split()] for itm in examples["src_ali"]]
            # src_sem_size = [len(itm) for itm in src_sem]
            src_rel_mask = [itm for itm in examples["rel_mask"]]
            
            # src_rel_mask = [adjlst_to_adjmat(itm) for itm in examples["rel_mask"]]
            # src_rel_size = [len(itm) for itm in src_rel_mask]
            # diff = sum(a - b for a, b in zip(src_sem_size, src_rel_size))
            # assert diff == 0, f"inconsistent lengths: {src_sem_size} vs {src_rel_size}"
            src_rel_label = []
            for itm in examples["rel_label"]:
                src_rel_label.append([self.rel2idx.get(iitm, 50) for iitm in itm.split()])
            # print("rel_mask", examples["rel_mask"])
            # print("rel_mask_matrix", src_rel_mask)
            # print("src_rel_label", examples["rel_label"])

            # src_rel_size2 = [len(itm) for itm in src_rel_label]
            # src_mask_size2 = [sum(sum(iitm) for iitm in itm) for itm in src_rel_mask]
            # diff = sum(a - b for a, b in zip(src_mask_size2, src_rel_size2))
            # assert diff == 0, f"inconsistent lengths: {src_mask_size2} vs {src_rel_size2}"
            src_rel_label = [itm if len(itm) > 0 else [0] for itm in src_rel_label]
            amr = examples["amr"]  # amr tokens

            src = [inp.split() for inp in src]
            amr = [self.tgt_prefix + inp for inp in amr]
            src_ids = [self.tokenizer.convert_tokens_to_ids(itm) for itm in src]
            tgt_ids = self.tokenizer(
                amr, max_length=self.max_tgt_length, padding=False, truncation=True
            )
            
            # joint_ids = [
            #     (src + [self.tokenizer.amr_bos_token_id] + tgt[1:-1])[:max_length]
            #     for src, tgt in zip(src_ids, tgt_ids["input_ids"])
            # ]
            # joint_labels = [
            #     (src + tgt[1:-1])[: max_length - 1] + [self.tokenizer.eos_token_id]
            #     for src, tgt in zip(src_ids, tgt_ids["input_ids"])
            # ]
            return {
                "input_ids": src_ids,
                "input_sem": src_sem,
                "amr_ids": tgt_ids["input_ids"],
                "src_rel_mask": src_rel_mask,
                "rel_label": src_rel_label,
            }
            # return {
            #     "input_ids": src_ids,
            #     "input_sem": src_sem,
            #     "amr_ids": tgt_ids["input_ids"],
            #     "joint_ids": joint_ids,
            #     "joint_labels": joint_labels,
            #     "rel_mask": src_rel_mask,
            #     "rel_label": src_rel_label,
            # }
        print("Dataset cache dir:", self.cache_dir)
        with torch_distributed_zero_first(int(os.getenv("LOCAL_RANK", "0"))):
            if not len(os.listdir(self.cache_dir)):
                self.train_dataset = self.datasets["train"].map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.preprocess_worker,
                    batch_size=1000,
                    load_from_cache_file=True,
                    remove_columns=["src", "amr", "src_ali", "rel_label", "rel_mask"],
                )
                print(f"ALL {len(self.train_dataset)} training instances")
                
                self.valid_dataset = self.datasets["validation"].map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.preprocess_worker,
                    load_from_cache_file=True,
                    remove_columns=["src", "amr", "src_ali", "rel_label", "rel_mask"],
                )
                print(f"ALL {len(self.valid_dataset)} validation instances")
                
                self.test_dataset = self.datasets["test"].map(
                    tokenize_function,
                    batched=True,
                    num_proc=self.preprocess_worker,
                    load_from_cache_file=True,
                    remove_columns=["src", "amr", "src_ali", "rel_label", "rel_mask"],
                )
                print(f"ALL {len(self.test_dataset)} test instances")
                print(f"Saving processed dataset to {self.cache_dir}")
                self.train_dataset.save_to_disk(self.cache_dir + "/train")
                self.valid_dataset.save_to_disk(self.cache_dir + "/valid")
                self.test_dataset.save_to_disk(self.cache_dir + "/test")
            else:
                self.train_dataset = load_from_disk(self.cache_dir + "/train")
                self.valid_dataset = load_from_disk(self.cache_dir + "/valid")
                self.test_dataset = load_from_disk(self.cache_dir + "/test")
                
                print(f"ALL {len(self.train_dataset)} training instances")
                print(f"ALL {len(self.valid_dataset)} validation instances")
                print(f"ALL {len(self.test_dataset)} test instances")
            print("Dataset Instance Example:", self.train_dataset[0])
            del self.datasets

    def train_dataloader(self):
        if self.train_dataset:
            print(f"Training set shuffle=={self.shuffle_train},###############################")
            return DataLoader(
                self.train_dataset,
                batch_size=self.train_batchsize,
                collate_fn=self.collate_fn,
                shuffle=self.shuffle_train,
                num_workers=self.train_num_worker,
            )
        else:
            return None

    def val_dataloader(self):
        if self.valid_dataset:
            return DataLoader(
                self.valid_dataset,
                batch_size=self.val_batchsize,
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.val_num_worker,
            )
        else:
            return None

    def test_dataloader(self):
        if self.test_dataset:
            return DataLoader(
                self.test_dataset,
                batch_size=self.val_batchsize,
                collate_fn=self.collate_fn,
                shuffle=False,
                num_workers=self.val_num_worker,
            )
        return None

    def load_data_module(self):
        name = self.dataset
        # Change the `snake_case.py` file name to `CamelCase` class name.
        # Please always name your model file name as `snake_case.py` and
        # class name corresponding `CamelCase`.
        camel_name = "".join([i.capitalize() for i in name.split("_")])
        try:
            self.data_module = getattr(
                importlib.import_module("." + name, package=__package__), camel_name
            )
        except:
            raise ValueError(
                f"Invalid Dataset File Name or Invalid Class Name data.{name}.{camel_name}"
            )

    def instancialize(self, **other_args):
        """ Instancialize a model using the corresponding parameters
            from self.hparams dictionary. You can also input any args
            to overwrite the corresponding value in self.kwargs.
        """
        class_args = inspect.getargspec(self.data_module.__init__).args[1:]
        inkeys = self.kwargs.keys()
        args1 = {}
        for arg in class_args:
            if arg in inkeys:
                args1[arg] = self.kwargs[arg]
        args1.update(other_args)
        return self.data_module(**args1)


def padding_func(features, padding_side="right", pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    for feature in features:
        remainder = [pad_token_id] * (max_label_length - len(feature[key]))
        feature[key] = (
            feature[key] + remainder if padding_side == "right" else remainder + feature[key]
        )
    return


def padding_func_concat(features, key="label"):
    concat_labels = []
    for feature in features:
        concat_labels += [int(itm) for itm in feature[key] if itm != 0]
    for feature in features:
        feature[key] = concat_labels
    # print("concated labels", concat_labels)
    return


def padding_func_matrix(features, pad_token_id=1, key="label"):
    assert key in features[0].keys(), f"{key} not in {features[0].keys()}"
    max_label_length = max(len(feature[key]) for feature in features)
    # padded_res = [[0 for l in range(max_label_length)] for r in range(max_label_length)]         # [bsz, max_len, max_len]
    for feature in features:
        padded_idx = [
            [pad_token_id for l in range(max_label_length)] for r in range(max_label_length)
        ]
        for r_idx, row in enumerate(feature[key]):
            r_len = len(row)
            padded_idx[r_idx][:r_len] = row
        feature[key] = padded_idx
    return


def build_additional_features(features, tokenizer):
    for feature in features:
        src = feature["input_ids"]
        tgt = feature["amr_ids"]
        joint_ids = (src + [tokenizer.amr_bos_token_id] + tgt[1:-1])[:max_length]
        if tokenizer.eos_token_id:
            joint_labels = (src + tgt[1:-1])[: max_length - 1] + [tokenizer.eos_token_id]
        else:
            joint_labels = (src + tgt[1:-1])[: max_length - 1] + [tokenizer.sep_token_id]

        feature["joint_ids"] = joint_ids
        feature["joint_labels"] = joint_labels
        rel_mask = adjlst_to_adjmat(feature["src_rel_mask"])
        assert rel_mask is not None
        feature["rel_mask"] = rel_mask
        del feature["src_rel_mask"]

    return 
    

@dataclass
class DataCollatorForDialogAMR:
    """
    Data collator that will dynamically pad the inputs received, as well as the labels.

    Args:
        tokenizer (:class:`~transformers.PreTrainedTokenizer` or :class:`~transformers.PreTrainedTokenizerFast`):
            The tokenizer used for encoding the data.
        model (:class:`~transformers.PreTrainedModel`):
            The model that is being trained. If set and has the `prepare_decoder_input_ids_from_labels`, use it to
            prepare the `decoder_input_ids`

            This is useful when using `label_smoothing` to avoid calculating loss twice.
        padding (:obj:`bool`, :obj:`str` or :class:`~transformers.file_utils.PaddingStrategy`, `optional`, defaults to :obj:`True`):
            Select a strategy to pad the returned sequences (according to the model's padding side and padding index)
            among:

            * :obj:`True` or :obj:`'longest'`: Pad to the longest sequence in the batch (or no padding if only a single
              sequence is provided).
            * :obj:`'max_length'`: Pad to a maximum length specified with the argument :obj:`max_length` or to the
              maximum acceptable input length for the model if that argument is not provided.
            * :obj:`False` or :obj:`'do_not_pad'` (default): No padding (i.e., can output a batch with sequences of
              different lengths).
        max_length (:obj:`int`, `optional`):
            Maximum length of the returned list and optionally padding length (see above).
        pad_to_multiple_of (:obj:`int`, `optional`):
            If set will pad the sequence to a multiple of the provided value.

            This is especially useful to enable the use of Tensor Cores on NVIDIA hardware with compute capability >=
            7.5 (Volta).
        label_pad_token_id (:obj:`int`, `optional`, defaults to -100):
            The id to use when padding the labels (-100 will be automatically ignored by PyTorch loss functions).
    """

    tokenizer: PreTrainedTokenizerBase
    model: Optional[PreTrainedModel] = None
    padding: Union[bool, str, PaddingStrategy] = True
    max_length: Optional[int] = None
    pad_to_multiple_of: Optional[int] = None
    label_pad_token_id: int = -100

    def __call__(self, features):
        build_additional_features(
            features,
            self.tokenizer,
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="amr_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="joint_ids",
        )
        padding_func(
            features,
            padding_side=self.tokenizer.padding_side,
            pad_token_id=self.tokenizer.pad_token_id,
            key="joint_labels",
        )
        gen_casual_mask(features, "joint_ids", self.tokenizer.amr_bos_token_id)
        padding_func(
            features, padding_side=self.tokenizer.padding_side, pad_token_id=0, key="input_sem",
        )
        padding_func_concat(
            features, key="rel_label",
        )
        padding_func_matrix(
            features, pad_token_id=0, key="rel_mask",
        )
        # lengths = {}
        # for itm in features:
        #     for k,v in itm.items():
        #         if k in ["rel_mask","casual_mask"]:
        #             if k not in lengths:
        #                 lengths[k] = [(len(v), len(v[0]))]
        #             else:
        #                 lengths[k].append((len(v), len(v[0])))
        #             # print(f'{k}:{len(v)}, {len(v[0])}')
        #         else:
        #             if k not in lengths:
        #                 lengths[k] = [len(v)]
        #             else:
        #                 lengths[k].append(len(v))
        #             # print(f'{k}:{len(v)}')
        # print(lengths)
        # print("Features:", features)
        features = self.tokenizer.pad(
            features,
            padding=self.padding,
            max_length=self.max_length,
            pad_to_multiple_of=self.pad_to_multiple_of,
            return_tensors="pt",
        )
        return {
            "input_ids": features["input_ids"],
            "input_sem": features["input_sem"],
            "joint_ids": features["joint_ids"],
            "attention_mask": features["attention_mask"],
            "casual_mask": features["casual_mask"],
            "joint_labels": features["joint_labels"],
            "rel_mask": features["rel_mask"],
            "rel_label": features["rel_label"],
            "amr_ids": features["amr_ids"],
        }
