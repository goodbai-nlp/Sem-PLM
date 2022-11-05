# coding:utf-8
import sys
sys.path.append("..")

import torch
from pathlib import Path
from collections import defaultdict
from typing import Dict, List, Tuple, Any
from common.constant import (
    amr_tokens,
    arg_to_scheduler,
    arg_to_scheduler_choices,
    arg_to_scheduler_metavar,
)
from .modeling_roberta_parser import DualPLMRobertaModel, RobertaForMaskedLMWithPool, ContrastiveSimilarity, GatherLayer
# from .modeling_bert_parser import DualPLMBertModel, BertForMaskedLMWithPool, ContrastiveSimilarity, GatherLayer
from transformers import (
    WEIGHTS_NAME,
    AdamW,
    Adafactor,
    AutoConfig,
    AutoTokenizer,
    BertTokenizer,
    RobertaTokenizer,
)
import pytorch_lightning as pl
from common.utils import mask_tokens, mask_tokens_sem, save_dummy_batch
import transformers.models.bert.modeling_bert as BERT
import transformers.models.roberta.modeling_roberta as RoBerta

class DialogLMModel(pl.LightningModule):
    loss_names = ["loss"]
    default_val_metric = "loss"
    
    def __init__(self, args):
        super().__init__()
        self.save_hyperparameters(args)
        self.metrics = defaultdict(list)
        if args.config_name:
            config = AutoConfig.from_pretrained(args.config_name, cache_dir=args.cache_dir)
        elif args.model_name_or_path:
            config = AutoConfig.from_pretrained(args.model_name_or_path, cache_dir=args.cache_dir)
        else:
            raise ValueError(
                "You are instantiating a new config instance from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --config_name"
            )
        self.modify_config(args, config, ())

        setattr(config, 'rel_vocab_size', 51)
        setattr(config, 'use_rel', getattr(args, 'use_rel'))
        setattr(config, 'rel_ratio', getattr(args, 'rel_ratio'))
        setattr(config, 'recon_ratio', getattr(args, 'recon_ratio'))
        self.model_arch = args.model_type
        self.contrastive_tau = args.temperature
        if self.model_arch.lower() == 'bert':
            tokenizer_type = BertTokenizer
            model_type = DualPLMBertModel
            sent_model_type = BertForMaskedLMWithPool
            amr_model_type = BERT.BertModel
            special_toks = ["[unused1]", "[unused2]"]
            assert 'bert' in args.tokenizer_name and 'roberta' not in args.tokenizer_name
            assert 'bert' in args.model_name_or_path and 'roberta' not in args.model_name_or_path
        elif self.model_arch.lower() == 'roberta':
            tokenizer_type = RobertaTokenizer
            model_type = DualPLMRobertaModel
            sent_model_type = RobertaForMaskedLMWithPool
            amr_model_type = RoBerta.RobertaModel
            special_toks = ["madeupword0001", "madeupword0002"]
            assert 'roberta' in args.tokenizer_name
            assert 'roberta' in args.model_name_or_path
        else:
            print(f'The model_type provided:{self.model_arch} is not supported!!, exit...')

        if args.tokenizer_name:
            self.tokenizer = tokenizer_type.from_pretrained(
                args.tokenizer_name, cache_dir=args.cache_dir
            )
        elif args.model_name_or_path:
            self.tokenizer = tokenizer_type.from_pretrained(
                args.model_name_or_path, cache_dir=args.cache_dir
            )
        else:
            raise ValueError(
                "You are instantiating a new tokenizer from scratch. This is not supported, but you can do it from another script, save it,"
                "and load it from here, using --tokenizer_name"
            )
        
        self.tokenizer.add_special_tokens({"additional_special_tokens": special_toks + amr_tokens})
        self.model = model_type(config=config)

        if args.model_name_or_path:
            self.model.sent_model = sent_model_type.from_pretrained(
                args.model_name_or_path, config=config, cache_dir=args.cache_dir
            )
            if args.amr_model_name_or_path is not None:                 # Pretrained AMR Encoder
                self.model.amr_model =amr_model_type.from_pretrained(
                    args.amr_model_name_or_path, cache_dir=args.cache_dir, config=config, add_pooling_layer=True,
                )
            else:
                self.model.amr_model = amr_model_type.from_pretrained(
                    args.model_name_or_path, cache_dir=args.cache_dir, config=config, add_pooling_layer=True,
                )
        else:
            self.logger.info("Model name or path is not provided")
            exit()

        self.model.sent_model.resize_token_embeddings(len(self.tokenizer))
        self.model.amr_model.resize_token_embeddings(len(self.tokenizer))

        self.val_metric = (
            self.default_val_metric if self.hparams.val_metric is None else self.hparams.val_metric
        )
        self.step_count = 0
        self.val_count = -1
        self.saved_dummy = False
        self.cross_ratio = args.cross_ratio
        self.use_amr_val = args.use_amr_val
        self.sem_mlm = args.sem_mlm
        self.use_rel = args.use_rel
        self.recon_amr = args.recon_amr
        
        if self.use_amr_val:
            print("Using cross loss for validition!!!")
        self.use_amr = True if args.use_amr_val else args.use_amr
        # self.sim_loss_fn = ContrastiveSimilarity(tau=self.contrastive_tau, reduction='mean', eps=1e-6, cos_sim=False)
        self.sim_loss_fn = ContrastiveSimilarity(tau=self.contrastive_tau, reduction='mean', eps=1e-6, cos_sim=True)
        self.use_cos_sim = args.use_cos_sim
        if self.use_cos_sim:
            self.sim_loss_fn = torch.nn.CosineSimilarity()
        
        speaker_idxs = []
        for idx in range(32):         # ma
            speaker_idxs += self.tokenizer.convert_tokens_to_ids([f"speaker{idx+1}"])
        self.speaker_idxs = {v: idx for idx, v in enumerate(speaker_idxs)}
        self.start_speaker_idx = 50485
        print('speaker Ids:', self.speaker_idxs)

    def setup(self, stage=None):
        if stage == "fit":
            num_devices = max(1, self.hparams.gpus)
            effective_batch_size = (
                self.hparams.per_gpu_train_batch_size
                * self.hparams.accumulate_grad_batches
                * num_devices
            )
            print(f"Effective batch size: {effective_batch_size}")
            if self.hparams.max_steps <= 0:
                self.total_steps = (
                    self.hparams.train_dataset_size / effective_batch_size
                ) * self.hparams.max_epochs
            else:
                self.total_steps = self.hparams.max_steps
                '''
                effective_epochs = self.hparams.max_steps / (
                    self.hparams.train_dataset_size / effective_batch_size
                )
                print(f"Effective training epoches: {effective_epochs}")
                '''
    def modify_config(self, args, config, modified_params=("dropout")):
        for p in modified_params:
            if getattr(args, p, None):
                assert hasattr(config, p), f"model config doesn't have a `{p}` attribute"
                setattr(config, p, getattr(args, p))
                print("Manually set:", p, getattr(args, p))
            else:
                print("Args don't have:", p)

    def configure_optimizers(self):
        no_decay = ["bias", "LayerNorm.weight"]
        optimizer_grouped_parameters = [
            {
                "params": [
                    p
                    for n, p in self.model.sent_model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.sent_model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.hparams.learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.amr_model.named_parameters()
                    if not any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": self.hparams.weight_decay,
                "lr": self.hparams.amr_learning_rate,
            },
            {
                "params": [
                    p
                    for n, p in self.model.amr_model.named_parameters()
                    if any(nd in n for nd in no_decay) and p.requires_grad
                ],
                "weight_decay": 0.0,
                "lr": self.hparams.amr_learning_rate,
            },
        ]
        optimizer = AdamW(
            optimizer_grouped_parameters,
            lr=self.hparams.learning_rate,
            eps=self.hparams.adam_epsilon,
        )
        scheduler = self.get_lr_scheduler(optimizer)
        return [optimizer], [scheduler]

    def get_lr_scheduler(self, optimizer):
        get_schedule_func = arg_to_scheduler[self.hparams.lr_scheduler]
        if self.hparams.lr_scheduler == "constant":
            scheduler = get_schedule_func(optimizer, num_warmup_steps=self.hparams.warmup_steps)
        elif self.hparams.lr_scheduler == "cosine_w_restarts":
            scheduler = get_schedule_func(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps,
                num_cycles=self.hparams.max_epochs,
            )
        else:
            scheduler = get_schedule_func(
                optimizer,
                num_warmup_steps=self.hparams.warmup_steps,
                num_training_steps=self.total_steps,
            )
        # print("scheduler total steps:", self.total_steps)
        scheduler = {"scheduler": scheduler, "interval": "step", "frequency": 1}
        return scheduler

    @pl.utilities.rank_zero_only
    def on_save_checkpoint(self, checkpoint: Dict[str, Any]) -> None:
        save_path = Path(self.hparams.output_dir).joinpath(f"best_tfmr_{self.val_count}")
        self.model.config.save_step = self.val_count
        self.model.sent_model.save_pretrained(save_path)
        self.tokenizer.save_pretrained(save_path)

    def _step(self, batch: dict) -> Tuple:
        if self.sem_mlm:
            masked_sent_inputs, sent_labels = mask_tokens_sem(
                batch["input_ids"], batch["input_sem"], self.tokenizer, self.hparams.mlm_probability, self.start_speaker_idx,
            )
        else:
            masked_sent_inputs, sent_labels = mask_tokens(
                batch["input_ids"], self.tokenizer, self.hparams.mlm_probability, self.start_speaker_idx,
            )

        amr_inputs = batch["amr_ids"] if self.use_amr else None
        joint_inputs = batch["joint_ids"] if self.recon_amr else None
        joint_attn_mask = batch["casual_mask"] if self.recon_amr else None
        joint_labels = batch["joint_labels"] if self.recon_amr else None
        
        rel_mask = batch["rel_mask"] if self.use_rel else None
        rel_label = batch["rel_label"] if self.use_rel else None
        
        if not self.saved_dummy:
            save_dummy_batch(batch, masked_sent_inputs, self.tokenizer, self.hparams.output_dir)
            self.saved_dummy = True
            for k, v in batch.items():
                print(k, v.size())
                
        outputs = self.model(
            sent_input_ids=masked_sent_inputs,
            ori_sent_input_ids=batch["input_ids"],
            amr_input_ids=amr_inputs,
            labels=sent_labels,
            rel_mask=rel_mask,
            rel_labels=rel_label,
            joint_input_ids=joint_inputs,
            joint_attention_mask=joint_attn_mask,
            joint_labels=joint_labels,
        )
        
        return outputs

    def training_step(self, batch, batch_idx) -> Dict:
        outputs = self._step(batch)
        sent_loss = outputs[0]
        if not self.use_amr:
            self.log("train_loss", outputs[0].item(), prog_bar=True)
            self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0], prog_bar=True)
            # exit()
            return {
                "loss": sent_loss
            }
        else:
            sent_vec, amr_vec = outputs[1], outputs[2]
            sent_vec = torch.cat(GatherLayer.apply(sent_vec), dim=0)    # [N_gpu * Bsz, H]
            amr_vec = torch.cat(GatherLayer.apply(amr_vec), dim=0)      # [N_gpu * Bsz, H]
            if self.use_cos_sim:
                cross_loss = self.cross_ratio * (1.0 - self.sim_loss_fn(sent_vec, amr_vec).mean())
            else:
                cross_loss = self.cross_ratio * self.sim_loss_fn(sent_vec, amr_vec)

            self.log("cross_loss", cross_loss.item(), prog_bar=True)
            self.log("sent_loss", outputs[0].item(), prog_bar=True)
            self.log("lr", self.trainer.lr_schedulers[0]["scheduler"].get_lr()[0], prog_bar=True)
            return {
                "loss":sent_loss + cross_loss
            }

    def training_step_end(self, *args, **kwargs):
        return super().training_step_end(*args, **kwargs)

    def training_epoch_end(self, outputs, prefix="train") -> Dict:
        losses = {k: torch.stack([x[k] for x in outputs]).mean().item() for k in self.loss_names}
        self.metrics["training"].append(losses)

    def validation_step(self, batch, batch_idx) -> Dict:
        outputs = self._step(batch)                                     # 
        sent_loss = outputs[0]
        if not self.use_amr_val:
            return {
                "loss": sent_loss.detach()
            }
        else:
            sent_vec, amr_vec = outputs[1], outputs[2]
            sent_vec = torch.cat(GatherLayer.apply(sent_vec), dim=0)    # [N_gpu * Bsz, H]
            amr_vec = torch.cat(GatherLayer.apply(amr_vec), dim=0)      # [N_gpu * Bsz, H]
            if self.use_cos_sim:
                cross_loss = self.cross_ratio * (1.0 - self.sim_loss_fn(sent_vec, amr_vec).mean())
            else:
                cross_loss = self.cross_ratio * self.sim_loss_fn(sent_vec, amr_vec)

            self.log("cross_loss", cross_loss.item(), prog_bar=True)
            self.log("sent_loss", sent_loss.item(), prog_bar=True)

            return {
                "loss": (sent_loss + cross_loss).detach()
            }

    def validation_epoch_end(self, outputs, prefix="val") -> Dict:
        self.val_count += 1
        outputs = self.all_gather(outputs)
        # print('Gathered outputs', outputs)
        losses = {k: torch.stack([x[k] for x in outputs]).mean().detach() for k in self.loss_names}
        loss = losses["loss"]
        all_metrics = {f"{prefix}_avg_{k}": x for k, x in losses.items()}
        all_metrics["val_count"] = self.val_count
        self.metrics[prefix].append(all_metrics)
        self.log_dict(all_metrics, sync_dist=True)
        return {
            "log": all_metrics,
            f"{prefix}_loss": loss,
        }

    def test_step(self, batch, batch_idx):
        outputs = self._step(batch)                                     # 
        sent_loss = outputs[0]
        if not self.use_amr_val:
            return {
                "loss": sent_loss.detach()
            }
        else:
            sent_vec, amr_vec = outputs[1], outputs[2]
            sent_vec = torch.cat(GatherLayer.apply(sent_vec), dim=0)    # [N_gpu * Bsz, H]
            amr_vec = torch.cat(GatherLayer.apply(amr_vec), dim=0)      # [N_gpu * Bsz, H]
            if self.use_cos_sim:
                cross_loss = self.cross_ratio * (1.0 - self.sim_loss_fn(sent_vec, amr_vec).mean())
            else:
                cross_loss = self.cross_ratio * self.sim_loss_fn(sent_vec, amr_vec)
            return {
                "loss": (sent_loss + cross_loss).detach()
            }

    def test_epoch_end(self, outputs):
        return self.validation_epoch_end(outputs, prefix="test")
