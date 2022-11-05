# coding:utf-8
import argparse


def add_model_specific_args(parser, root_dir):
    parser.add_argument(
        "--train_data_file",
        default=None,
        type=str,
        required=True,
        help="The input training data file (a text file).",
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        required=True,
        help="The output directory where the model predictions and checkpoints will be written.",
    )
    parser.add_argument(
        "--model_type",
        type=str,
        required=True,
        help="The model architecture to be trained or fine-tuned.",
    )

    # Other parameters
    parser.add_argument(
        "--eval_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--test_data_file",
        default=None,
        type=str,
        help="An optional input evaluation data file to evaluate the perplexity on (a text file).",
    )
    parser.add_argument(
        "--model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )
    parser.add_argument(
        "--amr_model_name_or_path",
        default=None,
        type=str,
        help="The model checkpoint for weights initialization. Leave None if you want to train a model from scratch.",
    )

    parser.add_argument(
        "--mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--sem_mlm",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--use_amr",
        action="store_true",
        help="Train with masked-language modeling loss instead of language modeling.",
    )
    parser.add_argument(
        "--use_amr_val",
        action="store_true",
        help="Using amr for calculating vallidation loss.",
    )
    parser.add_argument(
        "--use_cos_sim",
        action="store_true",
        help="Using amr for calculating vallidation loss.",
    )
    parser.add_argument(
        "--no_poly",
        action="store_true",
        help="Without using poly encoder for calculating cross loss.",
    )
    parser.add_argument(
        "--use_rel",
        action="store_true",
        help="Without using relation for calculating relation prediction loss.",
    )
    parser.add_argument(
        "--recon_amr",
        action="store_true",
        help="Reconstruct AMR based on Textual inputs",
    )
    parser.add_argument(
        "--mlm_probability",
        type=float,
        default=0.15,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--cross_ratio",
        type=float,
        default=1.0,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--rel_ratio",
        type=float,
        default=1.0,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--recon_ratio",
        type=float,
        default=1.0,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--temperature",
        type=float,
        default=1.0,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--cls_labels",
        type=int,
        default=2,
        help="Ratio of tokens to mask for masked language modeling loss",
    )
    parser.add_argument(
        "--config_name",
        default=None,
        type=str,
        help="Optional pretrained config name or path if not the same as model_name_or_path. If both are None, initialize a new config.",
    )
    parser.add_argument(
        "--tokenizer_name",
        default=None,
        type=str,
        help="Optional pretrained tokenizer name or path if not the same as model_name_or_path. If both are None, initialize a new tokenizer.",
    )
    parser.add_argument(
        "--cache_dir",
        default=None,
        type=str,
        help="Optional directory to store the pre-trained models downloaded from s3 (instead of the default one)",
    )
    parser.add_argument(
        "--src_block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization.",
    )
    parser.add_argument(
        "--tgt_block_size",
        default=512,
        type=int,
        help="Optional input sequence length after tokenization.",
    )
    parser.add_argument(
        "--src_prefix",
        default="",
        type=str,
        help="Source prefix",
    )
    parser.add_argument(
        "--tgt_prefix",
        default="",
        type=str,
        help="Target prefix",
    )
    parser.add_argument(
        "--val_metric",
        default="loss",
        type=str,
        help="validation metric",
    )
    parser.add_argument("--do_train", action="store_true", help="Whether to run training.")
    parser.add_argument(
        "--do_predict", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--do_eval", action="store_true", help="Whether to run eval on the dev set."
    )
    parser.add_argument(
        "--evaluate_during_training",
        action="store_true",
        help="Run evaluation during training at each logging step.",
    )

    parser.add_argument(
        "--per_gpu_train_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--per_gpu_eval_batch_size",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--train_num_workers",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for training.",
    )
    parser.add_argument(
        "--eval_num_workers",
        default=4,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--process_num_workers",
        default=1,
        type=int,
        help="Batch size per GPU/CPU for evaluation.",
    )
    parser.add_argument(
        "--early_stopping_patience",
        type=int,
        default=0,
        help="Number of updates steps to control early stop",
    )
    parser.add_argument(
        "--lr_scheduler", default="linear", type=str, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--learning_rate", default=5e-5, type=float, help="The initial learning rate for Adam."
    )
    parser.add_argument(
        "--amr_learning_rate",
        default=5e-5,
        type=float,
        help="The initial AMR learning rate for Adam.",
    )
    parser.add_argument(
        "--weight_decay", default=0.0, type=float, help="Weight decay if we apply some."
    )
    parser.add_argument(
        "--adam_epsilon", default=1e-8, type=float, help="Epsilon for Adam optimizer."
    )
    parser.add_argument("--max_grad_norm", default=1.0, type=float, help="Max gradient norm.")
    parser.add_argument(
        "--warmup_steps", default=0, type=int, help="Linear warmup over warmup_steps."
    )
    parser.add_argument(
        "--overwrite_output_dir",
        action="store_true",
        help="Overwrite the content of the output directory",
    )
    parser.add_argument("--seed", type=int, default=42, help="random seed for initialization")
    parser.add_argument(
        "--fp16",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--shuffle",
        action="store_true",
        help="Whether to use 16-bit (mixed) precision (through NVIDIA apex) instead of 32-bit",
    )
    parser.add_argument(
        "--save_total_limit",
        type=int,
        default=None,
        help="Limit the total amount of checkpoints, delete the older checkpoints in the output_dir, does not delete by default",
    )
    parser.add_argument(
        "--save_interval",
        type=int,
        default=-1,
        help="save step interval",
    )
    parser.add_argument("--resume", action="store_true", help="Whether to continue run training.")
    return parser

