import argparse
import logging
import os
import deepspeed
import CM2
from loguru import logger
import warnings
from CM2.load_pretrain_data import load_all_data

os.environ["WANDB_DISABLED"] = "true"
# set random seed
CM2.random_seed(42)

warnings.filterwarnings("ignore")

def parse_args():
    parser = argparse.ArgumentParser(description='CM2-mask-pretrain-ds')
    parser.add_argument('--local_rank', type=int, default=-1, help='local rank passed from distributed launcher')
    parser.add_argument("--lable_data_args", type=str, default="OpenTabs/clean_labeled_dataset", help="pretrain_data's path")
    parser.add_argument("--unlable_data_args", type=str, default="OpenTabs/clean_unlabeled_dataset", help="pretrain_data's path")
    parser.add_argument("--save_model", type=str, default="./mask_v1", help="save_model's path")
    parser.add_argument("--num_data", type=int, default=20, help="num of the pretain datasets")
    parser.add_argument("--log_path", type=str, default="./logs/mask_v1.txt", help="")

    parser.add_argument("--is_supervised", type=int, default=1, help="if take supervised CL")
    parser.add_argument("--coresize", type=int, default=10000, help="the size of coreset")
    parser.add_argument("--vocab_freeze", type=int, default=1, help="vocab_freeze")

    parser.add_argument("--num_partition", type=int, default=3, help="num_partition")
    parser.add_argument("--num_layer", type=int, default=3, help="num_layer")
    parser.add_argument("--mlm_probability", type=float, default=0.35, help="num_layer")
    parser.add_argument("--overlap_ratio", type=int, default=0.5, help="overlap_ratio")
    parser.add_argument("--hidden_dim", type=int, default=128, help="hidden_dim")
    parser.add_argument("--ffn_dim", type=int, default=256, help="ffn_dim")
    parser.add_argument("--num_attention_head", type=int, default=8, help="num_attention_head")
    parser.add_argument("--hidden_dropout_prob", type=float, default=0.1, help="hidden_dropout_prob")

    parser.add_argument("--num_epoch", type=int, default=1, help="num_epoch")
    parser.add_argument("--batch_size", type=int, default=256, help="batch_size")
    parser.add_argument("--patience", type=int, default=5, help="patience")
    parser.add_argument("--lr", type=float, default=3e-4, help="learning rate")

    parser = deepspeed.add_config_arguments(parser)
    args = parser.parse_args()
    return args

deepspeed.init_distributed()

_args = parse_args()
dev = f'cuda:{_args.local_rank}'
logger.info(f'dev:{dev}')
cal_device = dev

if "OMPI_COMM_WORLD_RANK" in os.environ:
    # mpi env
    my_rank = int(os.getenv("OMPI_COMM_WORLD_RANK"))
elif "RANK" in os.environ:
    # torch distributed env
    my_rank = int(os.getenv("RANK"))
else:
    my_rank = 0

log_level: str = os.getenv("LOG_LEVEL", "INFO").upper()

logger_config = {
    "handlers": [
        {
            "sink": _args.log_path,
            "level": log_level,
            "colorize": True,
            "format": "[rank {extra[rank]}] [{time}] [{level}] {message}",
        },
    ],
    "extra": {"rank": my_rank},
}
logger.configure(**logger_config)


trainset, valset, cat_cols, num_cols, bin_cols, data_weight = load_all_data(
    label_data_path=_args.lable_data_args,
    unlabel_data_path=_args.unlable_data_args,
    limit=_args.num_data,
)

model = CM2.build_mask_features_learner(
    cat_cols, num_cols, bin_cols,
    mlm_probability=_args.mlm_probability,
    device=cal_device,
    hidden_dropout_prob=_args.hidden_dropout_prob,
    num_attention_head=_args.num_attention_head,
    num_layer=_args.num_layer,

    vocab_freeze=True,
    pretrain_table_num=len(trainset),
)

training_arguments = {
    'num_epoch': _args.num_epoch,
    'batch_size':_args.batch_size,
    'lr':_args.lr,
    'eval_metric':'val_loss',
    'eval_less_is_better':True,
    'output_dir':_args.save_model,
    'patience':_args.patience,
    'warmup_steps':1,
    'num_workers':0,
}
logging.info(training_arguments)

trainer = CM2.train(model, trainset, valset, data_weight=None, train_method='deepspeed', cmd_args=_args, **training_arguments)
trainer.train()