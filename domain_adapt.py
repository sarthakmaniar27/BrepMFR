# -*- coding: utf-8 -*-
import argparse
import pathlib
import time
import torch
from pytorch_lightning import Trainer
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.loggers import TensorBoardLogger

from models.transfer_model import DomainAdapt
from data.dataset import TransferDataset
from models.modules.utils.macro import *

torch.backends.cudnn.enabled = True
torch.backends.cudnn.benchmark = True

parser = argparse.ArgumentParser("BrepSeg Network model")
parser.add_argument("traintest", choices=("train", "test"), help="Whether to train or test")
parser.add_argument("--num_classes", type=int, default=25, help="Number of features")
parser.add_argument("--open_set", type=int, default=0)
parser.add_argument("--dataset", choices=("cadsynth", "transfer"), default="transfer", help="Dataset to train on")
parser.add_argument("--source_path", type=str, help="Path to source_dataset")
parser.add_argument("--target_path", type=str, help="Path to target_dataset")
parser.add_argument("--batch_size", type=int, default=64, help="Batch size")
parser.add_argument(
    "--num_workers",
    type=int,
    default=0,
    help="Number of workers for the dataloader. NOTE: set this to 0 on Windows.",
)
parser.add_argument(
    "--pre_train",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for pre-trained model",
)
parser.add_argument(
    "--checkpoint",
    type=str,
    default=None,
    help="Checkpoint file to load weights from for testing",
)
parser.add_argument(
    "--experiment_name",
    type=str,
    default="BrepToSeq-segmentation",
    help="Experiment name (used to create folder inside ./results/ to save logs and checkpoints)",
)
# Transformer module default parameters
parser.add_argument("--dropout", type=float, default=0.3)
parser.add_argument("--attention_dropout", type=float, default=0.3)
parser.add_argument("--act-dropout", type=float, default=0.3)
parser.add_argument("--d_model", type=int, default=512)
parser.add_argument("--dim_node", type=int, default=256)
parser.add_argument("--n_heads", type=int, default=32)
parser.add_argument("--n_layers_encode", type=int, default=8)

parser = Trainer.add_argparse_args(parser)
args = parser.parse_args()

results_path = (
    pathlib.Path(__file__).parent.joinpath("results").joinpath(args.experiment_name)
)
if not results_path.exists():
    results_path.mkdir(parents=True, exist_ok=True)

month_day = time.strftime("%m%d")
hour_min_second = time.strftime("%H%M%S")

# ModelCheckpoint:
# monitor="eval_loss" with mode="min" is correct because
# eval_loss = 1 / target_accuracy in validation_step.
# The checkpoint with lowest eval_loss has the highest target accuracy.
checkpoint_callback = ModelCheckpoint(
    monitor="eval_loss",
    mode="min",
    dirpath=str(results_path.joinpath(month_day, hour_min_second)),
    filename="best",
    save_top_k=10,
    save_last=True,
)

trainer = Trainer.from_argparse_args(
    args,
    callbacks=[checkpoint_callback],
    logger=TensorBoardLogger(
        str(results_path), name=month_day, version=hour_min_second,
    ),
    accelerator='gpu',
    devices=1,
    auto_select_gpus=True,
    gradient_clip_val=1.0,
)

if args.dataset == "transfer":
    Dataset = TransferDataset
else:
    raise ValueError("Unsupported dataset")

if args.traintest == "train":
    print(
        f"""
-----------------------------------------------------------------------------------
B-rep model feature recognition based on Transfer Learning
-----------------------------------------------------------------------------------
Logs written to results/{args.experiment_name}/{month_day}/{hour_min_second}

To monitor the logs, run:
tensorboard --logdir results/{args.experiment_name}/{month_day}/{hour_min_second}

The trained model with the best validation loss will be written to:
results/{args.experiment_name}/{month_day}/{hour_min_second}/best.ckpt
-----------------------------------------------------------------------------------
    """
    )
    model = DomainAdapt(args)
    train_data = Dataset(
        root_dir_source=args.source_path,
        root_dir_target=args.target_path,
        split="train",
        random_rotate=True,
        num_class=args.num_classes,
        open_set=args.open_set,
    )
    val_data = Dataset(
        root_dir_source=args.source_path,
        root_dir_target=args.target_path,
        split="val",
        random_rotate=False,
        num_class=args.num_classes,
        open_set=args.open_set,
    )
    train_loader = train_data.get_dataloader(
        batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers
    )
    val_loader = val_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    trainer.fit(model, train_loader, val_loader)

else:
    assert args.checkpoint is not None, "Expected the --checkpoint argument to be provided"
    test_data = Dataset(
        root_dir_source=args.source_path,
        root_dir_target=args.target_path,
        split="test",
        num_class=args.num_classes,
        open_set=args.open_set,
    )
    test_loader = test_data.get_dataloader(
        batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers
    )
    model = DomainAdapt.load_from_checkpoint(args.checkpoint)
    trainer.test(model, dataloaders=[test_loader], ckpt_path=args.checkpoint, verbose=False)
