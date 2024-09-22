import argparse
import os
import sys

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import pathlib

from utils import setup_data_and_model_from_args, get_callbacks
import constants

np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
pl.seed_everything(constants.SEED, workers=True)

NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
if NUM_AVAIL_GPUS:
    ACCELERATOR = "gpu"
else:
    ACCELERATOR = None
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS


def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument("--num_epochs", type=int,
                        default=100, help="number of epochs")
    parser.add_argument(
        "--load_checkpoint", type=str, default=None, help="If passed, loads a model from the provided path."
    )
    parser.add_argument(
        "--optimizer", type=str, default=constants.OPTIMIZER, help="Optimizer"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases. Otherwise logs only to local Tensorboard.",
    )
    parser.add_argument(
        "--lr", type=float, default=constants.LR, help="Learning Rate"
    )
    parser.add_argument("--loss", type=str,
                        default=constants.LOSS, choices=[
                            "MSELoss",
                            "L1Loss",
                            "SSIMLoss",
                        ], help="Loss Function")
    # Schedulers parameters
    parser.add_argument("--one_cycle_max_lr", type=float, default=None)
    parser.add_argument("--one_cycle_total_steps", type=int, default=constants.ONE_CYCLE_TOTAL_STEPS)
    # Callbacks
    parser.add_argument("--use_es", action="store_true",
                        help="use early stopping or not")
    parser.add_argument("--n_checkpoints", type=int,
                        default=constants.N_CHECKPOINTS, help="number of checkpoints")
    parser.add_argument("--patience", type=int,
                        default=constants.PATIENCE, help="patience for early stopping/checkpointing")
    parser.add_argument("--mode", type=str,
                        default="max", choices=[
                            "min",
                            "max"
                        ], help="mode for early stopping/checkpointing")
    parser.add_argument("--monitor", type=str,
                        default="val_PCK", choices=[
                            "val_loss",
                            "train_loss",
                            "val_PCK",
                        ], help="monitor for early stopping/checkpointing")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--accelerator", type=str, default=ACCELERATOR, help="accelerator")
    parser.add_argument("--devices", type=int, default=None, help="number of devices")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=DEFAULT_NUM_WORKERS,
        help="Number of workers for dataloaders"
    )
    parser.add_argument(
        "--pin_memory",
        type=str,
        default=constants.PINMEMORY,
        help="pin memory for dataloader")
    
    parser.add_argument(
        "--project_name",
        type=str,
        default="CUDALAB",
        help="W and b proj name")

    return parser

def _run_experiment(args):
    data_module, lit_model, args = setup_data_and_model_from_args(args)
    callbacks = get_callbacks(args)
    logdir = f'{constants.WORKING_DIR}/{constants.LOG_DIR}/' \
        + f'{args["config"]["data"]["dataset"]}/' \
        + f'{args["config"]["model"]["name"]}/' \
        + f'{args["exp_name"]}'
    
    # logger = pl_loggers.TensorBoardLogger(save_dir=logdir)
    
    if args["wandb"]:
        pathlib.Path(logdir).mkdir(parents=True, exist_ok=True)
        logger = pl_loggers.WandbLogger(
            project=args["project_name"],
            name=args["exp_name"],
            # log_model="all",
            save_dir=logdir,
            job_type="train",
            log_model=False,

        )
        # logger.watch(lit_model, log_freq=max(100, constants.LOG_STEPS))
        logger.log_hyperparams(args["config"]["model"])
    else: 
        logger = pl_loggers.TensorBoardLogger(save_dir=logdir)
    trainer = pl.Trainer(
        deterministic=True,
        accelerator=args["accelerator"],
        devices=args["devices"] if args["devices"] else constants.DEVICES,
        max_epochs=args["num_epochs"],
        callbacks=callbacks,
        logger=logger,
        log_every_n_steps=constants.LOG_STEPS,
    )
    trainer.fit(lit_model, data_module, ckpt_path=args["load_checkpoint"])
    best_model_path = callbacks[0].best_model_path
    print("best_model_path", best_model_path)
    print("args:", args)


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)
    _run_experiment(args)

if __name__ == "__main__":
    main()
