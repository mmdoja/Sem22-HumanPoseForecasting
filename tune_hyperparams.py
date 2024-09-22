import argparse
import os
import sys
import gc
import functools

import numpy as np
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import wandb

from utils import get_data_module, prepare_config, \
    get_lit_model, get_model
import constants
from configs import HYPERPARAM_CONFIGS



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
    parser.add_argument("--num_runs", type=int,
                        default=10, help="number of trials")
    parser.add_argument(
        "--optimizer", type=str, default=constants.OPTIMIZER, help="Optimizer"
    )
    parser.add_argument("--loss", type=str,
                        default=constants.LOSS, choices=[
                            "MSELoss",
                            "L1Loss",
                            "SSIMLoss",
                        ], help="Loss Function")
    
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--accelerator", type=str, default=ACCELERATOR, help="accelerator")
    # parser.add_argument("--devices", type=int, default=None, help="number of devices")
    parser.add_argument("--gpus_per_trail", type=int, default=1, help="number of gpus per trail")
    parser.add_argument(
        "--num_workers",
        type=int,
        default=2,
        help="Number of workers for dataloaders"
    )
    parser.add_argument(
        "--pin_memory",
        type=str,
        default=constants.PINMEMORY,
        help="pin memory for dataloader")
    
    parser.add_argument(
        "--sweep_id",
        type=str,
        default=None,
        help="W and b Sweep id")
    
    parser.add_argument(
        "--project_name",
        type=str,
        default="CUDALAB",
        help="W and b Sweep id")
    return parser


def train_fun(config=None, args=None):
    with wandb.init(config=config, project=args["project_name"]):
        # global args

        np.random.seed(constants.SEED)
        torch.manual_seed(constants.SEED)
        pl.seed_everything(constants.SEED, workers=True)
        # print(type(config), config)
        config = wandb.config
        for k,v in config.items():
            if k in ["encoder_layers", "enc_layers_ch"]:
                args["config"]["model"][k] = [v]
            if k in ["lr", "loss"]:
                args[k] = v
            elif k == "batch_size":
                args["config"]["data"][k] = v
            else:
                args["config"]["model"][k] = v
        data_module = get_data_module(args)
        model = get_model(args)
        lit_model = get_lit_model(model, args)
        logdir = f'{constants.WORKING_DIR}/{constants.LOG_DIR}/' \
            + f'{args["config"]["data"]["dataset"]}/' \
            + f'{args["config"]["model"]["name"]}/' \
            + f'{args["exp_name"]}'
        logger = pl_loggers.WandbLogger(
            project=args["project_name"],
            name=args["exp_name"],
            # log_model="all",
            save_dir=logdir,
            job_type="train",
            log_model=False,
        )
        trainer = pl.Trainer(
            deterministic=True,
            accelerator=args["accelerator"],
            devices=args["gpus_per_trail"],
            max_epochs=args["num_epochs"],
            # callbacks=callbacks,
            logger=logger,
            log_every_n_steps=constants.LOG_STEPS,
        )
        trainer.fit(lit_model, data_module)

        del lit_model
        del logger
        del data_module
        del trainer
        gc.collect()
        torch.cuda.empty_cache()


def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)
    args = prepare_config(args)
    if not args["sweep_id"]:
        sweep_id = wandb.sweep(HYPERPARAM_CONFIGS[args['config']['model']['name']], project=args["project_name"])
    else:
        sweep_id = args["sweep_id"]
    print(f"sweep_id = {sweep_id}")
    training_function = functools.partial(train_fun, args=args)
    wandb.agent(sweep_id, function=training_function, count=args["num_runs"])

if __name__ == "__main__":
    main()