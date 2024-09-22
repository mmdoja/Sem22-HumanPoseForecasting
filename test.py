import argparse
import os
import pathlib

import numpy as np
import matplotlib.pyplot as plt
import pytorch_lightning as pl
from pytorch_lightning import loggers as pl_loggers
import torch
import wandb

from lit_models import LitModule
from lit_models import un_normalize_joints, convert_heatmaps_to_skelton, evaluate
from utils import setup_data_and_model_from_args
import constants
from visualization import plot_pred_2d

NUM_AVAIL_CPUS = len(os.sched_getaffinity(0))
NUM_AVAIL_GPUS = torch.cuda.device_count()
if NUM_AVAIL_GPUS:
    ACCELERATOR = "gpu"
else:
    ACCELERATOR = None
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS
DEFAULT_NUM_WORKERS = NUM_AVAIL_CPUS // NUM_AVAIL_GPUS if NUM_AVAIL_GPUS else DEFAULT_NUM_WORKERS

np.random.seed(constants.SEED)
torch.manual_seed(constants.SEED)
pl.seed_everything(constants.SEED, workers=True)

def _setup_parser():
    parser = argparse.ArgumentParser()
    parser.add_argument("--config", type=str, help="JSON config file")
    parser.add_argument(
        "--checkpoint", type=str, help="path of the model checkpoint"
    )
    parser.add_argument(
        "--optimizer", type=str, default=constants.OPTIMIZER, help="Optimizer"
    )
    parser.add_argument(
        "--wandb",
        action="store_true",
        default=False,
        help="If passed, logs experiment results to Weights & Biases",
    )
    parser.add_argument(
        "--save_preds",
        action="store_true",
        default=False,
        help="If passed predictions are saved as a numpy array",
    )

    parser.add_argument(
        "--save_visuals",
        action="store_true",
        default=False,
        help="If passed 10 random result images are saved",
    )

    parser.add_argument(
        "--lr", type=float, default=constants.LR, help="Learning Rate"
    )
    parser.add_argument("--loss", type=str,
                        default=constants.LOSS, choices=[
                            "MSELoss",
                            "L1Loss",
                            "SSIMLoss"
                        ], help="Loss Function")
    parser.add_argument("--exp_name", type=str, help="experiment name")
    parser.add_argument("--accelerator", type=str, default=ACCELERATOR, help="accelerator")
    parser.add_argument("--devices", type=int, default=None, help="number of gpu devices")
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
        help="W and b project name")
    
    parser.add_argument(
        "--wandb_id",
        type=str,
        default=None,
        help="W and b run id to log in the same run")

    return parser

def run_test(args):
    data_module, lit_model, args = setup_data_and_model_from_args(args)
    checkpoint_path = args["checkpoint"]
    model = LitModule.load_from_checkpoint(checkpoint_path=checkpoint_path, args=args, model=lit_model.model, strict=False)
    model.eval()
    data_module.setup()
    test_loader = data_module.val_dataloader()
    trainer = pl.Trainer(
        accelerator='gpu',
        devices=1
    )
    print(trainer.test(model, test_loader))
    predictions = trainer.predict(model, test_loader)
    predictions = torch.vstack(predictions)
    # scaler = data_module.data_val.
    if "Heatmaps" in args["config"]["model"]["name"]:
        predictions = convert_heatmaps_to_skelton(predictions, (1002, 1000), (64, 64))
    else:
        predictions = un_normalize_joints(args, predictions)
    
    # The shape will be num_examples, 30, 17, 2
    # 30 (0:10-> seeds, 10:20-> targets, 20:-> predictions)

    predictions = predictions.numpy()
    if "Heatmaps" in args["config"]["model"]["name"]:
        predictions = np.flip(predictions, -1).copy()

    if args["save_visuals"]:
        image_dir = args["save_dir"]+ "/images"
        pathlib.Path(image_dir).mkdir(parents=True, exist_ok=True)
        random_indicies = np.random.randint(len(np.random.randint(2, size=10)), size=10)
        figures = []
        for i, idx in enumerate(random_indicies):
            pred_to_plot = predictions[idx]
            seeds = pred_to_plot[0:10]
            gt = pred_to_plot[10:20]
            pred = pred_to_plot[20:]
            fig = plot_pred_2d(seeds, gt, pred)
            fig_path = f"{image_dir}/{i}.png"
            fig.savefig(fig_path, dpi=fig.dpi)
            figures.append(fig)
        
        if args["wandb"]:
            if args["wandb_id"]:
                wandb.init(
                    project=args["project_name"],
                    id=args["wandb_id"]
                )
        
            else:
                wandb.init(
                    project=args["project_name"],
                    name=args["exp_name"]
                )
            wandb.log({
                "predictions": [
                    wandb.Image(figures[i]) for i, _ in enumerate(figures)
                ]
            })
            wandb.finish()
    
    if args["save_preds"]:
        np.save(
            f"{args['save_dir']}/predictions.npy",
            predictions
        )
    return predictions

def main():
    parser = _setup_parser()
    args = parser.parse_args()
    args = vars(args)
    _ =  run_test(args)

if __name__ == "__main__":
    main()