"""Lightning Model."""
import pytorch_lightning as pl
import torch

from .metrics import evaluate
from .utils import un_normalize_joints, convert_heatmaps_to_skelton, SSIMLoss

class LitModule(pl.LightningModule):
    """
    Generic PyTorch-Lightning class that must be initialized with a PyTorch module.
    """
    def __init__(self, model, args):
        super().__init__()
        self.model = model
        self.args = args

        optimizer = self.args.get("optimizer")
        self.optimizer_class = getattr(torch.optim, optimizer)

        self.lr = self.args.get("lr")

        loss = self.args.get("loss")
        if loss == "SSIMLoss":
            self.loss_fn = SSIMLoss()
        # print("loss func", loss)
        else:
            self.loss_fn = getattr(torch.nn, loss)()

        self.one_cycle_max_lr = self.args.get("one_cycle_max_lr", None)
        self.one_cycle_total_steps = self.args.get("one_cycle_total_steps")
    
    def configure_optimizers(self):
        optimizer = self.optimizer_class(self.parameters(), lr=self.lr)
        if self.one_cycle_max_lr is None:
            return optimizer
        scheduler = torch.optim.lr_scheduler.OneCycleLR(
            optimizer=optimizer, max_lr=self.one_cycle_max_lr, total_steps=self.one_cycle_total_steps
        )
        return {"optimizer": optimizer, "lr_scheduler": scheduler}
    
    def forward(self, x):
        return self.model(x)
    
    def predict_step(self, x, batch_idx):
        teacher_forcing_ratio = 0
        if hasattr(self.model, 'teacher_forcing_ratio'):
            teacher_forcing_ratio = self.model.teacher_forcing_ratio
            self.model.teacher_forcing_ratio = 0
        preds = self.model(x)
        if teacher_forcing_ratio:
            self.model.teacher_forcing_ratio = teacher_forcing_ratio
        return torch.vstack([
            x.reshape(-1, *x.shape[2:]),
            preds.reshape(-1, *preds.shape[2:])
        ]).reshape(x.shape[0], -1, *x.shape[2:])
    
    def _run_on_batch(self, batch):
        pred = self(batch)
        y = batch[:, self.model.n_seeds:, :]
        loss = self.loss_fn(pred, y)
        
        return pred, y, loss
    
    def training_step(self, batch, batch_idx):
        pred, y, loss = self._run_on_batch(batch)
        # self.log("train_loss", loss, batch_size=pred.shape[0], on_step=True, on_epoch=True)
        logs = {}
        logs["loss"] = loss

        for k,v in logs.items():
            self.log(f"train_{k}", v, batch_size=pred.shape[0], on_step=True, on_epoch=True, prog_bar=False)
        
        return loss
    
    def validation_step(self, batch, batch_idx):
        teacher_forcing_ratio = 0
        if hasattr(self.model, 'teacher_forcing_ratio'):
            teacher_forcing_ratio = self.model.teacher_forcing_ratio
            self.model.teacher_forcing_ratio = 0
        pred, y, loss = self._run_on_batch(batch)
        if teacher_forcing_ratio:
            self.model.teacher_forcing_ratio = teacher_forcing_ratio

        logs = {}
        logs["loss"] = loss
        for k,v in logs.items():
            prog_bar = True if k == "loss" else False
            self.log(f"val_{k}", v, batch_size=pred.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        
        if "Heatmaps" in self.args["config"]["model"]["name"]:
            pred = convert_heatmaps_to_skelton(pred, (1002, 1000), (64, 64))
            y = convert_heatmaps_to_skelton(y, (1002, 1000), (64, 64))
        else:
            pred = un_normalize_joints(self.args, pred)
            y = un_normalize_joints(self.args, y)
        
        evaluation_results = evaluate(pred, y)
        for k,v in evaluation_results.items():
            self.log(f"val_{k}", v, batch_size=pred.shape[0], on_step=False, on_epoch=True, prog_bar=False)

        
        return loss
    
    def test_step(self, batch, batch_idx):
        teacher_forcing_ratio = 0
        if hasattr(self.model, 'teacher_forcing_ratio'):
            teacher_forcing_ratio = self.model.teacher_forcing_ratio
            self.model.teacher_forcing_ratio = 0
        pred, y, loss = self._run_on_batch(batch)
        if teacher_forcing_ratio:
            self.model.teacher_forcing_ratio = teacher_forcing_ratio

        logs = {}
        logs["loss"] = loss
        for k,v in logs.items():
            prog_bar = True if k == "loss" else False
            self.log(f"test_{k}", v, batch_size=pred.shape[0], on_step=False, on_epoch=True, prog_bar=False)
        
        if "Heatmaps" in self.args["config"]["model"]["name"]:
            pred = convert_heatmaps_to_skelton(pred, (1002, 1000), (64, 64))
            y = convert_heatmaps_to_skelton(y, (1002, 1000), (64, 64))
        else:
            pred = un_normalize_joints(self.args, pred)
            y = un_normalize_joints(self.args, y)
        
        evaluation_results = evaluate(pred, y)
        for k,v in evaluation_results.items():
            self.log(f"test_{k}", v, batch_size=pred.shape[0], on_step=False, on_epoch=True, prog_bar=False)

        
        return loss