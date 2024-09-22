
import torch
from torch import nn
from joblib import dump, load
from sklearn.preprocessing import MinMaxScaler, StandardScaler
import numpy as np

from pytorch_msssim import SSIM


class SSIMLoss(nn.Module):
    """
        refer to: https://github.com/VainF/pytorch-msssim
    """
    def __init__(self, data_range=1, size_average=True, channels=17):
        super().__init__()
        self.ssim_module = SSIM(
            data_range=data_range,
            size_average=size_average,
            channel=channels
        )
    def forward(self, x, y):
        return 1 - self.ssim_module(
            x.reshape(-1, *x.shape[2:]),
            y.reshape(-1, *y.shape[2:])
        ).squeeze(0)






class Scaler:
    """Helper class for normalization"""
    def __init__(self, save_filename, mode="train", s_type="normalize"):
        self.f_name = save_filename
        self.mode = mode
        self.s_type = s_type
        self.scaler = self._get_scaler()
    
    def _get_scaler(self):
        if self.mode=="test":
            return load(self.f_name)
        else:
            return MinMaxScaler() if self.s_type=="normalize" else StandardScaler()
    
    def _fit(self, sequences_matrix):
        x_columns = range(0,34,2)
        y_columns = range(1,34,2)
        x_max = max(np.max(sequences_matrix, axis=0)[x_columns])
        y_max = max(np.max(sequences_matrix, axis=0)[y_columns])
        x_min = min(np.min(sequences_matrix, axis=0)[x_columns])
        y_min = min(np.min(sequences_matrix, axis=0)[y_columns])

        min_max_matrix = np.vstack(
            [np.array([x_max, y_max]*17).reshape(1,-1),
            np.array([x_min, y_min]*17).reshape(1,-1)]
        )
        _ = self.scaler.fit_transform(min_max_matrix)
        
    def fit(self, sequences_matrix):
        # scaled_sequences_matrix = self.scaler.fit_transform(sequences_matrix)
        self._fit(sequences_matrix)
        scaled_sequences_matrix = self.transform(sequences_matrix)
        # Saving the scaler so it can be loaded later during testing
        dump(self.scaler, self.f_name)

        return scaled_sequences_matrix
    
    def transform(self, sequences_matrix):
        return self.scaler.transform(sequences_matrix)
def un_normalize_joints(args, predictions):
    # print(predictions.shape)
    predictions = predictions.cpu().numpy()
    scaler = Scaler(args["config"]["data"]["s_fname"], mode="test")
    # Expects a 2d Array
    # predictions 
    batch_size, seq_length, n_joints_2 = predictions.shape
    un_normalizedpredictions = scaler.scaler.inverse_transform(predictions.reshape(-1, 34))
    un_normalizedpredictions = un_normalizedpredictions.reshape(batch_size, seq_length, n_joints_2)
    un_normalizedpredictions = torch.from_numpy(un_normalizedpredictions)
    return un_normalizedpredictions.reshape(*predictions.shape[0:2], 17 ,2)

def unravel_index(indices, shape):
    
    r"""Converts a tensor of flat indices into a tensor of coordinate vectors.
    This is a `torch` implementation of `numpy.unravel_index`.
    Args:
        indices: A tensor of flat indices, (*,).
        shape: The target shape.
    Returns:
        The unraveled coordinates, (*, D).
    Taken from: https://github.com/francois-rozet/torchist/blob/master/torchist/__init__.py
    """

    shape = indices.new_tensor(shape + (1,))
    coefs = shape[1:].flipud().cumprod(dim=0).flipud()

    return torch.div(indices[..., None], coefs, rounding_mode='trunc') % shape[:-1]

def convert_heatmaps_to_skelton(x, image_size, heatmap_size):
    max_inidicies = torch.argmax(torch.flatten(x, start_dim=-2), dim=-1)
    x_skelton = unravel_index(max_inidicies, (heatmap_size[0], heatmap_size[1]))
    feat_stride = [image_size[0] / heatmap_size[0], image_size[1] / heatmap_size[1]]
    feat_stride = torch.tensor(feat_stride).reshape(*[1]*(x_skelton.dim() -1 ), -1).type_as(x)
    return x_skelton * feat_stride