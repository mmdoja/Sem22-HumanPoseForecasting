import torch
from torch.nn import functional as F

def mpjpe(predicted, target):
    """
    taken from:
    https://github.com/Arthur151/SOTA-on-monocular-3D-pose-and-shape-estimation/blob/master/evaluation_matrix.py
    Mean per-joint position error (i.e. mean Euclidean distance),
    often referred to as "Protocol #1" in many papers.
    """
    
    assert predicted.shape == target.shape
    return torch.mean(torch.norm(predicted - target, dim=-1))

def PCK(predicted, target, threshold=0.2):
    """
    Percentage of correct keypoints (PCK) metric.
    As described in https://arxiv.org/pdf/2004.06366.pdf
    """

    assert predicted.shape == target.shape
    # predicted, target = reshape_util(predicted, target)
    error = torch.norm(predicted - target, dim=-1)
    # return error
    # For keypoints names and indices:
    # https://github.com/una-dinosauria/3d-pose-baseline/blob/master/src/data_utils.py
    rhip_idx = 1    # Right Hip
    lshoul_idx = 11  # Left shoulder

    rhip = target[:, :, rhip_idx]
    lshoul = target[:, :, lshoul_idx]
    torso_diameter = torch.norm(rhip - lshoul, dim=-1)

    torso_diameter = torso_diameter.unsqueeze(2)
    return torch.mean((error <= threshold * torso_diameter).float()) * 100


def evaluate(predicted, target):
    eval_metrics = {
        "MSE": F.mse_loss,
        "MAE": F.l1_loss,
        "MPJPE": mpjpe,
        "PCK": PCK,
    }
    eval_results = {}
    for k,v in eval_metrics.items():
        eval_results[k] = v(predicted, target)
    
    return eval_results