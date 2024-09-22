from matplotlib import pyplot as plt
from torchvision.utils import draw_keypoints
import torch

plt.style.use("seaborn")

def plot_keypoint(kpts, image_dim=(3, 1002, 1000), color="red"):
    H36M_SKELETON = [[0, 1], [1, 2], [2, 3], [0, 4], [4, 5], [5, 6], [0, 7], [7, 8], [8, 9], [9, 10],
                     [8, 11], [11, 12], [12, 13], [8, 14], [14, 15], [15, 16]]  # Connectivity between kpts
    kpts = torch.from_numpy(kpts).unsqueeze(0)
    image = torch.zeros(*image_dim).type(torch.uint8)
    kpt_img = draw_keypoints(
        image=image,
        keypoints=kpts,
        colors="red",
        radius=10,
        width=8,
        connectivity=H36M_SKELETON
    )
    return kpt_img

def plot_pred_2d(
    seeds,
    gt,
    pred,
    image_dim=(3, 1002, 1000),
    color="red",
):
    labels = ["Seeds", "Ground Truths", "Predictions"]
    frames_list = [seeds, gt, pred]
    fig = plt.figure(figsize=(20, 10))
    fig.suptitle('Random Example')
    subfigs = fig.subfigures(nrows=3, ncols=1, hspace=0, wspace=0)
    for row, subfig in enumerate(subfigs):
        subfig.suptitle(f'{labels[row]}', fontsize=30)
        axs = subfig.subplots(nrows=1, ncols=10)
        for col, ax in enumerate(axs):
            ax.imshow(plot_keypoint(
                frames_list[row][col],
                image_dim=image_dim,
                color=color
            ).permute(1,2,0))
            ax.axis("off")
    
    plt.tight_layout()
    plt.subplots_adjust(wspace=0, hspace=0)
    
    return fig