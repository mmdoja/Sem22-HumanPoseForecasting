import os
import pickle
from tqdm import tqdm
from matplotlib import pyplot as plt
from joblib import dump, load
import numpy as np
from sklearn.preprocessing import MinMaxScaler, StandardScaler


def load_dataset(file_path):
    """"""
    with open(file_path, "rb") as f:
        data = pickle.load(f)
    return data

def group_dataset(data):
    """group dataset according to video id"""
    grouped_data = {}
    for d in tqdm(data, desc="grouping data into videos"):
        video_id = d['video_id']
        if video_id not in grouped_data.keys():
            grouped_data[video_id] = []
        grouped_data[video_id].append(d)
    
    return grouped_data

def create_2d_joints_sequences(grouped_data, ds_factor=8):
    sequences = []
    for video in tqdm(grouped_data.values(), desc="creating joint sequences"):
        sequence = []
        for frame in video:
            joints_2d = frame["joints_2d"]
            sequence.append(joints_2d)
        
        sequences.append(sequence[::ds_factor])
        
    return sequences

def create_training_sequences(sequences, n=20):
    """creates sequences of n frames"""
    train_sequences = []
    for sequence in tqdm(sequences, desc=f"creating sequences of size {n}"):
        train_seq = []
        for frame in sequence:
            if len(train_seq) == n:
                train_sequences.append(train_seq)
                train_seq = []
            train_seq.append(frame)
    return train_sequences

def unroll_2d_sequences(sequences):
    sequences_updated = []
    for sequence in tqdm(sequences, desc=f"unrolling joint vectors"):
        sequences_updated.append(np.array([
            joints_vec.ravel().reshape(1, -1) for joints_vec in sequence
        ]))
    return sequences_updated

def stack_sequences(sequences):
    """Stacking sequences to make a single matrix"""
    return np.vstack(sequences).squeeze()


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

def prepare_2d_data(
        data_file,
        mode="train",
        s_fname="scaler.joblib",
        s_type="normalize",
        type="joints"
):
    data = load_dataset(data_file)
    grouped_data = group_dataset(data)
    sequences = create_2d_joints_sequences(grouped_data)
    train_sequences = create_training_sequences(sequences)
    train_sequences = unroll_2d_sequences(train_sequences)
    seq_matrix = stack_sequences(train_sequences)
    if type == "heatmaps":
        return seq_matrix
    scaler = Scaler(s_fname, mode=mode, s_type=s_type)
    if mode == "train":
        seq_matrix = scaler.fit(seq_matrix)
    else:
        seq_matrix = scaler.transform(seq_matrix)
    
    return seq_matrix

def joints_to_heatmap(
    joints,
    sigma=2,
    image_size=np.array([1002, 1000]),
    heatmap_size=np.array([64, 64]),
    num_joints=17,
):
    """ Method for generating heatmaps from joints
    Implementation is taken from the following resource:
    https://github.com/angelvillar96/STLPose/blob/21a7841cdcdd73c857d35a6eedd696c0b1a32aaa/src/data/JointsDataset.py#L230
    """
    target = np.zeros((
        num_joints,
        heatmap_size[1],
        heatmap_size[0]
    ), dtype=np.float32) 

    tmp_size = sigma * 3
    for joint_id in range(num_joints):
        feat_stride = image_size / heatmap_size
        mu_x = int(joints[joint_id][0] / feat_stride[0] + 0.5)
        mu_y = int(joints[joint_id][1] / feat_stride[1] + 0.5)
        # Check that any part of the gaussian is in-bounds
        ul = [int(mu_x - tmp_size), int(mu_y - tmp_size)]
        br = [int(mu_x + tmp_size + 1), int(mu_y + tmp_size + 1)]
        
        if ul[0] >= heatmap_size[0] or ul[1] >= heatmap_size[1] \
                or br[0] < 0 or br[1] < 0:
            # If not, just return the image as is
            continue
        # Generate gaussian
        size = 2 * tmp_size + 1
        x = np.arange(0, size, 1, np.float32)
        y = x[:, np.newaxis]
        x0 = y0 = size // 2
        # The gaussian is not normalized, we want the center value to equal 1
        g = np.exp(- ((x - x0) ** 2 + (y - y0) ** 2) / (2 * sigma ** 2))

        # Usable gaussian range
        g_x = max(0, -ul[0]), min(br[0], heatmap_size[0]) - ul[0]
        g_y = max(0, -ul[1]), min(br[1], heatmap_size[1]) - ul[1]
        # Image range
        img_x = max(0, ul[0]), min(br[0], heatmap_size[0])
        img_y = max(0, ul[1]), min(br[1], heatmap_size[1])

        target[joint_id][img_y[0]:img_y[1], img_x[0]:img_x[1]] = \
        g[g_y[0]:g_y[1], g_x[0]:g_x[1]]
    
    return target

def generate_heatmaps(sequence):
    heatmaps = []
    for joints in sequence:
        heatmaps.append(
           np.expand_dims(joints_to_heatmap(joints.reshape(17, 2)), 0) 
        )
    return np.vstack(heatmaps)

