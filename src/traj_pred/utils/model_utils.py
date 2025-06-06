"""Utility functions used in NN model definitions"""
import math
from enum import Enum

import numpy as np
import torch
import torch.nn.functional as F

class ModeKeys(Enum):
    TRAIN = 1
    EVAL = 2
    PREDICT = 3


class UpdateMode(Enum):
    BATCH_FROM_PRIOR = 1
    ITERATIVE = 2
    ONLINE_BATCH = 3
    NO_UPDATE = 4


def cyclical_lr(stepsize, min_lr=3e-4, max_lr=3e-3, decay=1.0):
    # Lambda function to calculate the LR
    lr_lambda = (
        lambda it: min_lr + (max_lr - min_lr) * relative(it, stepsize) * decay**it
    )

    # Additional function to see where on the cycle we are
    def relative(it, stepsize):
        cycle = math.floor(1 + it / (2 * stepsize))
        x = abs(it / stepsize - 2 * cycle + 1)
        return max(0, (1 - x))

    return lr_lambda


def to_one_hot(labels, n_labels):
    return torch.eye(n_labels, device=labels.device)[labels]


def mutual_inf_mc(x_dist):
    dist = x_dist.__class__
    H_y = dist(probs=x_dist.probs.mean(dim=0)).entropy()
    return (H_y - x_dist.entropy().mean(dim=0)).sum()


def unpack_rnn_state(state_tuple):
    """
    Convert output of bi-directional LSTMs to batch first 
    and squeeze along feature dimension
    """
    # PyTorch returned LSTM states have 3 dims:
    # (num_layers * num_directions, batch, hidden_size)

    state = torch.cat(state_tuple, dim=0).permute(1, 0, 2)
    # Now state is (batch, 2 * num_layers * num_directions, hidden_size)

    state_size = state.size()
    return torch.reshape(state, (-1, state_size[1] * state_size[2]))


def roll_by_gather(mat: torch.Tensor, dim: int, shifts: torch.LongTensor):
    """Shift up rows of arrays by specified amount"""
    # assumes 3D array
    batch, ts, dim = mat.shape

    arange1 = (
        torch.arange(ts, device=shifts.device)
        .unsqueeze(0)
        .unsqueeze(-1)
        .expand(batch, -1, dim)
    )
    # print(arange1)
    arange2 = (arange1 - shifts[:, None, None]) % ts
    # print(arange2)
    return torch.gather(mat, 1, arange2)


def get_agent_neigh_joint_state(
    node_history_st: torch.Tensor,
    node_history_len: torch.Tensor,
    neigh_hist: torch.Tensor,
    neigh_hist_len: torch.Tensor,
    neigh_types: torch.Tensor,
):
    """
    Creates node-neighbor history pairs
    """
    # pad to have equal sequence lengths
    if neigh_hist.shape[2] < node_history_st.shape[1]:
        neigh_hist = F.pad(
            neigh_hist,
            pad=(0, 0, 0, node_history_st.shape[1] - neigh_hist.shape[2]),
            value=np.nan,
        )
    elif neigh_hist.shape[2] > node_history_st.shape[1]:
        node_history_st = F.pad(
            node_history_st,
            pad=(0, 0, 0, neigh_hist.shape[2] - node_history_st.shape[1]),
            value=np.nan,
        )
    # repeat node history length, num. of neighbor times
    node_hist_lens_for_cat = node_history_len.unsqueeze(1).expand(
        (-1, neigh_hist.shape[1])
    )
    # find minimum history for each node-neighbor pair
    joint_history_len = torch.minimum(
        neigh_hist_len, node_hist_lens_for_cat
    ).flatten()
    has_data: torch.Tensor = joint_history_len > 0
    # repeat node history, num. of neighbor times and keep those with minimum history > 0
    node_hist_for_cat = node_history_st.repeat_interleave(
        neigh_hist.shape[1], dim=0, output_size=has_data.shape[0]
    )[has_data]
    # squeeze neigbor history along num. of neigbors dim.
    neigh_hist_for_cat = neigh_hist.reshape(-1, *neigh_hist.shape[2:])[has_data]
    # history lenght and neighbor types for joint node-neigh pairs under consideration
    joint_history_len = joint_history_len[has_data]
    joint_neigh_types = neigh_types.flatten()[has_data]

    # calculate shift in timesteps
    # e.g. if node history length > neigh. history length,
    # shift node history up to only consider common timesteps
    node_shifts = joint_history_len - node_hist_lens_for_cat.flatten()[has_data]
    neigh_shifts = joint_history_len - neigh_hist_len.flatten()[has_data]
    # execute the shifts
    node_hist_for_cat = roll_by_gather(
        node_hist_for_cat, dim=1, shifts=node_shifts.to(node_hist_for_cat.device)
    )
    neigh_hist_for_cat = roll_by_gather(
        neigh_hist_for_cat, dim=1, shifts=neigh_shifts.to(neigh_hist_for_cat.device)
    )
    # concatenate node and neigh. states
    joint_history = torch.cat([neigh_hist_for_cat, node_hist_for_cat], dim=-1)

    return joint_history, joint_history_len, joint_neigh_types


def extract_subtensor_per_batch_element(tensor, indices):
    batch_idxs = torch.arange(start=0, end=len(indices))

    batch_idxs = batch_idxs[~torch.isnan(indices)]
    indices = indices[~torch.isnan(indices)]
    if indices.size == 0:
        return None
    else:
        indices = indices.long()
    if tensor.is_cuda:
        batch_idxs = batch_idxs.to(tensor.get_device())
        indices = indices.to(tensor.get_device())
    return tensor[batch_idxs, indices]

