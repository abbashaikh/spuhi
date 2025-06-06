"""Encoder modules used by MG-CVAE"""
import torch
import torch.nn as nn
import torch.nn.functional as F

from traj_pred.utils.model_utils import (
    ModeKeys,
    unpack_rnn_state
)

############################
#   Node History Encoder   #
############################
class NodeHistoryEncoder(nn.Module):
    """
    Node History Encoder using LSTM
    """
    def __init__(self, state_length, hidden_dim):
        super(NodeHistoryEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=state_length,
            hidden_size=hidden_dim,
            batch_first=True
        )

    def forward(self, hp, mode, node_hist, node_hist_len) -> torch.Tensor:
        """
        Encodes the nodes history.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_hist: Historic and current state of the node. [bs, mhl, state]
        :param node_hist_len: Number of timesteps for which data is available [bs]
        :return: Encoded node history tensor. [bs, enc_rnn_dim]
        """
        packed_input = nn.utils.rnn.pack_padded_sequence(
            node_hist, node_hist_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        output, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=node_hist.shape[1]
        )
        output = F.dropout(
            output,
            p=1.0 - hp["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]
        return output[torch.arange(output.shape[0]), node_hist_len - 1]

############################
#   Node Future Encoder   #
############################
class NodeFutureEncoder(nn.Module):
    """
    Node Future Encoder using Bi-directional LSTM
    """
    def __init__(self, state_length, pred_state_length, hidden_dim):
        super(NodeFutureEncoder, self).__init__()
        self.lstm = nn.LSTM(
            input_size=pred_state_length,
            hidden_size=hidden_dim,
            bidirectional=True,
            batch_first=True
        )
        self.initial_h_model = nn.Linear(state_length, hidden_dim)
        self.initial_c_model = nn.Linear(state_length, hidden_dim)

    def forward(self, hp, mode, node_present, node_future, future_lens) -> torch.Tensor:
        """
        Encodes the node future (during training) using a bi-directional LSTM

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param node_present: Current state of the node. [bs, state]
        :param node_future: Future states of the node. [bs, ph, state]
        :return: Encoded future.
        """
        # Here we're initializing the forward hidden states,
        # but zeroing the backward ones.
        initial_h = self.initial_h_model(node_present)
        initial_h = torch.stack( #TODO: verify node_present.device
            [initial_h, torch.zeros_like(initial_h, device=node_present.device)], dim=0
        )
        initial_c = self.initial_c_model(node_present)
        initial_c = torch.stack( #TODO: verify node_present.device
            [initial_c, torch.zeros_like(initial_c, device=node_present.device)], dim=0
        )

        initial_state = (initial_h, initial_c)

        node_future_packed = nn.utils.rnn.pack_padded_sequence(
            node_future, future_lens, batch_first=True, enforce_sorted=False
        )

        _, state = self.lstm(node_future_packed, initial_state)
        state = unpack_rnn_state(state)
        state = F.dropout(
            state,
            p=1.0 - hp["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )

        return state

####################
#   Edge Encoder   #
####################
#TODO: implement edge encoder modules
class EdgeEncoder(nn.Module):
    """
    Edge Encoder using LSTM
    """
    def __init__(self, state_length, neighbor_state_length, rnn_hidden_dim):
        super(EdgeEncoder, self).__init__()
        edge_encoder_input_size = state_length + neighbor_state_length
        self.lstm = nn.LSTM(
            edge_encoder_input_size,
            rnn_hidden_dim,
            batch_first=True
        )

    def forward(
        self,
        mode,
        hp,
        joint_history,
        joint_history_len,
    ) -> torch.Tensor:
        """Encode all agent-neighbor joint histories"""
        #TODO: input should be only for a given neigh. type
        packed_input = nn.utils.rnn.pack_padded_sequence(
            joint_history, joint_history_len, batch_first=True, enforce_sorted=False
        )
        packed_output, _ = self.lstm(packed_input)
        outputs, _ = nn.utils.rnn.pad_packed_sequence(
            packed_output, batch_first=True, total_length=joint_history.shape[1]
        )
        outputs = F.dropout(
            outputs,
            p=1.0 - hp["rnn_kwargs"]["dropout_keep_prob"],
            training=(mode == ModeKeys.TRAIN),
        )  # [bs, max_time, enc_rnn_dim]

        return outputs[torch.arange(outputs.shape[0]), joint_history_len - 1]
    
################################
#   Edge Interaction Encoder   #
################################
#TODO: implement edge encoder modules
class EdgeInteractionEncoder(nn.Module):
    """
    Edge Interaction Encoder using Transformer with Edge Encoding
    """
    def __init__(self, edge_types, hyperparams):
        super(EdgeInteractionEncoder, self).__init__()

    def forward(self, mode, node_hist, node_hist_len) -> torch.Tensor:
        return None

###################
#   Map Encoder   #
###################
class CNNMapEncoder(nn.Module):
    """
    Convolutional Neural Network (CNN) based Map Encoder
    """
    def __init__(
        self, map_channels, hidden_channels, output_size, masks, strides, patch_size
    ):
        super(CNNMapEncoder, self).__init__()
        self.convs = nn.ModuleList()
        input_size = (map_channels, patch_size, patch_size)
        x_dummy = torch.ones(input_size).unsqueeze(0) * torch.tensor(float("nan"))

        for i, _ in enumerate(hidden_channels):
            self.convs.append(
                nn.Conv2d(
                    map_channels if i == 0 else hidden_channels[i - 1],
                    hidden_channels[i],
                    masks[i],
                    stride=strides[i],
                )
            )
            x_dummy = self.convs[i](x_dummy)

        self.fc = nn.Linear(x_dummy.numel(), output_size)

    def forward(self, hp, nt, mode, x):
        """Forward pass through the CNN Map Encoder"""
        for conv in self.convs:
            x = F.leaky_relu(conv(x), 0.2)
        x = torch.flatten(x, start_dim=1)
        x = self.fc(x)

        x = F.dropout(
            x,
            p=hp["map_encoder"][nt]["dropout"],
            training=(mode == ModeKeys.TRAIN),
        )
        return x

###################
# p_z_x OR q_z_xy #
###################
class LatentDistEncoder(nn.Module):
    """
    Latent Distribution Encoder of the CVAE
    """
    def __init__(self, p_input_dim, p_hidden_dim, latent_dim):
        super(LatentDistEncoder, self).__init__()
        self.p_hidden_dim = p_hidden_dim
        if self.p_hidden_dim != 0:
            self.p_dist = nn.Linear(p_input_dim, p_hidden_dim)
            h_input_dim = self.p_hidden_dim
        else:
            h_input_dim = p_input_dim
        self.h_dist = nn.Linear(h_input_dim, latent_dim)

    def forward(self, hp, mode, enc):
        """Forward pass (encoding -> latent value)"""
        if self.p_hidden_dim != 0:
            h = F.dropout(
                F.relu(self.p_dist(enc)),
                p=1.0 - hp["MLP_dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )
        else:
            h = enc
        latent = self.h_dist(h)

        return latent

