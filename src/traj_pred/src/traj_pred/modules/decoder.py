"""Decoder modules for MG-CVAE"""
import torch
import torch.nn as nn
import torch.nn.functional as F

#######################
#   Decoder Pre-GRU   #
#######################
class DecoderPreGRU(nn.Module):
    """
    To calculate initial values for Gated Recurrent Unit Decoder
    """
    def __init__(
            self,
            state_length,
            pred_state_length,
            z_size,
            x_size,
            rnn_hidden_dim,
    ):
        super(DecoderPreGRU, self).__init__()
        self.state_action = nn.Linear(state_length, pred_state_length)
        self.initial_h = nn.Linear(z_size + x_size, rnn_hidden_dim)

    def forward(
            self,
            zx,
            n_s_t0
    ):
        """Forward pass to get initial state for GRU decoder unit"""
        initial_state = self.initial_h(zx)
        a_0 = self.state_action(n_s_t0)

        return initial_state, a_0

###################
#   Decoder GRU   #
###################
class DecoderGRU(nn.Module):
    """
    Gated Recurrent Unit Decoder
    """
    def __init__(
            self,
            rnn_hidden_dim,
            decoder_input_dim,
            decoder_final_dim
    ):
        super(DecoderGRU, self).__init__()
        self.rnn_cell = nn.GRUCell(decoder_input_dim, rnn_hidden_dim)
        self.post_rnn = nn.Linear(rnn_hidden_dim, decoder_final_dim)

    def forward(
            self,
            gru_input,
            state
    ):
        """Forward pass of the GRU Decoder"""
        h_state = self.rnn_cell(gru_input, state)
        output = F.relu(self.post_rnn(h_state))

        return h_state, output

###################
#   Decoder GMM   #
###################
class DecoderGMM(nn.Module):
    """
    Gausian Mixture Model Decoder
    """
    def __init__(
            self,
            pred_state_length,
            decoder_final_dim,
            gmm_components
    ):
        super(DecoderGMM, self).__init__()
        # gmm_mus_dim = pred_state_length
        # gmm_log_sigmas_dim = pred_state_length
        # gmm_corrs_dim = 1
        # gmm_dims = gmm_mus_dim + gmm_log_sigmas_dim + gmm_corrs_dim

        self.proj_to_log_pis = nn.Linear(decoder_final_dim, gmm_components)
        self.proj_to_mus = nn.Linear(decoder_final_dim, gmm_components*pred_state_length)
        self.proj_to_log_sigmas = nn.Linear(decoder_final_dim, gmm_components*pred_state_length)
        self.proj_to_corrs = nn.Linear(decoder_final_dim, gmm_components)

    def forward(self, gmm_input):
        """Forward pass of the GMM decoder module"""
        log_pis = self.proj_to_log_pis(gmm_input)
        mus = self.proj_to_mus(gmm_input)
        log_sigmas = self.proj_to_log_sigmas(gmm_input)
        corrs = torch.tanh(self.proj_to_corrs(gmm_input))
        return log_pis, mus, log_sigmas, corrs
