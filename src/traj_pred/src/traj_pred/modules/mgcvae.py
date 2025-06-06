"""The MG-CVAE NN Module"""
from typing import Dict, List, Tuple

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from trajdata import AgentBatch, AgentType
import wandb

import traj_pred.modules as md
import traj_pred.dynamics as dynamic_module

from traj_pred.utils.model_utils import (
    ModeKeys,
    mutual_inf_mc,
    get_agent_neigh_joint_state
)

class MultimodalGenerativeCVAE(nn.Module):
    """MG-CVAE class definition"""
    def __init__(
        self,
        node_type_obj: AgentType,
        model_registrar,
        hyperparams: Dict,
        device,
        edge_types,
        log_writer=None
    ):
        """Initialization"""
        super(MultimodalGenerativeCVAE, self).__init__()

        self.hyperparams = hyperparams
        self.node_type_obj: AgentType = node_type_obj
        self.node_type: str = node_type_obj.name
        self.model_registrar = model_registrar
        self.device = device
        self.log_writer = log_writer

        self._curr_iter = 0
        self.node_modules = nn.ModuleDict()

        self.edge_types: List[Tuple[AgentType, AgentType]] = [
            edge_type for edge_type in edge_types if edge_type[1] is node_type_obj
        ]

        self.state = self.hyperparams["state"]
        self.pred_state = self.hyperparams["pred_state"][self.node_type]
        self.state_length = int(
            np.sum(
                [
                    len(entity_dims)
                    for entity_dims in self.state[self.node_type].values()
                ]
            )
        )
        if self.hyperparams["incl_robot_node"]:
            self.robot_state_length = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[AgentType.VEHICLE.name].values()
                    ]
                )
            )
        self.pred_state_length = int(
            np.sum([len(entity_dims) for entity_dims in self.pred_state.values()])
        )

        self.eie_output_dims = 0
        self.x_size = self.hyperparams["enc_rnn_dim_history"] # To keep track of the size of the encoding vector
        self.z_size = self.hyperparams["N"] * self.hyperparams["K"]
        self.latent = None
        # build mg-cvae model and update variables defining the model's internal sizes 
        self.build_model()

        dynamic_class = getattr(
            dynamic_module, hyperparams["dynamic"][self.node_type]["name"]
        )
        dyn_limits = hyperparams["dynamic"][self.node_type]["limits"]
        self.dynamic = dynamic_class(
            dyn_limits,
            device,
            self.model_registrar,
            self.x_size,
            self.node_type,
            self.hyperparams,
        )

    @property
    def curr_iter(self):
        """Returns the current iteration number"""
        return self._curr_iter

    @curr_iter.setter
    def curr_iter(self, value):
        self._curr_iter = value

    def add_submodule(self, name, model_if_absent):
        """Register modules of MGCVAE with the model registrar"""
        self.node_modules[name] = self.model_registrar.get_model(name, model_if_absent)

    def build_model(self):
        """Builds the MGCVAE model by creating all necessary components"""
        self.node_modules = nn.ModuleDict()
        hp = self.hyperparams
        nt = self.node_type

        # Node History Encoder
        self.add_submodule(
            nt + "/node_history_encoder",
            model_if_absent=md.NodeHistoryEncoder(self.state_length, hp["enc_rnn_dim_history"])
        )

        # Node Future Encoder
        self.add_submodule(
            nt + "/node_future_encoder",
            model_if_absent=md.NodeFutureEncoder(
                self.state_length,
                self.pred_state_length,
                hp["enc_rnn_dim_future"]
            )
        )

        # Robot Future Encoder
        if self.hyperparams["incl_robot_node"]:
            self.add_submodule(
                "robot_future_encoder",
                model_if_absent=md.NodeFutureEncoder(
                    self.robot_state_length,
                    self.robot_state_length,
                    hp["enc_rnn_dim_future"]
                )
            )
            self.x_size += 4 * self.hyperparams["enc_rnn_dim_future"]

        # Edge Interaction Encoders
        if self.hyperparams["edge_encoding"]:
            self.add_submodule(
                nt + "/edge_influence_encoder",
                model_if_absent=nn.MultiheadAttention( #TODO: update attention module
                    embed_dim=hp["enc_rnn_dim_history"],
                    num_heads=1,
                    kdim=hp["enc_rnn_dim_edge"],
                    vdim=hp["enc_rnn_dim_edge"],
                    batch_first=False,
                )
            )

            self.eie_output_dims = hp["enc_rnn_dim_history"]
            self.x_size += self.eie_output_dims

            # Edge Encoders
            for edge_type in self.edge_types:
                neighbor_state_length = int(
                    np.sum(
                        [
                            len(entity_dims)
                            for entity_dims in self.state[edge_type[0].name].values()
                        ]
                    )
                )

                #TODO: implement edge encoder modules
                self.add_submodule(
                    "->".join([edge_type[0].name, edge_type[1].name]) + "/edge_encoder",
                    model_if_absent=md.EdgeEncoder(
                        self.state_length,
                        neighbor_state_length,
                        hp["enc_rnn_dim_edge"]
                    )
                )

        # Map Encoder
        if hp["map_encoding"]:
            if nt in hp["map_encoder"]:
                me_params = hp["map_encoder"][nt]
                self.add_submodule(
                    nt + "/map_encoder",
                    model_if_absent=md.CNNMapEncoder(
                        me_params["map_channels"],
                        me_params["hidden_channels"],
                        me_params["output_size"],
                        me_params["masks"],
                        me_params["strides"],
                        me_params["patch_size"]
                    )
                )
                self.x_size += hp["map_encoder"][nt]["output_size"]

        # Discrete Latent Variable
        self.latent = md.DiscreteLatent(hp, self.device)

        # p_z_x
        self.add_submodule(
            nt + "/p_z_x",
            model_if_absent=md.LatentDistEncoder(
                self.x_size,
                hp["p_z_x_MLP_dims"],
                self.latent.z_dim)
        )

        # q_z_xy
        self.add_submodule(
            nt + "/q_z_xy",
            model_if_absent=md.LatentDistEncoder(
                self.x_size + 4 * hp["enc_rnn_dim_future"],
                hp["q_z_xy_MLP_dims"],
                self.latent.z_dim)
        )

        # Decoder
        if self.hyperparams["incl_robot_node"]:
            decoder_input_dims = (
                self.pred_state_length + self.robot_state_length + self.z_size + self.x_size
            )
        else:
            decoder_input_dims = self.pred_state_length + self.z_size + self.x_size
        # Decoder Pre-GRU
        self.add_submodule(
            nt + "/decoder/PreGRU",
            model_if_absent=md.DecoderPreGRU(
                self.state_length,
                self.pred_state_length,
                self.z_size,
                self.x_size,
                hp["dec_rnn_dim"],
            )
        )
        # Decoder GRU
        self.add_submodule(
            nt + "/decoder/GRU",
            model_if_absent=md.DecoderGRU(
                hp["dec_rnn_dim"],
                decoder_input_dims,
                hp["dec_final_dim"]
            )
        )
        # Decoder GMM
        self.add_submodule(
            nt + "/decoder/GMM",
            model_if_absent=md.DecoderGMM(
                self.pred_state_length,
                self.hyperparams["dec_final_dim"],
                hp["GMM_components"]
            )
        )

        # transfer modules to device
        for _, module in self.node_modules.items():
            module.to(self.device)

    def encode_total_edge_influence(
        self,
        mode: ModeKeys,
        encoded_edges: torch.Tensor,
        num_neighbors: torch.Tensor,
        node_history_encoded: torch.Tensor,
        batch_size: int,
    ) -> Tuple[torch.Tensor, torch.Tensor]:
        """Encode edge interactions using attention"""
        max_neighbors = encoded_edges.shape[0]

        if len(encoded_edges) == 0:
            combined_edges = torch.zeros(
                (batch_size, self.eie_output_dims), device=self.device
            )
        else:
            with_neighbors = num_neighbors > 0
            combined_edges = torch.zeros_like(node_history_encoded).unsqueeze(0)

            key_padding_mask = torch.triu(
                torch.ones(
                    (max_neighbors + 1, max_neighbors),
                    dtype=torch.bool,
                    device=self.device,
                ),
                diagonal=0,
            )[num_neighbors]
            combined_edges[:, with_neighbors], _ = self.node_modules[
                self.node_type + "/edge_influence_encoder"
            ](
                query=node_history_encoded[with_neighbors].unsqueeze(0),
                key=encoded_edges[:, with_neighbors],
                value=encoded_edges[:, with_neighbors],
                key_padding_mask=key_padding_mask[with_neighbors],
                attn_mask=None,
            )
            combined_edges = F.dropout(
                combined_edges.squeeze(0),
                p=1.0 - self.hyperparams["rnn_kwargs"]["dropout_keep_prob"],
                training=(mode == ModeKeys.TRAIN),
            )
        
        return combined_edges
    
    def obtain_encoded_tensors(
        self, mode: ModeKeys, batch: AgentBatch
    ) -> Tuple[
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
        torch.Tensor,
    ]:
        """
        Encodes input and output tensors for node and robot.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param batch: AgentBatch from trajdata UnifiedDataset.
        :return: tuple(x, x_nr_t, y_e, y_r, y, n_s_t0)
            WHERE
            - x: Encoded input / condition tensor to the CVAE x_e.
            - x_r_t: Robot state (if robot is in scene).
            - y_e: Encoded label / future of the node.
            - y_r: Encoded future of the robot.
            - y: Label / future of the node.
            - n_s_t0: Standardized current state of the node.
        """

        hp = self.hyperparams
        nt = self.node_type
        nm = self.node_modules

        enc, x_r_t, y_e, y_r, y = None, None, None, None, None

        batch_size = batch.agent_hist.shape[0]

        #########################################
        # Provide basic information to encoders #
        #########################################
        node_history_st = batch.agent_hist
        node_history_len = batch.agent_hist_len
        node_present_state_st = node_history_st[
            torch.arange(node_history_st.shape[0]), node_history_len - 1
        ]

        # to inititate dynamic model of node
        initial_dynamics = dict()
        initial_dynamics["pos"] = node_present_state_st[:, 0:2]
        initial_dynamics["vel"] = node_present_state_st[:, 2:4]
        self.dynamic.set_initial_condition(initial_dynamics)

        enc_concat_list = list()    # To concat encodings
        ##################
        # Encode History #
        ##################
        node_history_encoded = nm[nt + "/node_history_encoder"](
            hp, mode, node_history_st, node_history_len
        )
        enc_concat_list.append(node_history_encoded)  # [bs/nbs, enc_rnn_dim_history]

        ###############################################
        # Encode Edges and Node-Neighbor Interactions #
        ###############################################
        if self.hyperparams["edge_encoding"]:
            if batch.num_neigh.max() == 0:
                total_edge_influence = torch.zeros_like(node_history_encoded)
            else:
                ################
                # Encode Edges #
                ################
                # get concatenated node-neighbor history for valid timestpes
                joint_history, joint_history_len, joint_neigh_types = get_agent_neigh_joint_state(
                    node_history_st,
                    node_history_len,
                    batch.neigh_hist,
                    batch.neigh_hist_len,
                    batch.neigh_types,
                )
                total_neighbors: int = batch.num_neigh.sum().item()
                # to keep track of initial order as neighbors will be processed per type
                sorting_indices = torch.empty(
                    total_neighbors, dtype=torch.long, device=self.device
                )
                returns: List[torch.Tensor] = list()
                num_already_done = 0
                for neigh_type in joint_neigh_types.unique(sorted=False):
                    if neigh_type < 0:
                        # AgentType is non-negative (by design, negative values are padding values).
                        continue

                    matches_type = joint_neigh_types == neigh_type

                    num_matching_type = matches_type.sum()
                    sorting_indices[matches_type] = torch.arange(
                        start=num_already_done,
                        end=num_already_done + num_matching_type,
                        dtype=sorting_indices.dtype,
                        device=self.device,
                    )
                    joint_history_type = joint_history[matches_type]
                    joint_history_type_len = joint_history_len[matches_type]
                    # forward pass through edge encoder for node-neigh pair
                    edge_str = "->".join([AgentType(neigh_type.item()).name, self.node_type])
                    ret = nm[edge_str + "/edge_encoder"](
                        mode, hp, joint_history_type, joint_history_type_len
                    )
                    returns.append(ret)
                    num_already_done += num_matching_type

                batch_indexed_outputs = torch.concat(returns, dim=0)[sorting_indices]

                #TODO: are dynamic edges required?

                encoded_edges: Tuple[torch.Tensor, ...] = torch.split(
                    batch_indexed_outputs, batch.num_neigh.tolist()
                )
                encoded_edges = nn.utils.rnn.pad_sequence(encoded_edges)

                # Encode Interactions per type
                #TODO
                total_edge_influence = self.encode_total_edge_influence(
                    mode,
                    encoded_edges,
                    batch.num_neigh,
                    node_history_encoded,
                    batch_size,
                )

            enc_concat_list.append(total_edge_influence) # [bs/nbs, enc_rnn_dim]

        ################
        # Map Encoding #
        ################
        if (hp["map_encoding"] and nt in hp["map_encoder"]):
            if (
                hp["log_maps"]
                and self.log_writer
                and (self.curr_iter + 1) % 500 == 0
            ):
                image = wandb.Image(batch.maps[0], caption=f"Batch Map {self.curr_iter}")
                self.log_writer.log(
                    {f"{self.node_type}/maps": image}, step=self.curr_iter, commit=False
                )

            encoded_map = nm[nt + "/map_encoder"](hp, nt, mode, batch.maps * 2.0 - 1.0)

            if self.log_writer:
                self.log_writer.log(
                    {
                        f"{nt}/encoded_map_max": torch.max(
                            torch.abs(encoded_map)
                        ).item()
                    },
                    step=self.curr_iter,
                    commit=False,
                )
            enc_concat_list.append(encoded_map)

        #########################
        # Robot Future Encoding #
        #########################
        if hp["incl_robot_node"]:
            robot = batch.robot_fut
            robot_lens = batch.robot_fut_len
            x_r_t, y_r = robot[:, 0], robot[:, 1:]
            robot_future_encoder = nm["robot_future_encoder"](
                hp, mode, x_r_t, y_r, robot_lens
            )
            enc_concat_list.append(robot_future_encoder)

        # Final concatenated encoding
        enc = torch.cat(enc_concat_list, dim=-1)

        ################################
        # Encode Node Present & Future #
        ################################
        node_present = node_present_state_st  # [bs, state_dim]
        if mode == ModeKeys.TRAIN or mode == ModeKeys.EVAL:
            y = batch.agent_fut[..., :2]
            y_lens = batch.agent_fut_len
            y_e = nm[nt + "/node_future_encoder"](
                hp, mode, node_present, y, y_lens
            )

        return enc, x_r_t, y_e, y_r, y

    def p_y_xz(
        self,
        mode,
        x,
        x_nr_t,
        y_r,
        n_s_t0,
        z_stacked,
        dt,
        prediction_horizon,
        num_samples,
        num_components=1,
        gmm_mode=False,
    ):
        r"""
        .. math:: p_\psi(\mathbf{y}_i \mid \mathbf{x}_i, z)

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y_r: Encoded future robot tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z_stacked: Stacked latent state. [num_samples_z * num_samples_gmm, bs, latent_state]
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param num_components: Number of GMM components.
        :param z_mode: If True: The most likely z latent value is being used.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :return: GMM2D. If mode is Predict, also samples from the GMM.
        """
        ph = prediction_horizon
        pred_dim = self.pred_state_length

        z = torch.reshape(z_stacked, (-1, self.latent.z_dim))
        zx = torch.cat([z, x.repeat(num_samples * num_components, 1)], dim=1)

        # Infer initial action state for node from current state
        initial_state, a_0 = self.node_modules[self.node_type + "/decoder/PreGRU"](
            zx,
            n_s_t0,
        )

        log_pis, mus, log_sigmas, corrs, a_sample = [], [], [], [], []

        if self.hyperparams["incl_robot_node"]:
            input_ = torch.cat(
                [
                    zx,
                    a_0.repeat(num_samples * num_components, 1),
                    x_nr_t.repeat(num_samples * num_components, 1),
                ],
                dim=1,
            )
        else:
            input_ = torch.cat([zx, a_0.repeat(num_samples * num_components, 1)], dim=1)

        state = initial_state
        for j in range(ph):
            h_state, decoder_out = self.node_modules[self.node_type + "/decoder/GRU"](
                input_,
                state
            )
            log_pi_t, mu_t, log_sigma_t, corr_t = self.node_modules[
                self.node_type  + "/decoder/GMM"
            ](decoder_out)

            gmm = md.GMM2D(log_pi_t, mu_t, log_sigma_t, corr_t)  # [k;bs, pred_dim]

            if mode == ModeKeys.PREDICT and gmm_mode:
                a_t = gmm.mode()
            else:
                a_t = gmm.rsample()

            if num_components > 1:
                if mode == ModeKeys.PREDICT:
                    log_pis.append(self.latent.p_dist.logits.repeat(num_samples, 1, 1))
                else:
                    log_pis.append(self.latent.q_dist.logits.repeat(num_samples, 1, 1))
            else:
                log_pis.append(
                    torch.ones_like(
                        corr_t.reshape(num_samples, num_components, -1)
                        .permute(0, 2, 1)
                        .reshape(-1, 1)
                    )
                )

            mus.append(
                mu_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            log_sigmas.append(
                log_sigma_t.reshape(num_samples, num_components, -1, 2)
                .permute(0, 2, 1, 3)
                .reshape(-1, 2 * num_components)
            )
            corrs.append(
                corr_t.reshape(num_samples, num_components, -1)
                .permute(0, 2, 1)
                .reshape(-1, num_components)
            )

            if self.hyperparams["incl_robot_node"]:
                dec_inputs = [
                    zx,
                    a_t,
                    y_r[:, j].repeat(num_samples * num_components, 1),
                ]
            else:
                dec_inputs = [zx, a_t]
            input_ = torch.cat(dec_inputs, dim=1)
            state = h_state

        log_pis = torch.stack(log_pis, dim=1)
        mus = torch.stack(mus, dim=1)
        log_sigmas = torch.stack(log_sigmas, dim=1)
        corrs = torch.stack(corrs, dim=1)

        a_dist = md.GMM2D(
            torch.reshape(log_pis, [num_samples, -1, ph, num_components]),
            torch.reshape(mus, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(log_sigmas, [num_samples, -1, ph, num_components * pred_dim]),
            torch.reshape(corrs, [num_samples, -1, ph, num_components]),
        )

        if self.hyperparams["dynamic"][self.node_type]["distribution"]:
            y_dist = self.dynamic.integrate_distribution(a_dist, x, dt)
        else:
            y_dist = a_dist

        if mode == ModeKeys.PREDICT:
            if gmm_mode:
                a_sample = a_dist.mode()
            else:
                a_sample = a_dist.rsample()
            sampled_future = self.dynamic.integrate_samples(a_sample, x, dt)
            return y_dist, sampled_future
        else:
            return y_dist

    def encoder(self, mode, enc, y_e, num_samples=None):
        """
        Encoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param y_e: Encoded future tensor.
        :param num_samples: Number of samples from the latent space during Prediction.
        :return: tuple(z, kl_obj)
            WHERE
            - z: Samples from the latent space.
            - kl_obj: KL Divergenze between q and p
        """
        if mode == ModeKeys.TRAIN:
            sample_ct = self.hyperparams["k"]
        elif mode == ModeKeys.EVAL:
            sample_ct = self.hyperparams["k_eval"]
        elif mode == ModeKeys.PREDICT:
            sample_ct = num_samples
            if num_samples is None:
                raise ValueError("num_samples cannot be None with mode == PREDICT.")

        self.latent.q_dist = self.latent.dist_from_h(
            self.node_modules[self.node_type + "/q_z_xy"](
                self.hyperparams, mode, torch.cat([enc, y_e], dim=-1)
            ),
            mode
        )
        self.latent.p_dist = self.latent.dist_from_h(
            self.node_modules[self.node_type + "/p_z_x"](
                self.hyperparams, mode, enc
            ),
            mode
        )

        z = self.latent.sample_q(sample_ct, mode)

        if mode == ModeKeys.TRAIN:
            kl_obj = self.latent.kl_q_p(
                self.log_writer, self.node_type, self.curr_iter
            )
            if self.log_writer is not None:
                self.log_writer.log(
                    {f"{self.node_type}/kl": kl_obj.item()},
                    step=self.curr_iter,
                    commit=False,
                )
        else:
            kl_obj = None

        return z, kl_obj

    def decoder(
        self,
        mode: ModeKeys,
        enc: torch.Tensor,
        x_nr_t: torch.Tensor,
        y: torch.Tensor,
        y_r: torch.Tensor,
        pos_hist: torch.Tensor,
        z: torch.Tensor,
        dt: torch.Tensor,
        num_samples: int,
    ):
        """
        Decoder of the CVAE.

        :param mode: Mode in which the model is operated. E.g. Train, Eval, Predict.
        :param x: Input / Condition tensor.
        :param x: Input / Condition tensor.
        :param x_nr_t: Joint state of node and robot (if robot is in scene).
        :param y: Future tensor.
        :param y_r: Encoded future tensor.
        :param n_s_t0: Standardized current state of the node.
        :param z: Stacked latent state.
        :param num_samples: Number of samples from the latent space.
        :return: Log probability of y over p.
        """

        num_components = self.hyperparams["N"] * self.hyperparams["K"]
        y_dist = self.p_y_xz(
            mode,
            enc,
            x_nr_t,
            y_r,
            pos_hist,
            z,
            dt,
            y.shape[1],
            num_samples,
            num_components=num_components,
        )

        if self.hyperparams["single_mode_multi_sample"]:
            log_p_ynt_xz = y_dist.log_prob(torch.nan_to_num(y))
            log_p_yt_xz = torch.logsumexp(log_p_ynt_xz, dim=0, keepdim=True) - np.log(
                log_p_ynt_xz.shape[0]
            )
        else:
            log_p_yt_xz = torch.clamp(
                y_dist.log_prob(torch.nan_to_num(y)),
                max=self.hyperparams["log_p_yt_xz_max"],
            )

        if (
            self.hyperparams["log_histograms"]
            and self.log_writer
            and (self.curr_iter + 1) % 500 == 0
        ):
            self.log_writer.log(
                {
                    f"{str(self.node_type)}/log_p_yt_xz": wandb.Histogram(
                        log_p_yt_xz.detach().cpu().numpy()
                    )
                },
                step=self.curr_iter,
                commit=False,
            )

        nan_mask = (~y.isfinite()).any(dim=-1)
        log_p_y_xz = torch.sum(log_p_yt_xz.masked_fill_(nan_mask, 0.0), dim=-1)

        # if self.hyperparams["single_mode_multi_sample"]:
        #     log_p_y_xz /= log_p_yt_xz.shape[-1]

        return log_p_y_xz

    def forward(self, batch: AgentBatch) -> torch.Tensor:
        """Forward pass of MG-CVAE"""
        return self.train_loss(batch)

    def train_loss(self, batch: AgentBatch) -> torch.Tensor:
        """
        Calculates the training loss for a batch.

        :param inputs: Input tensor including the state for each agent over time [bs, t, state].
        :param inputs_st: Standardized input tensor.
        :param first_history_indices: First timestep (index) in scene for which data is available for a node [bs]
        :param labels: Label tensor including the label output for each agent over time [bs, t, pred_state].
        :param labels_st: Standardized label tensor.
        :param neighbors: Preprocessed dict (indexed by edge type) of list of neighbor states over time.
                            [[bs, t, neighbor state]]
        :param neighbors_edge_value: Preprocessed edge values for all neighbor nodes [[N]]
        :param robot: Standardized robot state over time. [bs, t, robot_state]
        :param map: Tensor of Map information. [bs, channels, x, y]
        :param prediction_horizon: Number of prediction timesteps.
        :return: tensor -> encoding, Scalar tensor -> nll loss
        """
        mode = ModeKeys.TRAIN

        enc, x_nr_t, y_e, y_r, y = self.obtain_encoded_tensors(mode, batch)

        z, kl = self.encoder(mode, enc, y_e)


        pos_hist: torch.Tensor = batch.agent_hist[
            torch.arange(batch.agent_hist.shape[0]), batch.agent_hist_len - 1
        ]

        log_p_y_xz = self.decoder(
            mode,
            enc,
            x_nr_t,
            y,
            y_r,
            pos_hist,
            z,
            batch.dt,
            self.hyperparams["k"],
        )

        log_p_y_xz_mean = torch.mean(log_p_y_xz, dim=0)  # [nbs]
        log_likelihood = torch.mean(log_p_y_xz_mean)

        mutual_inf_q = mutual_inf_mc(self.latent.q_dist)
        mutual_inf_p = mutual_inf_mc(self.latent.p_dist)

        ELBO = log_likelihood - self.kl_weight * kl + 1.0 * mutual_inf_p
        loss = -ELBO

        if (
            self.hyperparams["log_histograms"]
            and self.log_writer
            and (self.curr_iter + 1) % 500 == 0
        ):
            self.log_writer.log(
                {
                    f"{str(self.node_type)}/log_p_y_xz": wandb.Histogram(
                        log_p_y_xz_mean.detach().cpu().numpy()
                    )
                },
                step=self.curr_iter,
                commit=False,
            )

        if self.log_writer:
            self.log_writer.log(
                {
                    f"{str(self.node_type)}/mutual_information_q": mutual_inf_q.item(),
                    f"{str(self.node_type)}/mutual_information_p": mutual_inf_p.item(),
                    f"{str(self.node_type)}/log_likelihood": log_likelihood.item(),
                    f"{str(self.node_type)}/mgcvae loss": loss.item(),
                },
                step=self.curr_iter,
                commit=False,
            )
            if self.hyperparams["log_histograms"] and (self.curr_iter + 1) % 500 == 0:
                self.latent.summarize_for_tensorboard(
                    self.log_writer, str(self.node_type), self.curr_iter
                )
        return enc, loss
        # return loss_task

    def predict(
        self,
        batch: AgentBatch,
        prediction_horizon,
        num_samples,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
        output_dists=False,
    ):
        """
        Predicts the future of a batch of nodes.

        :param batch: Input batch of data.
        :param prediction_horizon: Number of prediction timesteps.
        :param num_samples: Number of samples from the latent space.
        :param z_mode: If True: Select the most likely latent state.
        :param gmm_mode: If True: The mode of the GMM is sampled.
        :param all_z_sep: Samples each latent mode individually without merging them into a GMM.
        :param full_dist: Samples all latent states and merges them into a GMM as output.
        :return:
        """
        mode = ModeKeys.PREDICT

        enc, x_nr_t, _, y_r, _ = self.obtain_encoded_tensors(mode, batch)

        self.latent.p_dist = self.latent.dist_from_h(
            self.node_modules[self.node_type + "/p_z_x"](
                self.hyperparams, mode, enc
            ),
            mode
        )

        z, num_samples, num_components = self.latent.sample_p(
            num_samples,
            mode,
            most_likely_z=z_mode,
            full_dist=full_dist,
            all_z_sep=all_z_sep,
        )


        # This is the old n_s_t0 (just the state at the current timestep, t=0).
        pos_hist: torch.Tensor = batch.agent_hist[
            torch.arange(batch.agent_hist.shape[0]), batch.agent_hist_len - 1
        ]

        y_dist, our_sampled_future = self.p_y_xz(
            mode,
            enc,
            x_nr_t,
            y_r,
            pos_hist,
            z,
            batch.dt,
            prediction_horizon,
            num_samples,
            num_components,
            gmm_mode,
        )

        if output_dists:
            return y_dist, our_sampled_future
        else:
            return our_sampled_future

