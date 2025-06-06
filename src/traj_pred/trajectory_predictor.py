"""Main trajectory predictor class"""
from itertools import product
from typing import Any, Dict, List, Union

import numpy as np
import torch
import torch.nn as nn
from trajdata import AgentBatch, SceneBatch, AgentType

from traj_pred.modules.mgcvae import MultimodalGenerativeCVAE
from traj_pred.modules.snce import SocialNCE
from traj_pred.utils.evaluation import compute_batch_statistics_pt
from traj_pred.utils.annealing import step_annealers, set_annealing_params

class TrajectoryPredictor(nn.Module):
    """
    Class for training a model that predicts human trajectories
    """
    def __init__(self, model_registrar, hyperparams, log_writer, device):
        super(TrajectoryPredictor, self).__init__()
        self.hyperparams = hyperparams
        self.log_writer = log_writer
        self.device = device
        self._curr_iter = 0

        self.model_registrar = model_registrar
        self.node_models_dict = nn.ModuleDict()
        self.nodes = set()

        self.state = self.hyperparams["state"]
        self.state_length = dict()
        for state_type in self.state.keys():
            self.state_length[state_type] = int(
                np.sum(
                    [
                        len(entity_dims)
                        for entity_dims in self.state[state_type].values()
                    ]
                )
            )
        self.pred_state = self.hyperparams["pred_state"]


    def set_environment(self):
        """Initialize the MG-CVAE and Social NCE Models"""
        self.node_models_dict.clear()
        edge_types = list(product(AgentType, repeat=2))

        for node_type in AgentType:
            # Only add MG-CVAE Models for NodeTypes we want to predict
            if node_type.name in self.pred_state.keys():
                self.node_models_dict[node_type.name] = MultimodalGenerativeCVAE(
                    node_type,
                    self.model_registrar,
                    self.hyperparams,
                    self.device,
                    edge_types,
                    log_writer=self.log_writer
                )

        if self.hyperparams["incl_robot_node"]:
            x_size = 2*self.hyperparams["enc_rnn_dim_history"] + 4*self.hyperparams["enc_rnn_dim_future"]
        else:
            x_size = 2*self.hyperparams["enc_rnn_dim_history"]
        # Add Social NCE Model
        snce = SocialNCE(
            feat_dim=x_size,
            proj_head_dim=self.hyperparams["proj_head_dim"],
            event_enc_dim=self.hyperparams["event_enc_dim"],
            snce_head_dim=self.hyperparams["snce_head_dim"],
            hyperparams=self.hyperparams,
            device=self.device
        )
        self.node_models_dict["snce"] = self.model_registrar.get_model("snce", snce)

    @property
    def curr_iter(self):
        """Returns the current iteration number"""
        return self._curr_iter

    @curr_iter.setter
    def curr_iter(self, value):
        self._curr_iter = value
        for name, model in self.node_models_dict.items():
            if name != "snce":
                model.curr_iter = value

    def set_all_annealing_params(self):
        """Set the annealing parameters for all models in the predictor"""
        for name, model in self.node_models_dict.items():
            set_annealing_params(name, model)

    def step_all_annealers(self):
        """Step the annealers for all models in the predictor"""
        for _, model in self.node_models_dict.items():
            step_annealers(model)

    def forward(self, batch):
        """Forward pass of the trajectory predictor"""
        return self.train_loss(batch)

    def train_loss(self, batch: AgentBatch):
        """Calculate loss of the MGCVAE model as well as the Social NCE Loss"""
        batch.to(self.device)

        # Run forward pass
        losses: List[torch.Tensor] = list()
        losses_mgcvae: List[torch.Tensor] = list()
        losses_nce: List[torch.Tensor] = list()

        # Loss of the MG-CVAE model
        node_type: AgentType
        for node_type in batch.agent_types():
            # MG-CVAE loss
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]
            agent_type_batch = batch.for_agent_type(node_type)
            enc, loss_mgcvae = model(agent_type_batch)
            losses_mgcvae.append(loss_mgcvae)
            # Social NCE loss
            snce_model: SocialNCE = self.node_models_dict["snce"]
            if self.hyperparams['contrastive_weight'] > 0:
                loss_nce = snce_model(enc, batch)
                losses.append(loss_mgcvae + loss_nce * self.hyperparams["contrastive_weight"])
            else:
                loss_nce = torch.tensor([0.0])
                losses.append(loss_mgcvae)
            losses_nce.append(loss_nce)

        return sum(losses), sum(losses_mgcvae), sum(losses_nce)

    def predict_and_evaluate_batch(
        self,
        batch: AgentBatch,
        output_for_pd: bool = False,
    ) -> Union[List[Dict[str, Any]], Dict[AgentType, Dict[str, torch.Tensor]]]:
        """Predicts from a batch and then evaluates the output, returning the batched errors."""
        batch.to(self.device)

        # Run forward pass
        if output_for_pd:
            results: List[Dict[str, Any]] = list()
        else:
            results: Dict[AgentType, Dict[str, torch.Tensor]] = dict()

        node_type: AgentType
        for node_type in batch.agent_types():
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)

            ph = agent_type_batch.agent_fut.shape[1]

            predictions = model.predict(
                agent_type_batch,
                prediction_horizon=ph,
                num_samples=1,
                z_mode=True,
                gmm_mode=True,
                full_dist=False,
                output_dists=False
            )

            # Run forward pass
            y_dists, _ = model.predict(
                agent_type_batch,
                prediction_horizon=ph,
                num_samples=1,
                z_mode=False,
                gmm_mode=False,
                full_dist=True,
                output_dists=True
            )

            batch_eval: Dict[
                str, torch.Tensor
            ] = compute_batch_statistics_pt(
                agent_type_batch.agent_fut[..., :2],
                prediction_output_dict=predictions,
                y_dists=y_dists,
            )

            if output_for_pd:
                batch_eval["data_idx"] = agent_type_batch.data_idx
                results.append(batch_eval)
            else:
                results[node_type] = batch_eval

        return results

    def predict(
        self,
        batch: AgentBatch,
        num_samples=1,
        prediction_horizon=None,
        z_mode=False,
        gmm_mode=False,
        full_dist=True,
        all_z_sep=False,
        output_dists=True,
    ):
        """Obtains model predictions for a batch of data.

        Args:
            batch (AgentBatch): _description_
            num_samples (int, optional): _description_. Defaults to 1.
            prediction_horizon (_type_, optional): _description_. Defaults to None.
            z_mode (bool, optional): _description_. Defaults to False.
            gmm_mode (bool, optional): _description_. Defaults to False.
            full_dist (bool, optional): _description_. Defaults to True.
            all_z_sep (bool, optional): _description_. Defaults to False.
            output_dists (bool, optional): _description_. Defaults to True.

        Returns:
            _type_: _description_
        """
        batch.to(self.device)

        predictions_dict = {}
        dists_dict = {}

        node_type: AgentType
        for node_type in batch.agent_types():
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]

            agent_type_batch = batch.for_agent_type(node_type)

            if prediction_horizon is None:
                ph = agent_type_batch.agent_fut.shape[1]
            else:
                ph = prediction_horizon

            # Run forward pass
            pred_object = model.predict(
                agent_type_batch,
                prediction_horizon=ph,
                num_samples=num_samples,
                z_mode=z_mode,
                gmm_mode=gmm_mode,
                full_dist=full_dist,
                all_z_sep=all_z_sep,
                output_dists=output_dists,
            )

            if output_dists:
                y_dists, predictions = pred_object
            else:
                predictions = pred_object

            predictions_np = predictions.cpu().detach().numpy()
            if output_dists:
                y_dists.set_device(torch.device("cpu"))

            # Assign predictions to node
            for i, agent_name in enumerate(agent_type_batch.agent_name):
                predictions_dict[f"{str(node_type)}/{agent_name}"] = predictions_np[
                    :, i
                ]
                if output_dists:
                    dists_dict[f"{str(node_type)}/{agent_name}"] = y_dists.get_for_node(
                        i
                    )

        if output_dists:
            return dists_dict, predictions_dict
        else:
            return predictions_dict
        
    def get_encoding(self, batch: SceneBatch):
        """
        Get encodings for all agents in a scene
        (works only with batch_size==1)

        Args:
            batch (SceneBatch): information of all agents in a scene
        """
        node_type: AgentType
        for node_type in batch.agent_types():
            model: MultimodalGenerativeCVAE = self.node_models_dict[node_type.name]
            enc, _, _, _, _ = model.obtain_encoded_tensors(batch)
        return enc

    def incremental_forward(
        self,
        batch: SceneBatch,
        prediction_horizon,
        num_samples,
        full_dist=False
    ):
        """Predict future of all agents in a scene"""
        #TODO
        return None
