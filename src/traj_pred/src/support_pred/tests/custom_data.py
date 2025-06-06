import os
import json
from functools import partial
from collections import defaultdict
from typing import Tuple, List, Callable, Optional

import numpy as np
# import torch

from torch.utils import data
from trajdata import UnifiedDataset, SceneBatch
from trajdata.data_structures.batch_element import SceneBatchElement
from trajdata.data_structures.agent import AgentMetadata, AgentType
from trajdata.data_structures.state import StateArray
from trajdata.augmentation import NoiseHistories
from trajdata.utils.state_utils import transform_state_np_2d

def all_current_states(
    batch_elem: SceneBatchElement,
) -> np.ndarray:
    agents: List[AgentMetadata] = batch_elem.agents
    curr_pos = []
    for agent in agents:
        raw_state: StateArray = batch_elem.cache.get_raw_state(
            agent.name, batch_elem.scene_ts
        )
        state = np.asarray(raw_state)
        curr_pos.append(state)
    return np.stack(curr_pos, axis=0)

def custom_agent_hist(
    batch_elem: SceneBatchElement,
    history_sec: Tuple[Optional[float], Optional[float]],
) -> np.ndarray:
    """agent’s history transformed into agent-centric coordinates"""
    dt = batch_elem.dt
    max_hist_len: int = round((history_sec[1]/dt)) + 1
    world_agent_hist: List[np.ndarray] = batch_elem.agent_histories
    state_dim = world_agent_hist[0].shape[-1]

    agent_pos_list = [hist[-1,:2] for hist in world_agent_hist]
    agent_sc_list = [hist[-1,-2:] for hist in world_agent_hist]

    agents_hist_st: List[np.ndarray] = []
    for idx, (pos, sc) in enumerate(zip(agent_pos_list, agent_sc_list)):
        cos_agent = sc[-1]
        sin_agent = sc[-2]
        centered_world_from_agent_tf: np.ndarray = np.array(
            [
                [cos_agent, -sin_agent, pos[0]],
                [sin_agent, cos_agent, pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        centered_agent_from_world_tf: np.ndarray = np.linalg.inv(
            centered_world_from_agent_tf
        )
        hist = world_agent_hist[idx]
        hist_st = transform_state_np_2d(hist, centered_agent_from_world_tf)

        t_i = hist_st.shape[0]
        if t_i<max_hist_len:
            padding = np.full((max_hist_len-t_i, state_dim), np.nan, dtype=hist_st.dtype)
            hist_st_padded = np.concatenate([padding, hist_st], axis=0)
        else:
            hist_st_padded = hist_st[:max_hist_len]
        agents_hist_st.append(hist_st_padded)

    return np.stack(agents_hist_st, axis=0)

def get_neighs(
    batch_elem: SceneBatchElement,
    interaction_radius: float,
) -> np.ndarray:
    """Provides adjacency matrix"""
    agents: List[AgentMetadata] = batch_elem.agents
    curr_states = []
    for agent in agents:
        raw_state: StateArray = batch_elem.cache.get_raw_state(
            agent.name, batch_elem.scene_ts
        )
        state = np.asarray(raw_state)
        curr_states.append(state)

    is_neigh = []
    for state in curr_states:
        distances = [
            np.linalg.norm(state[:2] - agent_st[:2])
            for agent_st in curr_states
        ]
        is_neigh.append([dist <= interaction_radius for dist in distances])
    is_neigh_mat = np.stack(is_neigh, axis=0)
    np.fill_diagonal(is_neigh_mat, False)

    return is_neigh_mat

def per_agent_neigh_hist(
    batch_elem: SceneBatchElement,
    history_sec: Tuple[Optional[float], Optional[float]],
) -> np.ndarray:
    """
    Provide neighbor history of each agent in scene
    in respective agent-centric frames
    """
    assert batch_elem.standardize_data is False, \
        "Per-agent history requires a non-standarized dataset (set standardize_data=False)"

    dt = batch_elem.dt
    max_hist_len: int = round((history_sec[1]/dt)) + 1
    world_agent_hist: List[np.ndarray] = batch_elem.agent_histories
    state_dim = world_agent_hist[0].shape[-1]
    num_agents = batch_elem.num_agents

    agent_pos_list = [hist[-1,:2] for hist in world_agent_hist]
    agent_sc_list = [hist[-1,-2:] for hist in world_agent_hist]

    neigh_hists: List[List[np.ndarray]] = []
    for idx, (pos, sc) in enumerate(zip(agent_pos_list, agent_sc_list)):
        # calculate transformation matrix to go from world frame to agent frame
        cos_agent = sc[-1]
        sin_agent = sc[-2]
        centered_world_from_agent_tf: np.ndarray = np.array(
            [
                [cos_agent, -sin_agent, pos[0]],
                [sin_agent, cos_agent, pos[1]],
                [0.0, 0.0, 1.0],
            ]
        )
        centered_agent_from_world_tf: np.ndarray = np.linalg.inv(
            centered_world_from_agent_tf
        )

        row_hists: List[Optional[np.ndarray]] = []
        for jdx, agent_hist in enumerate(world_agent_hist):
            # skip self
            if jdx == idx:
                continue
            # append if neighbor
            if batch_elem.extras["is_neigh"][idx, jdx]:
                hist_st = transform_state_np_2d(agent_hist, centered_agent_from_world_tf)
                row_hists.append(hist_st)
            # else append None
            else:
                row_hists.append(None)
        neigh_hists.append(row_hists)

    output = np.full((num_agents, num_agents-1, max_hist_len, state_dim), np.nan, dtype=float)
    # pad arrays to match maximum history length (8 steps),
    # and max possible neighbors (num of agents - 1)
    for i in range(num_agents):
        for k, hist_ij in enumerate(neigh_hists[i]):
            if hist_ij is None:
                continue
            len_j = hist_ij.shape[0]
            output[i, k, -len_j:, :] = hist_ij

    return output

def extras_collate_fn(
    batch_elems: List[SceneBatchElement],
    history_sec: Tuple[Optional[float], Optional[float]],
    base_collate: Callable[[List[SceneBatchElement]], SceneBatch],
) -> SceneBatch:
    """
    1) Pads each extra (“agent_hist_st”, “is_neigh”, “neigh_hist”) 
    2) Calls base_collate(...) (i.e. scene_collate_fn) on the padded batch.
    """
    max_agent_num: int = max(elem.num_agents for elem in batch_elems) # M
    dt = batch_elems[0].dt
    max_hist_len: int = round((history_sec[1]/dt)) + 1
    state_dim = batch_elems[0].agent_histories[0].shape[-1]

    for elem in batch_elems:
        n_i = elem.num_agents
        agent_hist = elem.extras.get("agent_hist_st")
        mat = elem.extras["is_neigh"]
        neigh_hist = elem.extras["neigh_hist"]

        if n_i < max_agent_num:
            # Pad "agent_hist_st": shape (n_i, max_hist_len, 8) -> (M, max_hist_len, 8)
            pad_agent_hist = np.full(
                (max_agent_num-n_i, max_hist_len, state_dim),
                np.nan,
                dtype=agent_hist.dtype
            )
            padded_agent_hist = np.concatenate([agent_hist, pad_agent_hist], axis=0)
            # Pad "is_neigh": shape (n_i, n_i) -> (max_agents_in_batch, max_agents_in_batch)
            pad_mat = np.zeros((max_agent_num, max_agent_num), dtype=mat.dtype)
            pad_mat[:n_i, :n_i] = mat
            elem.extras["is_neigh"] = pad_mat
            # Pad "neigh_hist": shape (n_i, n_i-1, max_hist_len, 8) -> (M, M-1, max_hist_len, 8)
            padded_neigh_hist = np.full(
                (max_agent_num, max_agent_num-1, max_hist_len, state_dim),
                np.nan,
                dtype=neigh_hist.dtype
            )
            padded_neigh_hist[:n_i, :(n_i - 1), :, :] = neigh_hist
        else:
            padded_agent_hist = agent_hist[:max_agent_num]
            elem.extras["is_neigh"] = mat[:max_agent_num, :max_agent_num]
            padded_neigh_hist = neigh_hist[:max_agent_num]

        elem.extras["agent_hist_st"] = padded_agent_hist
        elem.extras["neigh_hist"] = padded_neigh_hist

    return base_collate(batch_elems)

def main():
    log_dir = 'data/trained_models/trajectory_prediction'
    model_dir = os.path.join(log_dir, "eth-28_May_2025_10_28_45")

    with open(os.path.join(model_dir, 'config.json'), 'r', encoding="utf-8") as config_json:
        hyperparams = json.load(config_json)
    # device
    hyperparams["device"] = "cpu"
    hyperparams["trajdata_cache_dir"] = "data/pedestrian_datasets/.unified_data_cache"

    desired_data=[
        "eupeds_eth-train",
    ]
    max_agent_num = 20
    data_dirs = {
        "eupeds_eth": "data/pedestrian_datasets/eth_ucy_peds",
    }

    attention_radius = defaultdict(
        lambda: 20.0
    )  # Default range is 20m unless otherwise specified.
    # attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 5.0
    interaction_radius = 5.0

    history_sec = (0.1, hyperparams["history_sec"])
    future_sec = (0.1, hyperparams["prediction_sec"])

    input_noise = 0.0
    augmentations = list()
    if input_noise > 0.0:
        augmentations.append(NoiseHistories(stddev=input_noise))

    batch_size = 4

    dataset = UnifiedDataset(
        desired_data=desired_data,
        centric="scene",
        history_sec=history_sec,
        future_sec=future_sec,
        agent_interaction_distances=attention_radius,
        max_agent_num=max_agent_num,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        augmentations=augmentations if len(augmentations) > 0 else None,
        standardize_data=False,
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
        extras={
            "agent_hist_st": partial(custom_agent_hist, history_sec=history_sec),
            "is_neigh": partial(get_neighs, interaction_radius=interaction_radius),
            "neigh_hist": partial(per_agent_neigh_hist, history_sec=history_sec),
        }
    )

    print(f"# Data Samples: {len(dataset)}")

    base_collate = dataset.get_collate_fn(pad_format="right")

    dataloader = data.DataLoader(
        dataset,
        # collate_fn=dataset.get_collate_fn(pad_format="right"),
        collate_fn=partial(
            extras_collate_fn,
            history_sec=(0.1, hyperparams["history_sec"]),
            base_collate=base_collate),
        pin_memory=False if hyperparams["device"] == "cpu" else True,
        batch_size=batch_size,
        shuffle=True,
        num_workers=hyperparams["preprocess_workers"],
        sampler=None,
    )

    batch: SceneBatch = next(iter(dataloader))

    return batch

if __name__ == '__main__':
    batch = main()
    print(f"Num of agents in scenes: {batch.num_agents}")
    print(f"Shape of agent_hist_st array: {batch.extras['agent_hist_st'].shape}")
    print(f"Shape of is_neigh array: {batch.extras['is_neigh'].shape}")
    print(f"Shape of neigh hist array: {batch.extras['neigh_hist'].shape}")
