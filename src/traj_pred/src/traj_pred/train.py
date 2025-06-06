"""
Train trajectpory predictor
"""

import os
import json
import random
import time
import pathlib
# import pickle
from collections import defaultdict
from typing import Dict, Optional

import numpy as np
import torch
import torch.distributed as dist
from torch import optim
from torch.nn.parallel import DistributedDataParallel as DDP
from torch.utils import data
from tqdm import tqdm
from trajdata import AgentType, UnifiedDataset
from trajdata.augmentation import NoiseHistories
from trajdata.data_structures.batch import AgentBatch
import wandb

from traj_pred.modules import ModelRegistrar
from traj_pred import TrajectoryPredictor
from traj_pred.utils import evaluation
# from utils import visualization_utils
from traj_pred.utils import args
from traj_pred.utils.comm import all_gather

# TODO: For nuScenes dataset
# def restrict_to_predchal(
#     dataset: UnifiedDataset,
#     split: str,
#     city: str = "",
# ) -> None:
#     curr_dir = pathlib.Path(__file__).parent.resolve()
#     with open(
#         curr_dir / f"experiments/nuScenes/predchal{city}_{split}_index.pkl", "rb"
#     ) as f:
#         within_challenge_split = pickle.load(f)

#     within_challenge_split = [
#         (dataset.cache_path / scene_info_path, num_elems, elems)
#         for scene_info_path, num_elems, elems in within_challenge_split
#     ]

#     dataset._scene_index = [orig_path for orig_path, _, _ in within_challenge_split]

#     # The data index is effectively a big list of tuples taking the form:
#     # (scene_path: str, index_len: int, valid_timesteps: np.ndarray[, agent_name: str])
#     dataset._data_index = AgentDataIndex(within_challenge_split, dataset.verbose)
#     dataset._data_len: int = len(dataset._data_index)

def set_seed(seed: Optional[int]):
    """Set manual seed to replicate, if provided"""
    if seed is None:
        return
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)


def get_device(rank: int) -> torch.device:
    """CPU or CUDA"""
    if torch.cuda.is_available():
        torch.cuda.set_device(rank)
        return torch.device(f"cuda:{rank}")
    return torch.device("cpu")


def init_distributed():
    """TODO"""
    backend = "nccl" if torch.cuda.is_available() else "gloo"
    dist.init_process_group(backend=backend)

    print(
        f"[{os.getpid()}]: world_size = {dist.get_world_size()}, "
        + f"rank = {dist.get_rank()}, backend={dist.get_backend()}, "
        + f"port = {os.environ['MASTER_PORT']} \n",
        end="",
    )

    return dist.get_rank(), dist.get_world_size()


def initialize_training(rank, world_size):
    """
    Load hyperparams; initiate wandb logging; create log directory
    """
    # Load hyperparameters from json
    if not os.path.exists(args.conf):
        raise ValueError(f"Config json at {args.conf} not found!")
    with open(args.conf, "r", encoding="utf-8") as conf_json:
        hyperparams = json.load(conf_json)
    hyperparams.update({k: v for k, v in vars(args).items() if v is not None})
    hyperparams['edge_encoding'] = not args.no_edge_encoding
    # scale learning rate
    hyperparams['learning_rate'] *= world_size

    # init wandb if needed
    if rank == 0 and not hyperparams.get('debug', False):
        if "eupeds" in hyperparams["train_data"]:
            train_scene = hyperparams["train_data"].split("-")[0][len("eupeds_") :]
        else:
            train_scene = hyperparams["train_data"][:4]
        #######################################################################
        # Make sure to specify your desired project and entity names if needed!
        run = wandb.init(
            project="free-space-predictor",
            entity="shaikh18-purdue-university",
            name=hyperparams["log_tag"],
            notes=f"{train_scene}",
            job_type="train",
            group=hyperparams["train_data"],
            config=hyperparams,
        )
        #######################################################################
        hyperparams = run.config

    # create log and model directory
    log_writer = None
    model_dir = None
    if not hyperparams["debug"]:
        model_dir_subfolder = hyperparams["log_tag"] + time.strftime(
            "-%d_%b_%Y_%H_%M_%S", time.localtime()
        )
        model_dir = os.path.join(hyperparams["log_dir"], model_dir_subfolder)

        if rank == 0:
            pathlib.Path(model_dir).mkdir(parents=True, exist_ok=True)

            # Save config to model directory
            with open(os.path.join(model_dir, "config.json"), "w", encoding="utf-8") as conf_json:
                json.dump(hyperparams.as_dict(), conf_json)

            log_writer = run

        print("model_dir:", model_dir_subfolder)

    return hyperparams, log_writer, model_dir


def print_params(hyperparams) -> None:
    """Print important parameters"""
    print("-----------------------")
    print("| TRAINING PARAMETERS |")
    print("-----------------------")
    print(f"| Max History: {hyperparams['history_sec']}s")
    print(f"| Max Future: {hyperparams['prediction_sec']}s")
    print(f"| Batch Size: {hyperparams['batch_size']}")
    print(f"| Eval Batch Size: {hyperparams['eval_batch_size']}")
    print(f"| Device: {hyperparams['device']}")
    print(f"| Learning Rate: {hyperparams['learning_rate']}")
    print(f"| Learning Rate Step Every: {hyperparams['lr_step']}")
    print(f"| Preprocess Workers: {hyperparams['preprocess_workers']}")
    print(f"| Robot Future: {hyperparams['incl_robot_node']}")
    print(f"| Map Encoding: {hyperparams['map_encoding']}")
    print(f"| Added Input Noise: {hyperparams['augment_input_noise']:.2f}")
    print(f"| Overall GMM Components: {hyperparams['K']}")
    print("-----------------------")


def build_datasets_and_loaders(hyperparams, rank, world_size):
    ''' Load training and evaluation environments and scenes '''
    # set up data dirs and parameters
    data_dirs = {
        "eupeds_eth": "./data/pedestrian_datasets/eth_ucy_peds",
    }
    attention_radius = defaultdict(lambda: 20.0)
    attention_radius[(AgentType.PEDESTRIAN, AgentType.PEDESTRIAN)] = 10.0

    augmentations = []
    if hyperparams['augment_input_noise'] > 0.0:
        augmentations.append(NoiseHistories(stddev=hyperparams['augment_input_noise']))

    map_params = {"px_per_m": 2, "map_size_px": 100, "offset_frac_xy": (-0.75, 0.0)}

    train_ds = UnifiedDataset(
        desired_data=[hyperparams["train_data"]],
        history_sec=(0.1, hyperparams["history_sec"]),
        future_sec=(0.1, hyperparams["prediction_sec"]),
        agent_interaction_distances=attention_radius,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        raster_map_params=map_params,
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        augmentations=augmentations if len(augmentations) > 0 else None,
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )
    # TODO: For nuScenes dataset
    # if hyperparams["train_data"] == "nusc_trainval-train":
    #     restrict_to_predchal(train_dataset, "train")
    train_sampler = data.distributed.DistributedSampler(
        train_ds,
        num_replicas=world_size,
        rank=rank
    )
    train_loader = data.DataLoader(
        train_ds,
        collate_fn=train_ds.get_collate_fn(pad_format="right"),
        pin_memory=False if hyperparams["device"] == "cpu" else True,
        batch_size=hyperparams["batch_size"],
        shuffle=False,
        num_workers=hyperparams["preprocess_workers"],
        sampler=train_sampler,
    )

    eval_ds = UnifiedDataset(
        desired_data=[hyperparams["eval_data"]],
        history_sec=(hyperparams["history_sec"], hyperparams["history_sec"]),
        future_sec=(hyperparams["prediction_sec"], hyperparams["prediction_sec"]),
        agent_interaction_distances=attention_radius,
        incl_robot_future=hyperparams["incl_robot_node"],
        incl_raster_map=hyperparams["map_encoding"],
        raster_map_params=map_params,
        only_predict=[AgentType.PEDESTRIAN],
        no_types=[AgentType.UNKNOWN],
        num_workers=hyperparams["preprocess_workers"],
        cache_location=hyperparams["trajdata_cache_dir"],
        data_dirs=data_dirs,
        verbose=True,
    )
    # TODO: For nuScenes dataset
    # if hyperparams["eval_data"] == "nusc_trainval-train_val":
    #     restrict_to_predchal(eval_dataset, "train_val")
    eval_sampler = data.distributed.DistributedSampler(eval_ds, num_replicas=world_size, rank=rank)
    eval_loader = data.DataLoader(
        eval_ds,
        collate_fn=eval_ds.get_collate_fn(pad_format="right"),
        pin_memory=False if hyperparams["device"] == "cpu" else True,
        batch_size=hyperparams["eval_batch_size"],
        shuffle=False,
        num_workers=hyperparams["preprocess_workers"],
        sampler=eval_sampler,
    )

    return train_sampler, train_loader, eval_loader


def init_model_and_optimizer(hyperparams, log_writer, device, model_dir):
    """Initiate model and optimizers"""
    model_registrar = ModelRegistrar(model_dir, device)
    fsp = TrajectoryPredictor(model_registrar, hyperparams, log_writer, device)
    fsp.set_environment()
    fsp.set_all_annealing_params()
    # optionally wrap in DDP
    if torch.cuda.is_available():
        fsp = DDP(
            fsp,
            device_ids=[device.index],
            output_device=device.index,
            find_unused_parameters=True,
        )
        fsp_module = fsp.module
    else:
        fsp_module = fsp
    # set optimizers and schedulers
    lr_scheduler = None
    step_scheduler = None
    optimizer = optim.Adam(
        [
            {
                "params": model_registrar.get_all_but_name_match(
                    "map_encoder"
                ).parameters()
            },
            {
                "params": model_registrar.get_name_match("map_encoder").parameters(),
                "lr": hyperparams["map_enc_learning_rate"],
            },
        ],
        lr=hyperparams["learning_rate"],
    )
    # Set Learning Rate
    if hyperparams["learning_rate_style"] == "const":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(optimizer, gamma=1.0)
    elif hyperparams["learning_rate_style"] == "exp":
        lr_scheduler = optim.lr_scheduler.ExponentialLR(
            optimizer, gamma=hyperparams["learning_decay_rate"]
        )
    if hyperparams["lr_step"] != 0:
        step_scheduler = optim.lr_scheduler.StepLR(
            optimizer, step_size=hyperparams["lr_step"], gamma=0.1
        )
    return fsp_module, model_registrar, optimizer, lr_scheduler, step_scheduler


def train_one_epoch(
        hyperparams,
        rank,
        model,
        dataloader,
        optimizer,
        lr_scheduler,
        log_writer,
        epoch,
        curr_iter: int,
    ):
    """Backpropagation"""
    pbar = tqdm(
        dataloader,
        ncols=80,
        unit_scale=dist.get_world_size(),
        disable=(rank > 0),
    )
    # initialize the timer for the 1st iteration
    step_timer_start = time.time()

    batch: AgentBatch
    for batch_idx, batch in enumerate(pbar):
        # print("\n----------------------------")
        # print(f"Batch Index: {batch_idx}")
        model.curr_iter = curr_iter
        model.step_all_annealers()
        optimizer.zero_grad(set_to_none=True)
        train_loss, loss_task, loss_nce = model(batch)
        pbar.set_description(
            f"Total Loss: {train_loss.detach().item():.4f}, " +
            f"Task Loss: {loss_task.item():.4f}, " +
            f"SNCE Loss: {loss_nce.item():.4f}"
        )
        train_loss.backward()
        optimizer.step()

        # Stepping forward the learning rate scheduler and annealers.
        lr_scheduler.step()
        if rank == 0 and not hyperparams.get("debug", False):
            step_timer_stop = time.time()
            elapsed = step_timer_stop - step_timer_start

            log_writer.log(
                {
                    "train/learning_rate": lr_scheduler.get_last_lr()[0],
                    "train/loss": train_loss.detach().item(),
                    "steps_per_sec": 1 / elapsed,
                    "epoch": epoch,
                    "batch": batch_idx,
                },
                step=curr_iter,
            )
        curr_iter += 1
        # initialize the timer for the following iteration
        step_timer_start = time.time()
    return curr_iter


def evaluate(
        rank,
        model: TrajectoryPredictor,
        dataloader,
        log_writer,
        epoch,
        curr_iter):
    """Evaluate model"""
    with torch.no_grad():
        # Calculate evaluation loss
        eval_perf = defaultdict(lambda: defaultdict(list))

        batch: AgentBatch
        for batch in tqdm(
            dataloader,
            ncols=80,
            unit_scale=dist.get_world_size(),
            disable=(rank > 0),
            desc=f"Epoch {epoch} Eval",
        ):
            results: Dict[AgentType, Dict[str, torch.Tensor]]
            results = model.predict_and_evaluate_batch(batch)
            for agent_type, metric_dict in results.items():
                for metric, values in metric_dict.items():
                    eval_perf[agent_type][metric].append(values.cpu().numpy())

        if torch.cuda.is_available() and dist.get_world_size() > 1:
            gathered_values = all_gather(eval_perf)
            if rank == 0:
                eval_perf = []
                for eval_dicts in gathered_values:
                    eval_perf.extend(eval_dicts)

        if rank == 0:
            evaluation.log_batch_errors(
                eval_perf,
                [
                    "ml_ade",
                    "ml_fde",
                    "min_ade_5",
                    "min_ade_10",
                    "nll_mean",
                    "nll_final",
                ],
                log_writer,
                "eval",
                epoch,
                curr_iter,
            )


def save_checkpoint(
    save_dir: str,
    model: TrajectoryPredictor,
    optimizer: optim.Optimizer,
    lr_scheduler: Optional[optim.lr_scheduler._LRScheduler],
    step_scheduler: Optional[optim.lr_scheduler.StepLR],
    epoch: int,
) -> None:
    """Save model checkpoint"""
    save_path = pathlib.Path(save_dir) / f"model_registrar-{epoch}.pt"

    torch.save(
        {
            "epoch": epoch,
            "model_state_dict": model.state_dict(),
            "optimizer_state_dict": optimizer.state_dict(),
            "lr_scheduler_state_dict": lr_scheduler.state_dict()
            if lr_scheduler is not None
            else None,
            "step_scheduler_state_dict": step_scheduler.state_dict()
            if step_scheduler is not None
            else None,
        },
        save_path,
    )


def main(rank, world_size, device):
    """Main function"""
    set_seed(args.seed)
    hyperparams, log_writer, model_dir = initialize_training(rank, world_size)
    print_params(hyperparams)

    (train_sampler,
     train_loader,
     eval_loader) = build_datasets_and_loaders(hyperparams, rank, world_size)

    (module,
     registrar,
     optimizer,
     lr_scheduler,
     step_scheduler) = init_model_and_optimizer(
        hyperparams, log_writer, device, model_dir)

    # torch.autograd.set_detect_anomaly(True)
    curr_iter: int = 0
    for epoch in range(1, hyperparams['train_epochs'] + 1):
        #################################
        #           TRAINING            #
        #################################
        train_sampler.set_epoch(epoch)
        curr_iter = train_one_epoch(
            hyperparams=hyperparams,
            rank=rank,
            model=module,
            dataloader=train_loader,
            optimizer=optimizer,
            lr_scheduler=lr_scheduler,
            log_writer=log_writer,
            epoch=epoch,
            curr_iter=curr_iter,
        )

        if hyperparams["lr_step"] != 0:
            step_scheduler.step()

        #################################
        #           EVALUATION          #
        #################################
        if (
            hyperparams["eval_every"] is not None
            and not hyperparams["debug"]
            and epoch % hyperparams["eval_every"] == 0
            and epoch > 0
        ):
            evaluate(
                rank=rank,
                model=module,
                dataloader=eval_loader,
                log_writer=log_writer,
                epoch=epoch,
                curr_iter=curr_iter
            )

        #################################
        #          CHECKPOINT           #
        #################################
        if rank == 0 and (
            hyperparams["save_every"] is not None
            and hyperparams["debug"] is False
            and epoch % hyperparams["save_every"] == 0
        ):
            save_checkpoint(
                save_dir=registrar.model_dir,
                model=module,
                optimizer=optimizer,
                lr_scheduler=lr_scheduler,
                step_scheduler=step_scheduler,
                epoch=epoch,
            )

        # Waiting for process 0 to be done its evaluation and visualization.
        if torch.cuda.is_available():
            dist.barrier()


if __name__ == '__main__':
    rank, world_size = init_distributed()
    device = get_device(rank)
    main(rank, world_size, device)
