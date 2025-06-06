import functools
import warnings

import torch
import torch.optim as optim

class CustomLR(torch.optim.lr_scheduler.LambdaLR):
    """Custom learning rate scheduler"""
    def __init__(self, optimizer, lr_lambda, last_epoch=-1):
        super(CustomLR, self).__init__(optimizer, lr_lambda, last_epoch)

    def get_lr(self):
        """Get the current learning rates from the scheduler"""
        return [
            lmbda(self.last_epoch)
            for lmbda in self.lr_lambdas
        ]

def rsetattr(obj, attr, val):
    """set nested attributes of an object"""
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)

# using wonder's beautiful simplification:
# https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-objects/31174427?noredirect=1#comment86638618_31174427
def rgetattr(obj, attr, *args):
    """get nested attributes of an object"""
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))

def exp_anneal(anneal_kws):
    """Exponential annealing function"""
    device = anneal_kws["device"]
    start = torch.tensor(anneal_kws["start"], device=device)
    finish = torch.tensor(anneal_kws["finish"], device=device)
    rate = torch.tensor(anneal_kws["rate"], device=device)
    return lambda step: finish - (finish - start) * torch.pow(
        rate, torch.tensor(step, dtype=torch.float, device=device)
    )

def sigmoid_anneal(anneal_kws):
    """Sigmoid annealing function"""
    device = anneal_kws["device"]
    start = torch.tensor(anneal_kws["start"], device=device)
    finish = torch.tensor(anneal_kws["finish"], device=device)
    center_step = torch.tensor(
        anneal_kws["center_step"], device=device, dtype=torch.float
    )
    steps_lo_to_hi = torch.tensor(
        anneal_kws["steps_lo_to_hi"], device=device, dtype=torch.float
    )
    return lambda step: start + (finish - start) * torch.sigmoid(
        (torch.tensor(float(step), device=device) - center_step)
        * (1.0 / steps_lo_to_hi)
    )

def create_new_scheduler(
    obj, name, annealer, annealer_kws, creation_condition=True
):
    """Create a new scheduler for the given object"""
    value_scheduler = None
    rsetattr(obj, name + "_scheduler", value_scheduler)
    if creation_condition:
        annealer_kws["device"] = obj.device
        value_annealer = annealer(annealer_kws)
        rsetattr(obj, name + "_annealer", value_annealer)

        # This is the value that we'll update on each call of
        # step_annealers().
        rsetattr(obj, name, value_annealer(0).clone().detach())
        dummy_optimizer = optim.Optimizer(
            [rgetattr(obj, name)], {"lr": value_annealer(0).clone().detach()}
        )
        rsetattr(obj, name + "_optimizer", dummy_optimizer)

        value_scheduler = CustomLR(dummy_optimizer, value_annealer)
        rsetattr(obj, name + "_scheduler", value_scheduler)

    obj.schedulers.append(value_scheduler)
    obj.annealed_vars.append(name)

def set_annealing_params(name, obj):
    """Set the annealing parameters for the object"""
    obj.schedulers = list()
    obj.annealed_vars = list()

    if name=="snce":
        create_new_scheduler(
            obj,
            name="temperature",
            annealer=exp_anneal,
            annealer_kws={
                "start": obj.hyperparams["contrastive_tau_init"],
                "finish": obj.hyperparams["contrastive_tau_final"],
                "rate": obj.hyperparams["tau_decay_rate"],
            },
        )
    else:
        create_new_scheduler(
            obj,
            name="kl_weight",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": obj.hyperparams["kl_weight_start"],
                "finish": obj.hyperparams["kl_weight"],
                "center_step": obj.hyperparams["kl_crossover"],
                "steps_lo_to_hi": obj.hyperparams["kl_crossover"]
                / obj.hyperparams["kl_sigmoid_divisor"],
            },
        )
        create_new_scheduler(
            obj,
            name="latent.temp",
            annealer=exp_anneal,
            annealer_kws={
                "start": obj.hyperparams["latent_tau_init"],
                "finish": obj.hyperparams["latent_tau_final"],
                "rate": obj.hyperparams["tau_decay_rate"],
            },
        )
        create_new_scheduler(
            obj,
            name="latent.z_logit_clip",
            annealer=sigmoid_anneal,
            annealer_kws={
                "start": obj.hyperparams["z_logit_clip_start"],
                "finish": obj.hyperparams["z_logit_clip_final"],
                "center_step": obj.hyperparams["z_logit_clip_crossover"],
                "steps_lo_to_hi": obj.hyperparams["z_logit_clip_crossover"]
                / obj.hyperparams["z_logit_clip_divisor"],
            },
            creation_condition=obj.hyperparams["use_z_logit_clipping"],
        )

def step_annealers(obj):
    """Step the annealers for the object"""
    # This should manage all of the step-wise changed
    # parameters automatically.
    for annealed_var in obj.annealed_vars:
        if rgetattr(obj, annealed_var + "_scheduler") is not None:
            # First we step the scheduler.
            with warnings.catch_warnings():  # We use a dummy optimizer: Warning because no .step() was called on it
                warnings.simplefilter("ignore")
                rgetattr(obj, annealed_var + "_scheduler").step()

            # Then we set the annealed vars' value.
            rsetattr(
                obj,
                annealed_var,
                rgetattr(obj, annealed_var + "_optimizer").param_groups[0]["lr"],
            )

    if obj.hyperparams.get("log_annealers", False):
        obj.summarize_annealers()

def summarize_annealers(obj):
    """Log the current values of the annealers"""
    if obj.log_writer is not None:
        for annealed_var in obj.annealed_vars:
            if rgetattr(obj, annealed_var) is not None:
                obj.log_writer.log(
                    {
                        f"{str(obj.node_type)}/{annealed_var.replace('.', '/')}": rgetattr(
                            obj, annealed_var
                        )
                    },
                    step=obj.curr_iter,
                    commit=False,
                )
