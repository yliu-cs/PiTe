import os
import math
import torch
import random
import warnings
import numpy as np
import transformers
from torch import nn
from peft import PeftModel
from deepspeed import zero
from torchvision.io import read_video
from numerize.numerize import numerize
from deepspeed.runtime.zero.partition_parameters import ZeroParamStatus


def ignore_warnings() -> None:
    warnings.filterwarnings("ignore")
    transformers.logging.set_verbosity_error()


def seed_everything(
    seed: int
) -> None:
    os.environ["PYTHONHASHSEED"] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True


def count_params(model: nn.Module) -> tuple[int, int]:
    total_params, tunable_params = 0, 0
    for param in model.parameters():
        n_params = param.numel()
        if n_params == 0 and hasattr(param, "ds_numel"):
            n_params = param.ds_numel
        if param.__class__.__name__ == "Params4bit":
            n_params *= 2
        total_params += n_params
        if param.requires_grad:
            tunable_params += n_params
    return tunable_params, total_params


def print_tunable_parameters(model: nn.Module) -> None:
    tunable_params, total_params = count_params(model)
    s = "\033[32m"
    s += f"Tunable Parameters: {numerize(tunable_params)} || "
    s += f"All Parameters: {numerize(total_params)} || "
    s += f"Tunable%: {tunable_params / total_params * 100:.3f}%"
    s += "\033[0m"
    print(s)


def extract_video(video_path: str, fps: int = 100) -> np.ndarray:
    images = read_video(video_path)[0]
    indices = np.linspace(0, images.size(0) - 1, fps, dtype=np.int32)
    images = torch.index_select(images, 0, torch.from_numpy(indices).long())
    return images


# Following From LLaVA
def maybe_zero_3(param, ignore_status=False, name=None):
    if hasattr(param, "ds_id"):
        if param.ds_status == ZeroParamStatus.NOT_AVAILABLE:
            if not ignore_status:
                print(name, "no ignore status")
        with zero.GatheredParameters([param]):
            param = param.data.detach().cpu().clone()
    else:
        param = param.detach().cpu().clone()
    return param


def get_peft_state_maybe_zero_3(named_params, bias):
    if bias == "none":
        to_return = {k: t for k, t in named_params if "lora_" in k}
    elif bias == "all":
        to_return = {k: t for k, t in named_params if "lora_" in k or "bias" in k}
    elif bias == "lora_only":
        to_return = {}
        maybe_lora_bias = {}
        lora_bias_names = set()
        for k, t in named_params:
            if "lora_" in k:
                to_return[k] = t
                bias_name = k.split("lora_")[0] + "bias"
                lora_bias_names.add(bias_name)
            elif "bias" in k:
                maybe_lora_bias[k] = t
        for k, t in maybe_lora_bias:
            if bias_name in lora_bias_names:
                to_return[bias_name] = t
    else:
        raise NotImplementedError
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k) for k, v in to_return.items()}
    return to_return


def get_peft_state_non_lora_maybe_zero_3(named_params, require_grad_only=True):
    to_return = {k: t for k, t in named_params if "lora_" not in k}
    if require_grad_only:
        to_return = {k: t for k, t in to_return.items() if t.requires_grad}
    to_return = {k: maybe_zero_3(v, ignore_status=True).cpu() for k, v in to_return.items()}
    return to_return


def get_mlp_state_maybe_zero_3(named_params, keys_to_match):
    to_return = {k: t for k, t in named_params if any(key_match in k for key_match in keys_to_match)}
    to_return = {k: maybe_zero_3(v, ignore_status=True, name=k).cpu() for k, v in to_return.items()}
    return to_return


def find_all_linear_names(model: nn.Module):
    cls = torch.nn.Linear
    lora_module_names = set()
    for name, module in model.named_modules():
        if isinstance(module, cls):
            names = name.split(".")
            lora_module_names.add(names[0] if len(names) == 1 else names[-1])
    if "lm_head" in lora_module_names:
        lora_module_names.remove("lm_head")
    return list(lora_module_names)


# Following From Video-LLaVA
def split_list(lst: list[dict], n: int):
    chunk_size = math.ceil(len(lst) / n)
    return [lst[i:i + chunk_size] for i in range(0, len(lst), chunk_size)]


def get_chunk(lst: list[dict], n: int, k: int):
    chunks = split_list(lst, n)
    return chunks[k]


# Following From VTimeLLM
def load_lora(model, lora_path):
    non_lora_trainables_path = os.path.join(lora_path, "non_lora_trainables.bin")
    if os.path.exists(non_lora_trainables_path):
        non_lora_trainables = torch.load(non_lora_trainables_path, map_location="cpu")
        non_lora_trainables = {(k[11:] if k.startswith("base_model.") else k): v for k, v in non_lora_trainables.items()}
        if any(k.startswith("model.model.") for k in non_lora_trainables):
            non_lora_trainables = {(k[6:] if k.startswith("model.") else k): v for k, v in non_lora_trainables.items()}
        model.load_state_dict(non_lora_trainables, strict=False)
    print("\033[33mLoading LoRA weights...\033[0m")
    model = PeftModel.from_pretrained(model, lora_path)
    return model


def disable_torch_init():
    setattr(nn.Linear, "reset_parameters", lambda self: None)
    setattr(nn.LayerNorm, "reset_parameters", lambda self: None)