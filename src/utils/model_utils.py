import functools
import os
import random

import numpy as np
import torch
from nnsight import LanguageModel
from transformers import BitsAndBytesConfig


# thanks to https://stackoverflow.com/questions/31174295/getattr-and-setattr-on-nested-subobjects-chained-properties
def rsetattr(obj, attr, val):
    pre, _, post = attr.rpartition(".")
    return setattr(rgetattr(obj, pre) if pre else obj, post, val)


def rgetattr(obj, attr, *args):
    def _getattr(obj, attr):
        return getattr(obj, attr, *args)

    return functools.reduce(_getattr, [obj] + attr.split("."))


def get_device():
    if torch.cuda.is_available():
        device = "cuda"
    # elif torch.backends.mps.is_available():
    #     device = 'mps'
    else:
        device = "cpu"
    return device


def load_model_and_tokenizer(
    model_name: str,
    load_in_8bit: bool = False,
):

    device = get_device()

    if "gpt2" in model_name.lower():
        model = LanguageModel(
            "gpt2",
            device_map=device if not load_in_8bit else {"": 0},
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
            ),
            low_cpu_mem_usage=True if load_in_8bit else None,
            torch_dtype=torch.bfloat16,
        )
        # providing a standard config
        std_CONFIG = {
            "n_heads": model.config.n_head,
            "n_layers": model.config.n_layer,
            "d_model": model.config.n_embd,  # residual stream
            "name": model.config.name_or_path,
            "vocab_size": model.config.vocab_size,
            "layer_name": "transformer.h",
            "layer_hook_names": [
                f"transformer.h.{layer}" for layer in range(model.config.n_layer)
            ],
            "attn_name": "attn",
            "attn_hook_names": [
                f"transformer.h.{layer}.attn" for layer in range(model.config.n_layer)
            ],
        }

    elif "gpt-j" in model_name.lower():
        model = LanguageModel(
            model_name,
            device_map=device if not load_in_8bit else {"": 0},
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
            ),
            low_cpu_mem_usage=True if load_in_8bit else None,
            torch_dtype=torch.bfloat16,
        )
        std_CONFIG = {
            "n_heads": model.config.n_head,
            "n_layers": model.config.n_layer,
            "d_model": model.config.n_embd,  # residual stream
            "name": model.config.name_or_path,
            "vocab_size": model.config.vocab_size,
            "layer_name": "transformer.h",
            "layer_hook_names": [
                f"transformer.h.{layer}" for layer in range(model.config.n_layer)
            ],
            "attn_name": "attn.out_proj",
            "attn_hook_names": [
                f"transformer.h.{layer}.attn.out_proj"
                for layer in range(model.config.n_layer)
            ],
        }

    elif "gpt-neox" in model_name.lower():
        model = LanguageModel(
            model_name,
            device_map=device if not load_in_8bit else {"": 0},
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
            ),
            low_cpu_mem_usage=True if load_in_8bit else None,
            torch_dtype=torch.bfloat16,
        )
        std_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "d_model": model.config.hidden_size,  # residual stream
            "name": model.config.name_or_path,
            "vocab_size": model.config.vocab_size,
            "layer_name": "gpt_neox.layers",
            "layer_hook_names": [
                f"gpt_neox.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "attn_name": "attention",
            "attn_hook_names": [
                f"gpt_neox.layers.{layer}.attention"
                for layer in range(model.config.num_hidden_layers)
            ],
        }

    elif "pythia" in model_name.lower():
        model = LanguageModel(
            model_name,
            device_map=device if not load_in_8bit else {"": 0},
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
            ),
            low_cpu_mem_usage=True if load_in_8bit else None,
            torch_dtype=torch.bfloat16,
        )
        std_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "d_model": model.config.hidden_size,  # residual stream
            "name": model.config.model_type,
            "vocab_size": model.config.vocab_size,
            "layer_name": "gpt_neox.layers",
            "layer_hook_names": [
                f"gpt_neox.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "attn_name": "attention",
            "attn_hook_names": [
                f"gpt_neox.layers.{layer}.attention"
                for layer in range(model.config.num_hidden_layers)
            ],
        }

    elif "llama" in model_name.lower():
        model = LanguageModel(
            model_name,
            device_map=device if not load_in_8bit else {"": 0},
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
            ),
            low_cpu_mem_usage=True if load_in_8bit else None,
            torch_dtype=torch.bfloat16,
        )
        std_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "d_model": model.config.hidden_size,  # residual stream
            "name": model.config._name_or_path,
            "vocab_size": model.config.vocab_size,
            "layer_name": "model.layers",
            "layer_hook_names": [
                f"model.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "attn_name": "self_attn",
            "attn_hook_names": [
                f"model.layers.{layer}.self_attn.o_proj"
                for layer in range(model.config.num_hidden_layers)
            ],
        }

    elif (
            "zephyr" in model_name.lower() or 
            "stablelm" in model_name.lower() or 
            "gemma" in model_name.lower() or
            "mistral" in model_name.lower()
        ):
        model = LanguageModel(
            model_name,
            device_map=device if not load_in_8bit else {"": 0},
            trust_remote_code=True,
            quantization_config=(
                BitsAndBytesConfig(load_in_8bit=True) if load_in_8bit else None
            ),
            low_cpu_mem_usage=True if load_in_8bit else None,
            torch_dtype=torch.bfloat16,
        )
        std_CONFIG = {
            "n_heads": model.config.num_attention_heads,
            "n_layers": model.config.num_hidden_layers,
            "d_model": model.config.hidden_size,  # residual stream
            "name": model.config._name_or_path,
            "vocab_size": model.config.vocab_size,
            "layer_name": "model.layers",
            "layer_hook_names": [
                f"model.layers.{layer}"
                for layer in range(model.config.num_hidden_layers)
            ],
            "attn_name": "self_attn.o_proj",
            "attn_hook_names": [
                f"model.layers.{layer}.self_attn.o_proj"
                for layer in range(model.config.num_hidden_layers)
            ],
        }

    else:
        raise NotImplementedError(f"Model config not yet implemented ({model_name})")

    tokenizer = model.tokenizer
    if not tokenizer.pad_token_id:
        tokenizer.pad_token_id = tokenizer.eos_token_id

    return model, tokenizer, std_CONFIG, device


def set_seed(seed: int) -> None:
    """
    Sets the seed to make everything deterministic, for reproducibility of experiments

    Parameters:
    seed: the number to set the seed to

    Return: None
    """

    # Random seed
    random.seed(seed)

    # Numpy seed
    np.random.seed(seed)

    # Torch seed
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = True

    # os seed
    os.environ["PYTHONHASHSEED"] = str(seed)


def get_top_attention_heads(
    cie=torch.Tensor,
    num_heads: int = 10,
) -> list[tuple[int, int]]:
    """
    Get the indices of the top attention heads in CIE
    """
    # indeces of the top num_heads highest numbers
    flat_indices = np.argsort(cie.flatten().cpu())[-num_heads:]
    # convert flat indices to 2D incices
    top_indices = np.unravel_index(flat_indices, cie.shape)
    coordinates_list = list(zip(top_indices[0], top_indices[1]))

    # sort the list based on the corresponding values in descending order
    sorted_coordinates_list = sorted(
        coordinates_list,
        key=lambda x: cie[x[0], x[1]],
        reverse=True,
    )
    return sorted_coordinates_list
