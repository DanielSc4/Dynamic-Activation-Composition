import json
from typing import Any

import numpy as np
import torch
from nnsight import LanguageModel
from tqdm import tqdm
from transformers import PreTrainedTokenizer

from src.utils.eval.multi_token_evaluator import Evaluator
from src.utils.prompt_helper import pad_input_and_ids

from .utils.model_utils import rgetattr
from .utils.prompt_helper import find_missing_ranges


def filter_activations(activation, important_ids):
    """
    # TODO: Deprecated
    Average activations of multi-token words across all its tokens
    """
    to_avg = find_missing_ranges(important_ids)
    for i, j in to_avg:
        activation[:, :, j] = activation[:, :, i : j + 1].mean(axis=2)

    activation = activation[:, :, important_ids]
    return activation


def split_activation(activations, config) -> torch.Tensor:
    """split the residual stream (d_model) into n_heads activations for each layer

    Args:
        activations (list[torch.Tensor]): list of residual streams for each layer shape: (list: n_layers[tensor: batch, seq, d_model])
        config (dict[str, Any]): model's config

    Returns:
        activations: reshaped activation in [batch, n_layers, n_heads, seq, d_head]
    """
    new_shape = torch.Size(
        [
            activations[0].shape[0],  # batch_size
            activations[0].shape[1],  # seq_len
            config["n_heads"],  # n_heads
            config["d_model"] // config["n_heads"],  # d_head
        ]
    )
    attn_activations = torch.stack([act.view(*new_shape) for act in activations])
    attn_activations = torch.einsum(
        "lbshd -> blhsd", attn_activations
    )  # layers batch seq heads dhead -> batch layers heads seq dhead
    return attn_activations


def extract_activations(
    icl_tokens: torch.Tensor,
    model: LanguageModel,
    config: dict[str, Any],
    tokenizer: PreTrainedTokenizer,
    max_new_tokens: int = 100,
) -> dict[str, torch.Tensor]:
    """Extract the activation and the output produced from the model using the tokenized prompts provided

    Args:
        icl_tokens (torch.tensor): input_ids for the model
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        device (str): device

    Returns:
        dict[str, torch.Tensor | None]: dictionary containing the activations and output
        ```python
        {
            'activations': torch.Tensor,
            'output': torch.Tensor,
            'logits': torch.Tensor
        }
        ```
    """
    all_activations = []
    for step in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(icl_tokens, scan=False) as _:
                icl_vocab = model.lm_head.output.save()
                layer_attn_activations = []
                for layer_i in range(len(config["attn_hook_names"])):
                    layer_attn_activations.append(
                        rgetattr(model, config["attn_hook_names"][layer_i])
                        .input[0][0]
                        .save()
                    )
        # get the values list[tensor: batch, (inp + step(s)), d_model]
        layer_attn_activations = [att.value for att in layer_attn_activations]

        # from hidden state split heads and permute: batch, n_layers, tokens, n_heads, d_head -> batch, n_layers, n_heads, tokens, d_head
        attn_activations = split_activation(layer_attn_activations, config)
        attn_activations = attn_activations[0, :, :, -1, :].detach().cpu()     # select batch 0 and only last token
        all_activations.append(attn_activations)

        # get the next token for the generation
        next_token = icl_vocab.value[0, -1, :].argmax(-1)
        # update the input (both icl and noicl) for the next generation iteration
        icl_tokens = torch.cat(
            [
                icl_tokens.squeeze(),
                torch.tensor([next_token]),
            ]
        )
        # stop criteria
        if next_token == tokenizer.eos_token_id:
            break

    all_activations = torch.stack(all_activations)  # shape: [n_steps, n_layers, n_heads, d_head]

    return {
        "activations": all_activations,
        "output": icl_tokens,
    }


def get_mean_activations(
    tokenized_prompts: list[torch.Tensor],
    tokenizer: PreTrainedTokenizer,
    model: LanguageModel,
    config: dict[str, Any],
    device: str,
    max_new_tokens: int = 100,
    save_output_path: str | None = None,
):
    """Compute the average of all the model's activation on the provided prompts

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized prompts
        important_ids (list[int]): list of important indexes i.e. the tokens where the average is computed
        tokenizer (AutoTokenizer): HuggingFace tokenizer
        model (AutoModelForCausalLM): HuggingFace model
        config (dict[str, any]): model's config
        correct_labels (list[str]): list of correct labels for each ICL prompt
        device (str): device
        batch_size (int): batch size for the model. Default to 10.
        max_len (int): max lenght of the prompt to be used (tokenizer parameter). Default to 256.
        save_output_path (str | None): output path to save the model generation. Default to None.

    Returns:
        torch.Tensor: mean of activations (`n_layers, n_heads, seq_len, d_head`)

    """
    # list[shape: n_layers, n_heads, d_head]
    all_activations = []
    all_outputs = []
    all_prompts = []

    for index in tqdm(
        range(len(tokenized_prompts)),
        total=len(tokenized_prompts),
        desc="[x] Extracting activations",
    ):
        activations_dict = extract_activations(
            icl_tokens=tokenized_prompts[index],
            model=model,
            config=config,
            tokenizer=tokenizer,
            max_new_tokens=max_new_tokens,
        )
        all_activations.append(activations_dict["activations"].detach().cpu())
        
        # store the decoded prompt and the decoded output
        prompt = tokenizer.decode(tokenized_prompts[index])
        prompt_len = tokenized_prompts[index].shape[0]
        output = tokenizer.decode(activations_dict["output"][prompt_len:])
        all_prompts.append(prompt)
        all_outputs.append(output)

    # saves outpus
    if save_output_path:
        json_to_write = [
            {
                "input": prompt,
                "output": output,
            }
            for prompt, output in zip(
                all_prompts,
                all_outputs,
            )
        ]
        with open(save_output_path, "w+", encoding="utf-8") as f:
            json.dump(json_to_write, f, indent=4)

    # all_activations     # shape: list[step, n_layers, n_heads, d_head] * number of prompts

    mean_activations = torch.zeros(
        max_new_tokens,
        config["n_layers"],
        config["n_heads"],
        config["d_model"] // config["n_heads"],
    )

    # average on every step (the number of steps my vary for each prompt)
    for step in range(max_new_tokens):
        act_for_this_step = []
        for prompt_activation in all_activations:
            if step < prompt_activation.shape[0]:
                act_for_this_step.append(prompt_activation[step, :, :, :])

        # should be [n_prompts, n_layers, n_heads, d_head]
        mean_activations[step, :, :, :] = torch.stack(act_for_this_step).mean(dim=0)


    # mean_activations shape: [step, n_layers, n_heads, d_head]
    return mean_activations
