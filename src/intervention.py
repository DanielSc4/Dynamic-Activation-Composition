import json
from typing import Any

import torch
from torch.nn.functional import softmax
from nnsight import LanguageModel
from tqdm import tqdm

from src.utils.model_utils import rgetattr
from src.utils.prompt_helper import find_missing_ranges


def filter_activations(activation, important_ids):
    """
    Average activations of multi-token words across all its tokens
    """
    to_avg = find_missing_ranges(important_ids)
    for i, j in to_avg:
        activation[:, :, j] = activation[:, :, i : j + 1].mean(axis=2)

    activation = activation[:, :, important_ids]
    return activation


def simple_forward_pass(
    model: LanguageModel,
    tokenized_prompt: torch.Tensor | dict[str, torch.Tensor],
    pad_token_id: int | None = None,
    max_new_tokens: int = 100,
) -> torch.Tensor:
    """
    Perform a single forward pass with no intervention. Return a tensor [batch, full_output_len].
    """
    # use generate function and return the full output
    with model.generate(
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        validate=False,
    ) as generator:
        with generator.invoke(tokenized_prompt, scan=False) as _:
            ret = model.generator.output.save()

    return ret.value



def replace_heads_w_avg_multi_token_logits(
    tokenized_prompt: torch.Tensor,
    model: LanguageModel,
    config: dict[str, Any],
    layer_head: tuple[int, int] | None = None,
    avg_activation: torch.Tensor | None = None,
    max_new_tokens: int = 100,
    ) -> dict[str, torch.Tensor]:
    """
    Perform a generation pass with the model.
    A specific attention head (`layer_head = (num_layer, num_head)`) activation is replaced with the avg_activation at the last token position.
    If `layer_head` is None, the avg_activation is not used, and the model is used as is.
    Return the logits of the model at each generation step (torch.Tensor[steps, vocab_size])
    """

    if layer_head is not None:
        assert avg_activation is not None, "avg_activation must be provided if layer_head is provided"
    if avg_activation is not None:
        assert layer_head is not None, "layer_head must be provided if avg_activation is provided"

    d_head = config["d_model"] // config["n_heads"]
    all_logits = []

    for _ in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(tokenized_prompt, scan=False) as _:
                logits = model.lm_head.output.save()

                if layer_head is not None:
                    num_layer, num_head = layer_head
                    attention_head_values = rgetattr(
                        model, config["attn_hook_names"][num_layer]
                    ).input[0][0][:, :, (num_head * d_head) : ((num_head + 1) * d_head)]
                    # shape: [batch = 1, seq (e.g. 256), d_head]
                    attention_head_values[:, len(tokenized_prompt) - 1, :] = (
                        avg_activation
                    )

        all_logits.append(
            logits.value.squeeze()[-1, :]   # take the last token logits
        )
        next_token = logits.value[0, -1, :].argmax(-1)
        tokenized_prompt = torch.cat(
            [
                tokenized_prompt.squeeze(),
                torch.tensor([next_token]),
            ]
        )
        if next_token == model.tokenizer.eos_token_id:
            break

    return {
        'logits': torch.stack(all_logits),
        'output': tokenized_prompt,
    }



def replace_heads_w_avg_multi_token(
    tokenized_prompt: dict[str, torch.Tensor] | torch.Tensor,
    layers_heads: list[tuple[int, int]],
    avg_activations: list[torch.Tensor],
    model: LanguageModel,
    config: dict[str, Any],
    pad_token_id: int | None,
    max_new_tokens: int = 70,
) -> torch.Tensor:
    """
    Replace the activation of specific head(s) (listed in `layers_heads`) with the avg_activation for each
    specific head (listed in `avg_activations`) at the last token positions.
    Than return the output of the model's activations replaced.

    Only batchsize = 1 is allowed
    Args:
        tokenized_prompt (dict[str, torch.Tensor] | torch.Tensor): tokenized prompt (both input_ids and opt attention_mask are valid)
        layers_heads (list[tuple[int, int]]): list of tuples with the layer and head to replace (e.g. `[(0, 0), (1, 1)]` replace the first head of the first layer and the second head of the second layer)
        avg_activations (list[torch.Tensor]): list of activations to replace the heads with. Must be the same length as `layers_heads`.
        model (LanguageModel): model
        config (dict[str, Any]): model's config
        pad_token_id (int): pad token id for generation
        max_new_tokens (int, optional): max new tokens to generate. Defaults to 70.
    Return:
        torch.Tensor: output of the model with the replaced activations
    """

    assert len(layers_heads) == len(
        avg_activations
    ), f"layers_heads and avg_activations must have the same length. Got {len(layers_heads)} and {len(avg_activations)}"

    d_head = config["d_model"] // config["n_heads"]

    """
    Developer note: here avg_activations is a list of activations for each head to change
        if calcultaing AIE, the length of the list is 1.
    """

    with model.generate(
        max_new_tokens=max_new_tokens,
        pad_token_id=pad_token_id,
        validate=False,
    ) as generator:
        with generator.invoke(
            tokenized_prompt,
            scan=False,
        ) as _:
            for idx, (num_layer, num_head) in enumerate(layers_heads):
                attention_head_values = rgetattr(
                    model, config["attn_hook_names"][num_layer]
                ).input[0][0][:, :, (num_head * d_head) : ((num_head + 1) * d_head)]
                # shape: [batch = 1, seq (e.g. 256), d_head]
                attention_head_values[:, len(tokenized_prompt) - 1, :] = (
                    avg_activations[idx]
                )
            output = model.generator.output.save()

    return output.value


def _aie_loop(
    model: LanguageModel,
    config: dict[str, Any],
    main_pbar: tqdm,
    mean_activations: torch.Tensor,
    tokenized_prompt: torch.Tensor,
    max_new_tokens: int,
        ) -> dict[str, list[list[torch.Tensor]]]:
    """
    Loop over the layers and heads to replace the activations with the mean_activations and return the output of the model.

    Args:
        model (LanguageModel): model
        tokenizer (PreTrainedTokenizer): tokenizer
        config (dict[str, Any]): model's config
        main_pbar (tqdm): main progress bar to be updated during the loop
        mean_activations (torch.Tensor): mean activations for each head. Should be `[n_layers, n_heads, seq_len, d_head]`
        prompt (dict[str, torch.Tensor] | torch.Tensor): tokenized prompt (both input_ids and opt attention_mask are valid)
        important_ids (list[int] | None): list of important ids to replace. Defaults to None. --Deprecated--
    Returns:
        list[list[torch.Tensor]] | torch.Tensor: list of outputs for each layer and head where `len(outputs) = n_layers` and `len(outputs[i]) = n_heads`
        each output is a tensor of shape `[steps, vocab_size]`.
    """

    inner_bar_layers = tqdm(
        range(config["n_layers"]),
        total=config["n_layers"],
        leave=False,
        desc="  -th layer",
    )
    layers_logits = []
    layers_outputs = []

    for layer_i in inner_bar_layers:
        inner_bar_heads = tqdm(
            range(config["n_heads"]),
            total=config["n_heads"],
            leave=False,
            desc="    -th head",
        )
        heads_logits = []
        heads_outputs = []
        for head_j in inner_bar_heads:
            main_pbar.set_description(
                f'Processing edited model (l: {layer_i}/{config["n_layers"]}, h: {head_j}/{config["n_heads"]})'
            )
            # here the return value has already the softmaxed scored from the evaluator object
            logits_and_output = replace_heads_w_avg_multi_token_logits(
                tokenized_prompt=tokenized_prompt,
                model=model,
                config=config,
                layer_head=(layer_i, head_j),
                avg_activation=mean_activations[layer_i, head_j],
                max_new_tokens=max_new_tokens,
            )
            heads_logits.append(logits_and_output["logits"].detach().cpu())
            heads_outputs.append(logits_and_output["output"].detach().cpu())

        layers_logits.append(heads_logits)
        layers_outputs.append(heads_outputs)

    return {
        "logits": layers_logits,
        "output": layers_outputs,
    }


def _compute_scores_multi_token(
    model: LanguageModel,
    config: dict[str, Any],
    icl_tokenized: list[torch.Tensor],
    noicl_tokenized: list[torch.Tensor],
    mean_activations: torch.Tensor,
    save_output_path: str | None = None,
    max_new_tokens: int = 100,
) -> torch.Tensor:
    """
    Compute the scores for the original model and the edited model (model with the mean_activations for each head).
    """

    # intervention
    logits_and_outputs_original = {
        "logits": [],   # list of tensors [steps, vocab_size]
        "outputs": [],  # list of tensors [steps]
        "target_tokens": [],    # list of tensors [steps]
    }
    logits_and_outputs_edited = {
        "logits": [],   # list of list of tensors [n_layers x n_heads x tensor[steps, vocab_size]]
        "outputs": [],  # list of list of tensors [n_layers x n_heads x tensor[steps]]
    }
    for idx in (
        pbar := tqdm(
            range(len(icl_tokenized)),
            total=len(icl_tokenized),
        )
    ):
        pbar.set_description("Original forward pass")
        # simple forward pass
        logits_and_output = replace_heads_w_avg_multi_token_logits(
            model=model,
            config=config,
            tokenized_prompt=icl_tokenized[idx],
            max_new_tokens=max_new_tokens
        )
        logits_and_outputs_original["logits"].append(logits_and_output["logits"].detach().cpu())
        logits_and_outputs_original["outputs"].append(logits_and_output["output"].detach().cpu())
        
        pbar.set_description("Edited forward pass")
        # for each prompt the function returns a list of a list [n_layers x n_heads]
        edited_out = _aie_loop(
            model=model,
            config=config,
            main_pbar=pbar,
            mean_activations=mean_activations,
            tokenized_prompt=noicl_tokenized[idx],
            max_new_tokens=max_new_tokens,
        )
        logits_and_outputs_edited["logits"].append(edited_out["logits"])
        logits_and_outputs_edited["outputs"].append(edited_out["output"])

    # compute KL between logits from the original and the edited model
    scores = torch.zeros(
        len(icl_tokenized), 
        config["n_layers"], 
        config["n_heads"],
    )
    for idx in range(len(icl_tokenized)):
        for layer in range(config["n_layers"]):
            for head in range(config["n_heads"]):
                # NOTE: not going trough the generation steps, only considering the first generated token
                
                # take the first generated token from the original as the gold token (this is the token that the edited model should generate)
                first_gen_token_original = logits_and_outputs_original["logits"][idx].argmax(-1)[0].item()
                first_gen_token_edited = logits_and_outputs_edited["logits"][idx][layer][head][0].argmax(-1).item()

                # compute the softmax of both models and take the first generated token vocab
                first_gen_softmaxed_original = softmax(logits_and_outputs_original['logits'][idx], dim=-1)[0, :]
                first_gen_softmaxed_edited = softmax(logits_and_outputs_edited['logits'][idx][layer][head], dim=-1)[0, :]

                # take the score of the gold token from the first generated token of the edited model
                score_original = first_gen_softmaxed_original[first_gen_token_original].item()
                score_edited = first_gen_softmaxed_edited[first_gen_token_original].item()

                # NOTE: the closer to zero the better the edited model predicted the gold token (in general, the higher the better)
                scores[idx, layer, head] = score_edited - score_original

    # saving everything (logs_output)
    print("[x] Saving logs")
    logs_output = []
    for idx in range(len(icl_tokenized)):
        icl_prompt_len = len(icl_tokenized[idx])
        noicl_prompt_len = len(noicl_tokenized[idx])
        logs_output.append({
            "input": model.tokenizer.decode(icl_tokenized[idx], skip_special_tokens=True),
            "original": model.tokenizer.decode(
                logits_and_outputs_original["outputs"][idx][icl_prompt_len:],
                skip_special_tokens=True
            ),
            "edited": [
                (
                    f"{layer},{head}",
                    f"score: {scores[idx, layer, head]}",
                    model.tokenizer.decode(
                        logits_and_outputs_edited["outputs"][idx][layer][head][noicl_prompt_len:],
                        skip_special_tokens=True
                    ),
                )
                for layer in range(config["n_layers"])
                for head in range(config["n_heads"])
            ],
        })

    if save_output_path:
        with open(save_output_path, "w+") as fout:
            json.dump(logs_output, fout, indent=4)

    return scores


def compute_indirect_effect(
    model,
    config: dict[str, Any],
    icl_tokenized: list[torch.Tensor],
    noicl_tokenized: list[torch.Tensor],
    mean_activations: torch.Tensor,
    save_output_path: str | None = None,
    max_new_tokens: int = 100,
):
    """Compute indirect effect on the provided dataset by comparing the prediction of the original model
    to the predicition of the modified model. Specifically, for the modified model, each attention head
    activation is substituted with the corresponding mean_activation provided to measure the impact
    on the final correct label predicition.

    Args:
        model (_type_): Language model
        tokenizer (tokenizer): Tokenizer
        config (dict[str, Any]): model's config dictionary
        dataset (list[tuple[str, str]]): list of tuples with the first element being the prompt and the second the correct label
        mean_activations (torch.Tensor): mean activation for each head in the model. Should be [n_layers, n_heads, seq_len, d_head]
        ICL_examples (int, optional): number of ICL examples exluding the final prompt. Defaults to 4.
        batch_size (int, optional): batch size dimension. Defaults to 32.
        aie_support (int, optional): number of prompt supporting the average indirect effect on the model. Defaults to 25.

    Returns:
        _type_: TBD
    """
    assert len(icl_tokenized) == len(noicl_tokenized), "icl_tokenized and noicl_tokenized must have the same length"

    scores = _compute_scores_multi_token(
        model=model,
        config=config,
        icl_tokenized=icl_tokenized,
        noicl_tokenized=noicl_tokenized,
        mean_activations=mean_activations,
        save_output_path=save_output_path,
        max_new_tokens=max_new_tokens,
    )

    return scores
