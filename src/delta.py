import torch

from typing import Any
from nnsight import LanguageModel
from warnings import warn
import torch.nn.functional as F

from src.utils.model_utils import rgetattr
from torch.nn.functional import cross_entropy as ce_loss


def _stop_condition(token: int, tokenizer: Any) -> bool:
    if token == tokenizer.eos_token_id:
        return True
    # if "\n" in tokenizer.decode(token):
    #     return True
    if "Q:" in tokenizer.decode(token):     # starting a new question (usually observed in icl setting)
        return True
    return False

def generate_token_step(
    model: LanguageModel,
    prompt: torch.Tensor,
    max_new_tokens: int,
    return_ppl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[float]]:
    """
    Function to generate tokens step by step using the clean model.
    Can return the perplexity values for each token generated if return_ppl is True. (used with icl prompt)
    """
    
    nlls = []

    for step in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(prompt, scan=False) as _:
                icl_logits = model.lm_head.output.save()
        last_token_logits = icl_logits.value.squeeze()[-1].detach()       # dim: [vocab_size]
        predicted_token = last_token_logits.argmax(-1).detach()

        nll = ce_loss(
            last_token_logits, 
            predicted_token,
        ).detach().cpu()
        nlls.append(nll)

        # stopping conditions
        if _stop_condition(predicted_token.item(), model.tokenizer):
            break

        prompt = torch.cat([
            prompt.squeeze(),
            predicted_token.unsqueeze(0),
        ])

    if return_ppl:
        return prompt, nlls
    return prompt


def generate_edited_model(
    model: LanguageModel,
    config: dict[str, Any],
    prompt: torch.Tensor,
    max_new_tokens: int,
    diff_mean_activations: torch.Tensor,
    alpha_factor: float | list[float] = 1.0,
    diminishing_alpha: bool = False,
    only_start_token: bool = False,
    return_ppl: bool = False,
) -> torch.Tensor | tuple[torch.Tensor, list[float]]:

    # raise a warning ig only_start_token is True and diminishing_alpha is True
    if only_start_token and diminishing_alpha:
        warn("diminishing_alpha is not supported when only_start_token is True, setting diminishing_alpha to False")
        diminishing_alpha = False

    total_alphas = []
    if diminishing_alpha:
        assert isinstance(alpha_factor, float), "alpha_factor should be a float when diminishing_alpha is True"
        decrement = alpha_factor / max_new_tokens
        for i in range(max_new_tokens):
            total_alphas.append(alpha_factor - (decrement * i))
    elif only_start_token:
        assert isinstance(alpha_factor, float), "alpha_factor should be a float when starting_alpha is True"
        total_alphas = [alpha_factor] * max_new_tokens
    elif isinstance(alpha_factor, float):
        total_alphas = [alpha_factor] * max_new_tokens
    else:
        # the only other case is when alpha_factor is a list (when using dynamic alpha)
        assert isinstance(alpha_factor, list), "alpha_factor should be a list when not using diminishing_alpha or only_start_token"
        total_alphas = alpha_factor
        max_new_tokens = len(alpha_factor)      # could be shorter than the current max_new_tokens because of stopping critiria in dynamic alpha

    assert isinstance(total_alphas, list), "alpha_factor, after the checks should be a list or a float, check the code!"

    initial_prompt_len = prompt.shape[0]

    nlls = []
    for step in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(prompt, scan=False) as _:
                # editing model
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]       # [batch, total_tokens, d_model]
                    if only_start_token:
                        # replace activations, ### A
                        alpha_step = total_alphas[step]
                        layer_act[0, initial_prompt_len - 1, :] += alpha_step * diff_mean_activations[0, layer_i, :].reshape(-1)
                    else:
                        # replace activations, ### C
                        for i in range(0, step + 1):
                            alpha_step = total_alphas[i]
                            layer_act[0, initial_prompt_len - 1 + i, :] += alpha_step * diff_mean_activations[i, layer_i, :].reshape(-1)

                # getting the prediction
                edited_logits = model.lm_head.output.save()
        last_token_logits = edited_logits.value.squeeze()[-1].detach()       # dim: [vocab_size]
        predicted_token = last_token_logits.argmax(-1).detach()

        # stopping conditions
        if _stop_condition(predicted_token.item(), model.tokenizer):
            break

        # used only for icl, to compute perplexity on its own generation
        nll = ce_loss(
            last_token_logits, 
            predicted_token,
        ).detach().cpu()
        nlls.append(nll)

        prompt = torch.cat([
            prompt.squeeze(),
            predicted_token.unsqueeze(0),
        ])

    if return_ppl:
        return prompt, nlls
    return prompt


def controlled_gen_edited_model(
    model: LanguageModel,
    config: dict[str, Any],
    prompt: torch.Tensor,
    max_new_tokens: int,
    diff_mean_activations: torch.Tensor,
    gold_out: torch.Tensor,
    alpha_factor: float | list[float] = 1.0,
    diminishing_alpha: bool = False,
    only_start_token: bool = False,
    ):

    initial_prompt_len = prompt.shape[0]
    nlls = []

    total_alphas = []
    if diminishing_alpha:
        assert isinstance(alpha_factor, float), "alpha_factor should be a float when diminishing_alpha is True"
        decrement = alpha_factor / max_new_tokens
        for i in range(max_new_tokens):
            total_alphas.append(alpha_factor - (decrement * i))
    elif only_start_token:
        assert isinstance(alpha_factor, float), "alpha_factor should be a float when starting_alpha is True"
        total_alphas = [alpha_factor] * max_new_tokens
    elif isinstance(alpha_factor, float):
        total_alphas = [alpha_factor] * max_new_tokens
    else:
        # the only other case is when alpha_factor is a list (when using dynamic alpha)
        assert isinstance(alpha_factor, list), "alpha_factor should be a list when not using diminishing_alpha or only_start_token"
        total_alphas = alpha_factor
        max_new_tokens = len(alpha_factor)      # could be shorter than the current max_new_tokens because of stopping critiria in dynamic alpha

    assert isinstance(total_alphas, list), "alpha_factor, after the checks should be a list or a float, check the code!"
    
    # overwrite max_new_tokens in case there are less tokens in gold_out
    max_new_tokens = min(gold_out.shape[0], max_new_tokens)

    for step in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(prompt, scan=False) as _:
                # editing model
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]
                    if only_start_token:
                        # replace activations, ### A
                        alpha_step = total_alphas[step]
                        layer_act[0, initial_prompt_len - 1, :] += alpha_step * diff_mean_activations[0, layer_i, :].reshape(-1)
                    else:
                        # replace activations, ### C
                        for i in range(0, step + 1):
                            alpha_step = total_alphas[i]
                            layer_act[0, initial_prompt_len - 1 + i, :] += alpha_step * diff_mean_activations[i, layer_i, :].reshape(-1)
                # getting the prediction
                edited_logits = model.lm_head.output.save()
        last_token_logits = edited_logits.value.squeeze()[-1].detach()       # dim: [vocab_size]

        nll = ce_loss(last_token_logits, gold_out[step]).detach().cpu()

        # overwrite the predicted token with the gold one to force generation
        predicted_token = gold_out[step]

        nlls.append(nll)
        prompt = torch.cat([
            prompt.squeeze(),
            predicted_token.unsqueeze(0),
        ])
    return prompt, nlls


def _top_p_filtering(logits, top_p=0.9):
    sorted_logits, sorted_indices = torch.sort(logits, descending=True)
    cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
    # Create a mask to filter out tokens with cumulative probability above top_p
    sorted_indices_to_remove = cumulative_probs > top_p
    # Shift the mask to the right to keep also the first token above top_p
    sorted_indices_to_remove[..., 1:] = sorted_indices_to_remove[..., :-1].clone()
    sorted_indices_to_remove[..., 0] = 0

    indices_to_remove = sorted_indices_to_remove.scatter(0, sorted_indices, sorted_indices_to_remove)
    indices_to_keep = ~indices_to_remove  # Boolean mask to keep only the tokens to keep
    return indices_to_keep



def generate_dynamic_edited_model(
    model: LanguageModel,
    config: dict[str, Any],
    no_icl_prompt: torch.Tensor, 
    diff_mean_activations: torch.Tensor,
    max_new_tokens: int=30,
    starting_alpha: float=2.0,
    top_p: float=0.95,
) -> tuple[torch.Tensor, list[float], list[float]]:
    
    initial_prompt_len = no_icl_prompt.shape[0]

    alpha_used = []
    real_kls = []
    for step in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                original_logits = model.lm_head.output.save()

        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]
                    # replace activations, ### C
                    for i in range(0, step + 1):
                        # steer alpha as the previous alphas
                        layer_act[0, initial_prompt_len - 1 + i, :] += starting_alpha * diff_mean_activations[i, layer_i, :].reshape(-1)
                # getting the prediction
                steer_logits = model.lm_head.output.save()

        # logits
        original_logits = original_logits.value.squeeze()[-1].detach()
        steer_logits = steer_logits.value.squeeze()[-1].detach()

        # filter with top_p
        indices_to_select_original = _top_p_filtering(original_logits, top_p)
        indices_to_select_steer = _top_p_filtering(steer_logits, top_p)
        indices_to_select_merged = indices_to_select_original | indices_to_select_steer

        # select the tokens (use the following tensors only to compute KL)
        original_logits_filtered = original_logits.clone().masked_select(indices_to_select_merged)
        steer_logits_filtered = steer_logits.clone().masked_select(indices_to_select_merged)
        
        original_softmaxed = F.log_softmax(original_logits_filtered, dim=-1).detach()           # dim: [vocab_size] target distribution
        steer_softmaxed = F.log_softmax(steer_logits_filtered, dim=-1).detach()       # dim: [vocab_size] predicted distribution

        # compute the KL to get the alpha
        kl = F.kl_div(
            steer_softmaxed,   # input
            original_softmaxed,     # target
            reduction='batchmean',
            log_target=True,
        )   # the reductions is equivalent to 'mean' in this case

        # transform the KL to alpha
        kl_item = kl.item()
        current_alpha = max(min(kl_item, 2), 0)       # clip the alpha to [0.5, 2]
        alpha_used.append(current_alpha)
        real_kls.append(kl_item)

        # steer the model with the new alpha
        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]
                    # replace activations, ### C
                    for i in range(0, step + 1):
                        # steer alpha as the previous alphas
                        layer_act[0, initial_prompt_len - 1 + i, :] += alpha_used[i] * diff_mean_activations[i, layer_i, :].reshape(-1)
                # getting the prediction
                edited_logits = model.lm_head.output.save()

        last_token_logits = edited_logits.value.squeeze()[-1].detach()
        predicted_token = last_token_logits.argmax(-1).detach()

        # stopping conditions
        if _stop_condition(predicted_token.item(), model.tokenizer):
            break

        # update prompt for the next pass
        no_icl_prompt = torch.cat([
            no_icl_prompt.squeeze(),
            predicted_token.unsqueeze(0),
        ])

    return no_icl_prompt, alpha_used, real_kls




def generate_dynamic_edited_model_composition(
    model: LanguageModel,
    config: dict[str, Any],
    no_icl_prompt: torch.Tensor, 
    t1_diff_mean_activations: torch.Tensor,
    t2_diff_mean_activations: torch.Tensor,
    max_new_tokens: int=30,
    starting_alpha: float=2.0,
    top_p: float=0.95,
) -> tuple[torch.Tensor, list[float], list[float], list[float], list[float]]:

    initial_prompt_len = no_icl_prompt.shape[0]

    alpha_t1_used = []
    alpha_t2_used = []
    t1_real_kls = []
    t2_real_kls = []

    for step in range(max_new_tokens):
        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                original_logits = model.lm_head.output.save()

        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]
                    # replace activations, ### C
                    for i in range(0, step + 1):
                        # steer alpha as the previous alphas
                        layer_act[0, initial_prompt_len - 1 + i, :] += starting_alpha * t1_diff_mean_activations[i, layer_i, :].reshape(-1)
                # getting the prediction
                t1_steer_logits = model.lm_head.output.save()

        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]
                    # replace activations, ### C
                    for i in range(0, step + 1):
                        # steer alpha as the previous alphas
                        layer_act[0, initial_prompt_len - 1 + i, :] += starting_alpha * t2_diff_mean_activations[i, layer_i, :].reshape(-1)
                # getting the prediction
                t2_steer_logits = model.lm_head.output.save()

        # logits
        original_logits = original_logits.value.squeeze()[-1].detach()
        t1_steer_logits = t1_steer_logits.value.squeeze()[-1].detach()
        t2_steer_logits = t2_steer_logits.value.squeeze()[-1].detach()

        # filter with top_p
        indices_to_select_original = _top_p_filtering(original_logits, top_p)
        indices_to_select_t1 = _top_p_filtering(t1_steer_logits, top_p)
        indices_to_select_t2 = _top_p_filtering(t2_steer_logits, top_p)
        indices_to_select_merged_t1 = indices_to_select_original | indices_to_select_t1
        indices_to_select_merged_t2 = indices_to_select_original | indices_to_select_t2

        # select the tokens (use the following tensors only to compute KL)
        t1_original_logits_filtered = original_logits.clone().masked_select(indices_to_select_merged_t1)
        t2_original_logits_filtered = original_logits.clone().masked_select(indices_to_select_merged_t2)
        t1_steer_logits_filtered = t1_steer_logits.clone().masked_select(indices_to_select_merged_t1)
        t2_steer_logits_filtered = t2_steer_logits.clone().masked_select(indices_to_select_merged_t2)

        t1_original_softmaxed = F.log_softmax(t1_original_logits_filtered, dim=-1).detach()           # dim: [vocab_size] target distribution
        t1_steer_softmaxed = F.log_softmax(t1_steer_logits_filtered, dim=-1).detach()       # dim: [vocab_size] predicted distribution
        t2_original_softmaxed = F.log_softmax(t2_original_logits_filtered, dim=-1).detach()           # dim: [vocab_size] target distribution
        t2_steer_softmaxed = F.log_softmax(t2_steer_logits_filtered, dim=-1).detach()       # dim: [vocab_size] predicted distribution

        # compute the KL to get the alpha
        kl_t1 = F.kl_div(
            t1_steer_softmaxed,   # input
            t1_original_softmaxed,     # target
            reduction='batchmean',
            log_target=True,
        )
        kl_t2 = F.kl_div(
            t2_steer_softmaxed,   # input
            t2_original_softmaxed,     # target
            reduction='batchmean',
            log_target=True,
        )
        
        # ############# debug prints
        # indent = ' ' * (step * 2)
        # print(f"{indent} no_icl_prompt: {repr(model.tokenizer.decode(no_icl_prompt.squeeze()))}")
        # print(f"{indent} Step {step}")
        # original_tokens = indices_to_select_original.nonzero(as_tuple=True)[0]
        # print(f"{indent} Original tokens (len: {len(original_tokens)}): {original_tokens[:5]}, detokenized: {[model.tokenizer.decode([x.item()]) for x in original_tokens[:5]]}")
        # t1_tokens = indices_to_select_t1.nonzero(as_tuple=True)[0]
        # print(f"{indent} T1 tokens (len: {len(t1_tokens)}): {t1_tokens[:5]}, detokenized: {[model.tokenizer.decode([x.item()]) for x in t1_tokens[:5]]}")
        # t2_tokens = indices_to_select_t2.nonzero(as_tuple=True)[0]
        # print(f"{indent} T2 tokens (len: {len(t2_tokens)}): {t2_tokens[:5]}, detokenized: {[model.tokenizer.decode([x.item()]) for x in t2_tokens[:5]]}")
        #
        # t1_tmp_softmaxed = F.softmax(t1_steer_logits, dim=-1).detach()
        # sorted_logits, sorted_indices = torch.sort(t1_tmp_softmaxed.squeeze(), descending=True)
        # for i, (a, b) in enumerate(zip(sorted_indices.squeeze(), sorted_logits.squeeze())):
        #     if i < len(t1_tokens) and i < 5:
        #         print(f'{indent} - T1 tok: {repr(model.tokenizer.decode([a.item()]))} \t prob: {b.item():.3f}')
        #     else:
        #         break
        # t2_tmp_softmaxed = F.softmax(t2_steer_logits, dim=-1).detach()
        # sorted_logits, sorted_indices = torch.sort(t2_tmp_softmaxed.squeeze(), descending=True)
        # for i, (a, b) in enumerate(zip(sorted_indices.squeeze(), sorted_logits.squeeze())):
        #     if i < len(t2_tokens) and i < 5:
        #         print(f'{indent} - T2 tok: {repr(model.tokenizer.decode([a.item()]))} \t prob: {b.item():.3f}')
        #     else:
        #         break
        # print(f"{indent} KL T1: {kl_t1.item()}")
        # print(f"{indent} KL T2: {kl_t2.item()}")

        # transform the KL to alpha
        kl_t1_item = kl_t1.item()
        kl_t2_item = kl_t2.item()
        current_alpha_t1 = max(min(kl_t1_item, 2), 0)       # clip the alpha to [0.5, 2]
        current_alpha_t2 = max(min(kl_t2_item, 2), 0)       # clip the alpha to [0.5, 2]

        alpha_t1_used.append(current_alpha_t1)
        alpha_t2_used.append(current_alpha_t2)
        t1_real_kls.append(kl_t1_item)
        t2_real_kls.append(kl_t2_item)

        # steer the model with the new alpha
        with model.trace(validate=False) as tracer:
            with tracer.invoke(no_icl_prompt, scan=False) as _:
                for layer_i in range(0, config["n_layers"]):
                    layer_act = rgetattr(
                        model, config["attn_hook_names"][layer_i]
                    ).input[0][0]
                    # replace activations, ### C
                    for i in range(0, step + 1):
                        # steer alpha as the previous alphas
                        current_step_stering = alpha_t1_used[i] * t1_diff_mean_activations[i, layer_i, :].reshape(-1) + alpha_t2_used[i] * t2_diff_mean_activations[i, layer_i, :].reshape(-1)
                        layer_act[0, initial_prompt_len - 1 + i, :] += current_step_stering
                # getting the prediction
                edited_logits = model.lm_head.output.save()

        last_token_logits = edited_logits.value.squeeze()[-1].detach()
        predicted_token = last_token_logits.argmax(-1).detach()

        # stopping conditions
        if _stop_condition(predicted_token.item(), model.tokenizer):
            break

        # update prompt for the next pass
        no_icl_prompt = torch.cat([
            no_icl_prompt.squeeze(),
            predicted_token.unsqueeze(0),
        ])

    return no_icl_prompt, alpha_t1_used, alpha_t2_used, t1_real_kls, t2_real_kls


    






