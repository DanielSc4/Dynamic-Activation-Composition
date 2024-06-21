import torch
import fire
import random
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from pathlib import Path

from src.utils.model_utils import load_model_and_tokenizer, set_seed
from src.extraction import get_mean_activations
from src.utils.prompt_helper import load_dataset, tokenize_ICL
from src.delta import generate_token_step, generate_edited_model, controlled_gen_edited_model, generate_dynamic_edited_model


def main(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2", #"meta-llama/Meta-Llama-3-8B",# "stabilityai/stablelm-2-zephyr-1_6b",
    load_in_8bit: bool = True,
    dataset_A_name: str = "ITA",
    dataset_B_name: str = "ENG",
    icl_examples: int = 4,
    pre_append_instruction: bool = True,
    support: int = 30,
    max_new_tokens: int = 30,
    evaluation_size: int = 50,
    debug: bool = False,
    eval_dataset: str | None = None,
    skip_eval: bool = False,
):
    """
    Main function to compute the difference between the mean activations of two datasets
    and generate the edited model outputs.
    """
    if debug:
        evaluation_size = 20
        # model_name = "mistralai/Mistral-7B-Instruct-v0.2"
        load_in_8bit = True
        icl_examples = 0


    _name_dataset_A = dataset_A_name.split("_")[0]
    _name_dataset_B = dataset_B_name.split("_")[0]
    assert _name_dataset_A != _name_dataset_B, "The two datasets (A, B) should be different"

    path_to_output = (
        f'./output/{model_name.split("/")[1]}/{dataset_A_name.split("_")[0]}/diff'
    )
    Path(path_to_output).mkdir(parents=True, exist_ok=True)
    path_to_mean_activations_A = os.path.join(
        path_to_output, f"mean_activations_A_icl{icl_examples}_tok{max_new_tokens}_{_name_dataset_A}.pt"
    )
    path_to_mean_activations_B = os.path.join(
        path_to_output, f"mean_activations_B_icl{icl_examples}_tok{max_new_tokens}_{_name_dataset_B}.pt"
    )
    path_to_diff_mean_activations = os.path.join(
        path_to_output, f"diff_mean_act_icl{icl_examples}_tok{max_new_tokens}_{_name_dataset_A}-{_name_dataset_B}.pt"
    )
    path_to_results = os.path.join(
        path_to_output, f"results_icl{icl_examples}_tok{max_new_tokens}_{_name_dataset_A}-{_name_dataset_B}.json"
    )

    # load both datasets
    dataset_a, instruction_a, _ = load_dataset(dataset_A_name)
    dataset_a = list(map(lambda x: tuple(x.values()), dataset_a))
    print(f"[-] Loading dataset A, len: {len(dataset_a)}")

    dataset_b, instruction_b, _ = load_dataset(dataset_B_name)
    dataset_b = list(map(lambda x: tuple(x.values()), dataset_b))
    print(f"[-] Loading dataset B, len: {len(dataset_b)}")

    # load model, tokenizer and config
    model, tokenizer, config, device = load_model_and_tokenizer(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
    )
    print(f'{model_name} on {device} device')

    torch.set_grad_enabled(False)
    set_seed(32)

    # generate prompts from both datasets
    tokenized_dict_a = tokenize_ICL(
        tokenizer,
        ICL_examples=icl_examples,
        dataset=dataset_a,
        pre_append_instruction=instruction_a if pre_append_instruction else None,
    )
    icl_tokens_A = tokenized_dict_a["tokenized_prompts"]                # to compute the mean activations A (exp. behavior)
    no_icl_tokens_A = tokenized_dict_a["tokenized_prompts_no_ICL"]      # to evaluate the model (original and edited)
    gold_labels_A = tokenized_dict_a["correct_outputs"]                 # gold labels (how the model should behave)

    tokenized_dict_b = tokenize_ICL(
        tokenizer,
        ICL_examples=icl_examples,
        dataset=dataset_b,
        pre_append_instruction=instruction_b if pre_append_instruction else None,
    )
    icl_tokens_B = tokenized_dict_b["tokenized_prompts"]                # to compute the mean activations B (refused behavior)

    assert len(icl_tokens_A) == len(icl_tokens_B), "The two datasets should have the same number of examples"

    num_of_examples = min(len(icl_tokens_A), support)
    print(f"Using {num_of_examples} examples to compute the mean activations")

    
    if not os.path.exists(path_to_mean_activations_A):
        # select random prompts from the dataset 
        random_indexes = [random.randint(0, len(icl_tokens_A) - 1) for _ in range(num_of_examples)]

        print(f'[x] Computing mean activations for dataset')
        random_icl_tokens_A = [icl_tokens_A[i] for i in random_indexes]
        mean_activations_A = get_mean_activations(
            tokenized_prompts=random_icl_tokens_A,
            tokenizer=tokenizer,
            model=model,
            config=config,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        torch.save(mean_activations_A, path_to_mean_activations_A)

        random_icl_tokens_B = [icl_tokens_B[i] for i in random_indexes]
        mean_activations_B = get_mean_activations(
            tokenized_prompts=random_icl_tokens_B,
            tokenizer=tokenizer,
            model=model,
            config=config,
            device=device,
            max_new_tokens=max_new_tokens,
        )
        torch.save(mean_activations_B, path_to_mean_activations_B)
    else:
        print(f'[-] Found mean activations for dataset. Loading them')
        mean_activations_A = torch.load(path_to_mean_activations_A)
        mean_activations_B = torch.load(path_to_mean_activations_B)

    # compute the difference between the two mean activations
    diff_mean_activations = mean_activations_A - mean_activations_B
    print(f'{diff_mean_activations.shape = }')

    # store the activation difference
    torch.save(diff_mean_activations, path_to_diff_mean_activations)

    diff_mean_activations = diff_mean_activations.to(device)

    if skip_eval:
        print("[x] Skipping evaluation")
        return

    # evaluate the model
    if eval_dataset is not None:
        dataset_eval, instruction_eval, _ = load_dataset(eval_dataset)
        dataset_eval = list(map(lambda x: tuple(x.values()), dataset_eval)
        )
        print(f"[-] Loading evaluation dataset, len: {len(dataset_eval)}")

        tokenized_dict_eval = tokenize_ICL(
            tokenizer,
            ICL_examples=icl_examples,
            dataset=dataset_eval,
            pre_append_instruction=instruction_eval if pre_append_instruction else None,
        )
        # replace the current dataset with the new one
        icl_tokens_A = tokenized_dict_eval["tokenized_prompts"]
        no_icl_tokens_A = tokenized_dict_eval["tokenized_prompts_no_ICL"]
        gold_labels_A = tokenized_dict_eval["correct_outputs"]

        num_of_examples_evaluation = min(len(no_icl_tokens_A), evaluation_size)
        print(f"[-] Using {num_of_examples_evaluation} examples to evaluate the model")

        eval_idxs = list(range(num_of_examples_evaluation))     # do not randomize
    else:
        num_of_examples_evaluation = min(len(icl_tokens_B), evaluation_size)
        print(f"[-] Using {num_of_examples_evaluation} examples to evaluate the model")

        # eval_idxs = [random.randint(0, len(icl_tokens_B) - 1) for _ in range(num_of_examples_evaluation)]
        eval_idxs = list(range(num_of_examples_evaluation))     # do not randomize

    # NOTE: start for the results to be stored
    results = []
    pbar = tqdm(eval_idxs, total=num_of_examples_evaluation, desc="[x] Generating and editing model")
    for idx_prompt in pbar:
        current_prompt_noicl = no_icl_tokens_A[idx_prompt].to(device)
        current_prompt_icl = icl_tokens_A[idx_prompt].to(device)

        output_to_write = {}
        ppls_to_write = {}
        additional_to_write = {}

        # NOTE: Build the current prompt with the ICL (A) to evaluate the differences between the ICL approach and the diff approach
        pbar.set_description(f"[x] Generating and editing model [icl]")
        output_icl, nlls = generate_token_step(
            model, current_prompt_icl, max_new_tokens, return_ppl=True,
        )
        icl_only_output_ids = output_icl.squeeze()[current_prompt_icl.shape[0]:]
        decoded_icl = tokenizer.decode(
            icl_only_output_ids,
            skip_special_tokens=True
        )
        output_to_write["icl"] = decoded_icl
        perplexity_icl_baseline = torch.exp(torch.tensor(nlls).mean()).cpu().item()
        ppls_to_write["icl"] = perplexity_icl_baseline

        
        # NOTE: No ICL approach
        pbar.set_description(f"[x] Generating and editing model [no icl]")
        output_noicl = generate_token_step(
            model, current_prompt_noicl, max_new_tokens,
        )
        decoded_noicl = tokenizer.decode(
            output_noicl.squeeze()[current_prompt_noicl.shape[0]:], 
            skip_special_tokens=True,
        )
        output_to_write["no_icl"] = decoded_noicl

        additional_to_write = {}
        dy_start_alpha = 2.0
        for top_p in [0.5, 0.6, 0.7, 0.95]:
            pbar.set_description(f"[x] Generating and editing model [dynamic alpha {dy_start_alpha}]")
            output_dynamic, alpha_used, real_kls = generate_dynamic_edited_model(
                model, config, 
                no_icl_prompt=current_prompt_noicl, 
                max_new_tokens=max_new_tokens,
                diff_mean_activations=diff_mean_activations,
                starting_alpha=dy_start_alpha,
                top_p=top_p,
            )
            decoded_noicl_B_dynamic = tokenizer.decode(
                output_dynamic.squeeze()[current_prompt_noicl.shape[0]:],
                skip_special_tokens=True
            )
            output_to_write[f"dynamic_p{top_p}"] = decoded_noicl_B_dynamic

            additional_to_write[f"{top_p}_dynamic_alphas"] = alpha_used
            additional_to_write[f"{top_p}_dynamic_kls"] = real_kls

            pbar.set_description(f"[x] Generating and editing model [dynamic alpha - ppl]")
            _, nlls = controlled_gen_edited_model(
                model, config, current_prompt_noicl, max_new_tokens, 
                diff_mean_activations,
                gold_out=icl_only_output_ids,
                alpha_factor=alpha_used,
            )
            perplexity = torch.exp(torch.tensor(nlls).mean()).cpu().item()
            ppls_to_write[f"delta_dynamic_{top_p}"] = perplexity - perplexity_icl_baseline



        # NOTE: No ICL edited (different alpha factors)
        alphas: list[float] = [-1.0, 0.5, 1.0, 1.5, 2.0, 3.0, 4.0]
        # print(f'\n[NO ICL EDITED] ... (exp. gold behavior)')
        for alpha_factor in alphas:
            # NOTE: No ICL edited (plain alpha)
            pbar.set_description(f"[x] Generating and editing model [edited_{alpha_factor}]")
            output_noicl_B_edited = generate_edited_model(
                model, config, current_prompt_noicl, max_new_tokens, 
                diff_mean_activations,
                alpha_factor=alpha_factor,
            )
            decoded_noicl_B_edited = tokenizer.decode(
                output_noicl_B_edited.squeeze()[current_prompt_noicl.shape[0]:],
                skip_special_tokens=True
            )
            output_to_write[f"edited_{alpha_factor}"] = decoded_noicl_B_edited

            pbar.set_description(f"[x] Generating and editing model [edited_{alpha_factor} - ppl]")
            _, nlls = controlled_gen_edited_model(
                model, config, current_prompt_noicl, max_new_tokens, 
                diff_mean_activations,
                gold_out=icl_only_output_ids,
                alpha_factor=alpha_factor,
            )
            perplexity = torch.exp(torch.tensor(nlls).mean()).cpu().item()
            ppls_to_write[f"delta_edited_{alpha_factor}"] = perplexity - perplexity_icl_baseline

            
            # NOTE: No ICL edited (diminishing and start alpha)
            if alpha_factor > 0.5:
                # NOTE: start alpha
                pbar.set_description(f"[x] Generating and editing model [editedSTART_{alpha_factor}]")
                output_noicl_B_edited = generate_edited_model(
                    model, config, current_prompt_noicl, max_new_tokens, 
                    diff_mean_activations,
                    alpha_factor=alpha_factor,
                    only_start_token=True,
                )
                decoded_noicl_B_edited = tokenizer.decode(
                    output_noicl_B_edited.squeeze()[current_prompt_noicl.shape[0]:],
                    skip_special_tokens=True
                )
                output_to_write[f"editedSTART_{alpha_factor}"] = decoded_noicl_B_edited
                # compute perplexity on ICL output
                pbar.set_description(f"[x] Generating and editing model [editedSTART_{alpha_factor} - ppl]")
                _, nlls = controlled_gen_edited_model(
                    model, config, current_prompt_noicl, max_new_tokens, 
                    diff_mean_activations,
                    gold_out=icl_only_output_ids,
                    alpha_factor=alpha_factor,
                    only_start_token=True,
                )
                perplexity = torch.exp(torch.tensor(nlls).mean()).cpu().item()
                ppls_to_write[f"delta_editedSTART_{alpha_factor}"] = perplexity - perplexity_icl_baseline


                # NOTE: diminishing alpha
                pbar.set_description(f"[x] Generating and editing model [editedDIM_{alpha_factor}]")
                output_noicl_B_edited = generate_edited_model(
                    model, config, current_prompt_noicl, max_new_tokens, 
                    diff_mean_activations,
                    alpha_factor=alpha_factor,
                    diminishing_alpha=True,
                )
                decoded_noicl_B_edited = tokenizer.decode(
                    output_noicl_B_edited.squeeze()[current_prompt_noicl.shape[0]:],
                    skip_special_tokens=True
                )
                output_to_write[f"editedDIM_{alpha_factor}"] = decoded_noicl_B_edited
                pbar.set_description(f"[x] Generating and editing model [editedDIM_{alpha_factor} - ppl]")
                # compute perplexity on ICL output
                _, nlls = controlled_gen_edited_model(
                    model, config, current_prompt_noicl, max_new_tokens, 
                    diff_mean_activations,
                    gold_out=icl_only_output_ids,
                    alpha_factor=alpha_factor,
                    diminishing_alpha=True,
                )
                perplexity = torch.exp(torch.tensor(nlls).mean()).cpu().item()
                ppls_to_write[f"delta_editedDIM_{alpha_factor}"] = perplexity - perplexity_icl_baseline



        res_struct = {
            "prompt": tokenizer.decode(no_icl_tokens_A[idx_prompt], skip_special_tokens=True),
            "gold": gold_labels_A[idx_prompt],
            "output": output_to_write,
            "perplexity": ppls_to_write,
            "additional": additional_to_write,
        }
        results.append(res_struct)


    # save the results
    with open(path_to_results, "w+", encoding="utf-8") as f:
        json.dump(results, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    fire.Fire(main)

