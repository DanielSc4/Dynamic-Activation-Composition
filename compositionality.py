import fire
import torch
import json
import os
os.environ["TOKENIZERS_PARALLELISM"] = "false"

from tqdm import tqdm
from pathlib import Path

from src.utils.model_utils import load_model_and_tokenizer 
from src.utils.prompt_helper import tokenize_ICL, load_dataset
from src.delta import generate_token_step, generate_edited_model, controlled_gen_edited_model, generate_dynamic_edited_model_composition



def main(
    model_name: str = "mistralai/Mistral-7B-Instruct-v0.2",
    load_in_8bit: bool = True,
    source_dataset: str = "ENunsafe_train",           # the actual dataset that will be used to evaluate
    lang_dataset: str = "ITA_train",                # only for the ICL (correspond to the task2)
    source_lang_dataset: str = "ITunsafe_train",
    diff_path1: str = 'output/Mistral-7B-Instruct-v0.2/ENunsafe/diff/diff_mean_act_icl4_tok30_ENunsafe-ENsafe.pt',
    diff_path2: str = 'output/Mistral-7B-Instruct-v0.2/ITA/diff/diff_mean_act_icl4_tok30_ITA-ENG.pt',
    ICL_examples: int = 4,
    subset_eval: int = 50,
    debug: bool = False,
    ):
    if debug:
        subset_eval = 3
        ICL_examples = 2
        model_name = "stabilityai/stablelm-2-zephyr-1_6b"
        diff_path1 = f'output/{model_name.split("/")[-1]}/ENunsafe/diff/diff_mean_act_icl4_tok30_ENunsafe-ENsafe.pt'
        diff_path2 = f'output/{model_name.split("/")[-1]}/ITA/diff/diff_mean_act_icl4_tok30_ITA-ENG.pt'
        load_in_8bit = True


    task1 = source_dataset.split("_")[0]
    assert task1 == diff_path1.split("/")[2], f"The diff_path1 should be calculated over the source_dataset. {task1} != {diff_path1.split('/')[2]}"

    task2 = lang_dataset.split("_")[0]
    task3 = source_lang_dataset.split("_")[0]


    model, tokenizer, config, device = load_model_and_tokenizer(
        model_name=model_name,
        load_in_8bit=load_in_8bit,
    )

    path_to_output = (
        f'./output/{model_name.split("/")[1]}/compositionality/{source_dataset.split("_")[0]}/{task1}-{task2}'
    )
    Path(path_to_output).mkdir(parents=True, exist_ok=True)

    diff1 = torch.load(diff_path1)
    diff2 = torch.load(diff_path2)
    max_new_tokens = min(diff1.shape[0], diff2.shape[0])
    print(f'Using max_new_tokens {max_new_tokens}')
    print(diff1.shape, diff2.shape)

    dataset, instruction, _ = load_dataset(source_dataset)
    dataset = list(map(lambda x: tuple(x.values()), dataset))
    print(f"[-] Loading dataset, len: {len(dataset)}")

    dataset_task2, instruction_task2, _ = load_dataset(lang_dataset)
    dataset_task2 = list(map(lambda x: tuple(x.values()), dataset_task2))
    print(f"[-] Loading dataset task2, len: {len(dataset_task2)}")

    dataset_task3, instruction_task3, _ = load_dataset(source_lang_dataset)
    dataset_task3 = list(map(lambda x: tuple(x.values()), dataset_task3))
    print(f"[-] Loading dataset task3, len: {len(dataset_task3)}")

    # prompt = tokenizer(prompt_str, return_tensors="pt").input_ids.to(device)
    token_out = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=ICL_examples,
        dataset=dataset,
        pre_append_instruction=instruction,
    )
    icl_prompts = token_out['tokenized_prompts']
    noicl_prompts = token_out['tokenized_prompts_no_ICL']
    correct_outputs = token_out['correct_outputs']

    # get only the icl from the task2 dataset (just to evaluate the icl performance)
    token_out = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=ICL_examples,
        dataset=dataset_task2,
        pre_append_instruction=instruction_task2,
    )
    icl_prompts_task2 = token_out['tokenized_prompts']

    # get only the icl from the source_lang_dataset (just to evaluate the icl performance)
    token_out = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=ICL_examples,
        dataset=dataset_task3,
        pre_append_instruction=instruction_task3,
    )
    icl_prompts_task3 = token_out['tokenized_prompts']

    num_of_examples = min([
        subset_eval,
        len(icl_prompts),
        len(icl_prompts_task2),
        len(icl_prompts_task3),
    ])
    print(f"Using {num_of_examples} examples to compute the mean activations")
    icl_prompts = icl_prompts[:num_of_examples]
    noicl_prompts = noicl_prompts[:num_of_examples]
    correct_outputs = correct_outputs[:num_of_examples]
    icl_prompts_task2 = icl_prompts_task2[:num_of_examples]
    icl_prompts_task3 = icl_prompts_task3[:num_of_examples]

    # loop over the dataset to generate all combinations
    all_results = []
    pbar = tqdm(
        zip(noicl_prompts, icl_prompts_task3, correct_outputs), 
        total=subset_eval, 
        desc="[x] Generating outputs",
    )
    for noicl_prompt, icl_prompt_task3, correct_output in pbar:
        # prompt_icl = prompt_icl.to(device)
        # prompt_icl_task2 = prompt_icl_task2.to(device)
        # prompt = prompt.to(device)
        noicl_prompt = noicl_prompt.to(device)
        icl_prompt_task3 = icl_prompt_task3.to(device)

        struct_output = {
            "prompt": tokenizer.decode(noicl_prompt, skip_special_tokens=True),
            "gold": correct_output,
            "output": {},
            "perplexity": {},
        }
        output_results = {}
        perplexity_results = {}

        # whats really needed here
        # steer to italian on the icl_unsafe and get only the output (it is meant to measure the toxicity of the italian steering)
        # steer to italian AND to unsafe (task3), get the ppl and output (it is meant to measure both the toxicity and the fluency of the italian steering)
        #    here the ppl is computed wrt ICL comp. which is the ICL composition output. You should compute it too (source_lang_dataset)

        # actual baseline (clean prompt (noicl) with clean model)
        # NOTE: baseline with noicl prompt
        pbar.set_description(f"[x] Generating outputs - (no_icl) baseline")
        # clean generation (noicl - baseline)
        out = generate_token_step(
            model=model,
            prompt=noicl_prompt,
            max_new_tokens=max_new_tokens,
        )
        decoded = tokenizer.decode(out[noicl_prompt.shape[0]:])
        output_results[f"noicl_{task1}"] = decoded


        # only needed as reference generation for the ppl when combining the steering task1 and task2
        # NOTE: baseline with icl prompt (ITunsafe)
        pbar.set_description(f"[x] Generating outputs - (icl composition) icl {task3}")
        out, nlls = generate_token_step(
            model=model,
            prompt=icl_prompt_task3,
            max_new_tokens=max_new_tokens,
            return_ppl=True,
        )
        reference_tokens_task3 = out[icl_prompt_task3.shape[0]:].detach()
        decoded = tokenizer.decode(reference_tokens_task3)
        output_results[f"icl_{task3}"] = decoded
        ppl_reference_baseline = torch.exp(torch.stack(nlls).mean()).item()
        perplexity_results[f"icl_{task3}"] = ppl_reference_baseline

        
        # steering to italian on the icl_unsafe and get only the output
        # this is done to check how much the steering influence the performances on the noicl_prompt
        # e.g. task1 = ENunsafe, task2 = ITA, how much the steering to ITA makes the output unsafe?
        # ppl here is not needed since there is no icl output to compare with
        pbar.set_description(f"[x] Generating outputs - (no_icl) steering {task2}")
        out = generate_edited_model(
            model=model,
            config=config,
            prompt=noicl_prompt,
            max_new_tokens=max_new_tokens,
            diff_mean_activations=diff2,
        )
        decoded = tokenizer.decode(out[noicl_prompt.shape[0]:])
        output_results[f"noicl_{task2}"] = decoded


        additional_to_write = {}
        dy_start_alpha = 2.0
        for top_p in [0.4, 0.5, 0.6, 0.7, 0.9]:
            pbar.set_description(f"[x] Generating and editing model [dynamic alpha {top_p}]")
            out, alpha_t1_used, alpha_t2_used, t1_real_kls, t2_real_kls = generate_dynamic_edited_model_composition(
                model=model,
                config=config,
                no_icl_prompt=noicl_prompt,
                t1_diff_mean_activations=diff1,
                t2_diff_mean_activations=diff2,
                max_new_tokens=max_new_tokens,
                starting_alpha=dy_start_alpha,
                top_p=top_p,
            )
            decoded = tokenizer.decode(
                out.squeeze()[noicl_prompt.shape[0]:],
                skip_special_tokens=True,
            )
            output_results[f"dynamic_p{top_p}_{task1}-{task2}"] = decoded

            additional_to_write[f"{top_p}alpha_t1"] = alpha_t1_used
            additional_to_write[f"{top_p}alpha_t2"] = alpha_t2_used
            additional_to_write[f"{top_p}t1_real_kls"] = t1_real_kls
            additional_to_write[f"{top_p}t2_real_kls"] = t2_real_kls

            pbar.set_description(f"[x] Generating and editing model [dynamic alpha - ppl]")
            # composite the tasks manually based on the alpha_t1_used and alpha_t2_used
            current_dynamic_diff = 1 * diff1[:max_new_tokens] + 1 * diff2[:max_new_tokens]
            for i, (alpha_1, alpha_2) in enumerate(zip(alpha_t1_used, alpha_t2_used)):
                current_dynamic_diff[i] = alpha_1 * diff1[i] + alpha_2 * diff2[i]

            _, nlls = controlled_gen_edited_model(
                model=model,
                config=config,
                prompt=noicl_prompt,
                max_new_tokens=max_new_tokens,
                diff_mean_activations=current_dynamic_diff,
                gold_out=reference_tokens_task3,
                alpha_factor=1.0,       # keep this constant since the alpha is already used in the dynamic steering current_dynamic_diff
            )
            perplexity = torch.exp(torch.stack(nlls).mean()).item()
            perplexity_results[f"dynamic_p{top_p}_{task1}-{task2}"] = perplexity - ppl_reference_baseline


        alphas = [1.0, 1.5, 2.0]
        for alpha_1 in alphas:
            for alpha_2 in alphas:
                # compute the current composition
                comp_vector = alpha_1 * diff1[:max_new_tokens] + alpha_2 * diff2[:max_new_tokens]

                pbar.set_description(f"[x] Generating and editing [{alpha_1} x {task1} - {alpha_2} x {task2}]")
                out = generate_edited_model(
                    model=model,
                    config=config,
                    prompt=noicl_prompt,
                    max_new_tokens=max_new_tokens,
                    diff_mean_activations=comp_vector,
                )
                decoded = tokenizer.decode(out[noicl_prompt.shape[0]:])
                # store the results
                output_results[f"{alpha_1}_{task1}-{alpha_2}_{task2}"] = decoded

                pbar.set_description(f"[x] Generating and editing [{alpha_1} x {task1} - {alpha_2} x {task2} - ppl]")
                # compute perplexity here over the ICL task3 output
                _, nlls = controlled_gen_edited_model(
                    model=model,
                    config=config,
                    prompt=noicl_prompt,
                    max_new_tokens=max_new_tokens,
                    diff_mean_activations=comp_vector,
                    gold_out=reference_tokens_task3,
                    alpha_factor=1.0,
                )
                perplexity = torch.exp(torch.stack(nlls).mean()).item()
                perplexity_results[f"{alpha_1}_{task1}-{alpha_2}_{task2}"] = perplexity - ppl_reference_baseline

                if alpha_1 > 0.5 and alpha_2 > 0.5:
                    # NOTE: starting alpha
                    pbar.set_description(f"[x] Generating and editing [Start {alpha_1} x {task1} - {alpha_2} x {task2}]")
                    out = generate_edited_model(
                        model=model,
                        config=config,
                        prompt=noicl_prompt,
                        max_new_tokens=max_new_tokens,
                        diff_mean_activations=comp_vector,
                        only_start_token=True,
                    )
                    decoded = tokenizer.decode(out[noicl_prompt.shape[0]:])
                    # store the results
                    output_results[f"START{alpha_1}_{task1}-{alpha_2}_{task2}"] = decoded

                    pbar.set_description(f"[x] Generating and editing [Start {alpha_1} x {task1} - {alpha_2} x {task2} - ppl]")
                    # compute perplexity here over the ICL task3 output
                    _, nlls = controlled_gen_edited_model(
                        model=model,
                        config=config,
                        prompt=noicl_prompt,
                        max_new_tokens=max_new_tokens,
                        diff_mean_activations=comp_vector,
                        gold_out=reference_tokens_task3,
                        alpha_factor=1.0,
                        only_start_token=True,
                    )
                    perplexity = torch.exp(torch.stack(nlls).mean()).item()
                    perplexity_results[f"START{alpha_1}_{task1}-{alpha_2}_{task2}"] = perplexity - ppl_reference_baseline


                    # NOTE: diminishing steering
                    pbar.set_description(f"[x] Generating and editing [diminishing {alpha_1} x {task1} - {alpha_2} x {task2}]")
                    out = generate_edited_model(
                        model=model,
                        config=config,
                        prompt=noicl_prompt,
                        max_new_tokens=max_new_tokens,
                        diff_mean_activations=comp_vector,
                        diminishing_alpha=True,
                    )
                    decoded = tokenizer.decode(out[noicl_prompt.shape[0]:])
                    # store the results
                    output_results[f"DIM{alpha_1}_{task1}-{alpha_2}_{task2}"] = decoded

                    pbar.set_description(f"[x] Generating and editing [diminishing {alpha_1} x {task1} - {alpha_2} x {task2} - ppl]")
                    # compute perplexity here over the ICL task3 output
                    _, nlls = controlled_gen_edited_model(
                        model=model,
                        config=config,
                        prompt=noicl_prompt,
                        max_new_tokens=max_new_tokens,
                        diff_mean_activations=comp_vector,
                        gold_out=reference_tokens_task3,
                        alpha_factor=1.0,
                        diminishing_alpha=True,
                    )
                    perplexity = torch.exp(torch.stack(nlls).mean()).item()
                    perplexity_results[f"DIM{alpha_1}_{task1}-{alpha_2}_{task2}"] = perplexity - ppl_reference_baseline

        # save the results into the struct
        struct_output["output"] = output_results
        struct_output["perplexity"] = perplexity_results
        struct_output["additional"] = additional_to_write

        # append the results
        all_results.append(struct_output)


    # save the results to disk
    path_to_results = os.path.join(path_to_output, f"results_{task1}_{task2}_tok{max_new_tokens}.json")
    with open(path_to_results, "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)



if __name__ == "__main__":
    fire.Fire(main)


