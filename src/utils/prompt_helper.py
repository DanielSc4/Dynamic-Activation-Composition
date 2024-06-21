import json
import os
from warnings import warn

import torch
from transformers import PreTrainedTokenizer

# maps the first half of the dataset name (before the '_') to the task name (corresponding ICL template for evaluation)
_autotask = {
    # "ITtruthful": "truthfulness",
    "truthful": "truthfulness",
    "untruthful": "untruthfulness",
    "cona-facts": "hate",   # whatever, will not be used for evaluation anyway (using LLamaGuard for hate)
    "ITA": "ITA",
    "ENG": "ENG",
    "ENunsafe": "ENunsafe",
    "ITunsafe": "ITunsafe",
}

def load_dataset(dataset_name: str) -> tuple[list[dict[str, str]], str, str]:
    """
    Load the dataset and instruction from the dataset_name
    Args:
        dataset_name (str): full dataset name without extension (.json)
    Returns:
        tuple[list[dict[str, str]], str]: 
            - dataset: `[{"input": prompt, "output": expected generation}, ...]`
            - instruction: str
            - task: str
    """
    if dataset_name.split("_")[0] not in _autotask.keys():
        # give a warning
        warn(f"No task found for dataset {dataset_name}")
        task = None
    else:
        task = _autotask[
            dataset_name.split("_")[0]
        ] 

    clean_dataset_name = dataset_name.split('_')[0]     # exlude the _eval / _train part
    data_path = os.path.join("./data/datasets/", clean_dataset_name)

    dataset_json_path = os.path.join(data_path, f"{dataset_name}.json")
    dataset = _load_json_dataset(dataset_json_path)

    instruction_path = os.path.join(data_path, f"{clean_dataset_name}_instr.txt")
    # read from file
    with open(instruction_path, 'r') as file:
        instruction = file.read()

    return dataset, instruction, task


def _load_json_dataset(json_path):
    with open(json_path, encoding="utf-8") as file:
        dataset = json.load(file)
    return dataset


def build_prompt_txt(
    queries: list[str],
    answers: list[str],
    pre_append_instruction: str | None = None,
):
    """Build the prompt following the default template. Provide a list of queries (length = n ICL examples)
    and a list of answers (length = n ICL examples)

    Args:
        queries (list[str]): queries (ICL examples + final query).
        answers (list[str]): answers (ICL examples), must be one less than queries.
        pre_append_instruction (str | None): Optional instruction at the beginning of each prompt. Defaults to None.

    Returns:
        full_prompt: full prompt following the default template with notation
    """
    assert (
        len(queries) == len(answers) + 1
    ), f"queries (len={len(queries)}) should have one more element than answers (len={len(answers)})"

    full_prompt = []
    if pre_append_instruction:
        full_prompt.append((pre_append_instruction, "sentence"))
        full_prompt.append(("\n", "structural"))

    # prompt default template for structural parts
    begin = [
        ("Q:", "structural"),
    ]
    middle = [
        ("\nA:", "structural"),
    ]
    end = [
        ("\n\n", "structural"),
    ]

    for i in range(len(answers) - 1):
        full_prompt.extend(begin)
        full_prompt.append((queries[i], "sentence"))
        full_prompt.extend(middle)
        full_prompt.append((answers[i], "sentence"))
        full_prompt.extend(end)
    # append the final query without the answer
    full_prompt.extend(begin)
    full_prompt.append((queries[-1], "sentence"))
    full_prompt.extend(middle)

    return full_prompt


def tokenize_from_template(tokenizer, promtp_w_template: tuple[tuple[str, str]]):
    """tokenize the prompt following the provided template and return a list of indexes referring to the structural toknes
    and only the last token of sentences (automatically include the bos token at the beginning)

    Args:
        tokenizer (*Tokenizer): huggingface tokenizer
        promtp_w_template (tuple[tuple[str, str]], optional): prompt_template in the form of:
        ```python
            TODO
        ```. Defaults to None.

    Returns:
        torch.LongTensor: tokenized input ids
        list: index of every structural token and last sentence token
    """

    full_tokenized = torch.LongTensor([[tokenizer.bos_token_id]])   # manually adding BOS token
    indexes = [0]

    for prompt_str, prompt_type in promtp_w_template:
        tokenized = tokenizer(
            prompt_str, 
            return_tensors="pt", 
            # add_special_tokens=False
        ).input_ids.type(
            torch.int64
        )
        full_tokenized = torch.cat(
            (full_tokenized, tokenized),
            dim=-1,
        )

        if prompt_type == "structural":
            # all the index of structural tokens must be included
            actual_idxs = list(
                range(indexes[-1] + 1, tokenized.shape[-1] + indexes[-1] + 1)
            )
            indexes.extend(actual_idxs)
        elif prompt_type == "sentence":
            # include only the last index of the sentence tokens
            indexes.append(indexes[-1] + tokenized.shape[-1])

    full_tokenized = full_tokenized.squeeze()
    return full_tokenized, indexes


def tokenize_ICL(
    tokenizer: PreTrainedTokenizer,
    ICL_examples: int,
    dataset: list[tuple[str, str]],
    pre_append_instruction: str | None = None,
):
    """build ICL prompt from the dataset, tokenize it and return the tokenized prompt with the important ids.

    Args:
        tokenizer (PreTrainedTokenizer): tokenizer from HuggingFace
        ICL_examples (int): number of ICL examples (excluding the last one without the solution)
        dataset (list[tuple[str, str]]): list of tuples (query, answer).
        pre_append_instruction (str | None): Optional instruction at the beginning of each prompt. Defaults to None.

    Returns:
        ```python
        {
            "tokenized_prompts": list[torch.LongTensor],            # tokenized prompts with ICL
            "important_ids": list[list[int]],                       # important ids for each prompt
            "only_ICL": list[torch.LongTensor],                     # tokenized ICL examples (without the last query)
            "tokenized_prompts_no_ICL": list[torch.LongTensor],     # tokenized prompts without ICL (same as tokenized_prompts if ICL_examples == 0)
            "important_ids_no_ICL": list[list[int]],                # important ids for each prompt without ICL (same as important_ids if ICL_examples == 0)
            "correct_outputs": list[str],                           # correct labels for each prompt
        }
        ```
    """

    if len(dataset) <= ICL_examples:
        raise ValueError(
            f"dataset dimension ({len(dataset)}) is <= ICL_examples ({ICL_examples})"
        )

    prompts = []
    prompts_no_ICL = []
    only_ICL = []
    labels = []

    for i in range(0, len(dataset), ICL_examples + 1):
        # select examples to end up in the prompt
        group = dataset[i : i + ICL_examples + 1]

        # if enough ICL examples in the split (or group), otherwise don't use them
        if len(group) > ICL_examples:
            queries, answers = zip(*group)

            # build ICL prompt with pre_append_instruction + queries + answers + ... + last_query
            X_ICL = build_prompt_txt(
                queries=queries,
                answers=answers[:-1],
                pre_append_instruction=pre_append_instruction,
            )
            # store ICL prompt and label (last answer)
            prompts.append(X_ICL)
            only_ICL.append(X_ICL[:-3]) # remove the last query "Q: ... A:" and store only the ICL examples

            # build prompt without ICL examples and without pre_append_instruction (last_query only)
            last_query = queries[-1]
            X_no_ICL = build_prompt_txt(queries=[last_query], answers=[])
            prompts_no_ICL.append(X_no_ICL)

            prompt_label = answers[-1]
            labels.append(prompt_label)

    # tokenize every prompt and store the important ids
    all_tokenized, all_ids = [], []
    all_tokenized_no_ICL, all_ids_no_ICL = [], []
    all_ICL = []
    for prompt_template, prompts_no_ICL_template, only_ICL in zip(prompts, prompts_no_ICL, only_ICL):
        tokenized_prompt, important_ids = tokenize_from_template(
            tokenizer=tokenizer, promtp_w_template=prompt_template
        )
        all_tokenized.append(tokenized_prompt)
        all_ids.append(important_ids)

        tokenized_prompts_no_ICL, important_ids_no_ICL = tokenize_from_template(
            tokenizer=tokenizer, promtp_w_template=prompts_no_ICL_template
        )
        all_tokenized_no_ICL.append(tokenized_prompts_no_ICL)
        all_ids_no_ICL.append(important_ids_no_ICL)

        tokenized_ICL, _ = tokenize_from_template(
            tokenizer=tokenizer, promtp_w_template=only_ICL
        )
        all_ICL.append(tokenized_ICL)


    return {
        "tokenized_prompts": all_tokenized,
        "important_ids": all_ids,
        "only_ICL": all_ICL,
        "tokenized_prompts_no_ICL": all_tokenized_no_ICL,
        "important_ids_no_ICL": all_ids_no_ICL,
        "correct_outputs": labels,
    }


def tokenize_w_template(
    tokenizer: PreTrainedTokenizer,
    prompt: str,
    ) -> torch.LongTensor:
    """
    Tokenize a simple prompt and return the tokenized prompt.
    """

    tokenized_prompt = tokenize_ICL(
        tokenizer=tokenizer,
        ICL_examples=0,
        dataset=[(prompt, "")],
    )['tokenized_prompts_no_ICL'][0]

    return tokenized_prompt
    
    



def pad_input_and_ids(
    tokenized_prompts,
    important_ids: list[list[int]],
    max_len=256,
    pad_token_id: int | None = 50256,
):
    """pad a batched input

    Args:
        tokenized_prompts (list[torch.Tensor]): list of tokenized sentences
        important_ids (list[int]). Important ids to be shifted according to the pad length
        max_len (int, optional): max len to pad. Defaults to 256.
        pad_token_id (int, optional): as name. Defaults to tokenizer.eos_token_id.

    Returns:
        dict[torch.Tensor, torch.Tensor]: dict with padded sentences and corresponding attention mask
    """
    padded_prompts = []
    attention_masks = []
    adapted_ids = []

    # Process each tokenized prompt individually
    for tokenized_prompt, imp_ids in zip(tokenized_prompts, important_ids):
        padded_prompt = torch.nn.functional.pad(
            tokenized_prompt,
            pad=(max_len - len(tokenized_prompt), 0),
            value=pad_token_id,
        )
        padded_prompts.append(padded_prompt)

        attention_mask = torch.zeros(max_len, dtype=torch.long)
        attention_mask[-len(tokenized_prompt) :] = 1
        attention_masks.append(attention_mask)

        adapted_ids.append(
            [ele + torch.sum(attention_mask == 0).item() for ele in imp_ids]
        )

    padded_prompts_tensor = torch.vstack(padded_prompts)
    attention_masks_tensor = torch.vstack(attention_masks)

    return {
        "input_ids": padded_prompts_tensor,
        "attention_mask": attention_masks_tensor,
    }, adapted_ids


def find_missing_ranges(lst: list[int]):
    """
    Given a list of important_ids (e.g. [0, 3, 4, 5, 6, 8]) find the missing ranges
    to average on ([1, 3], [7, 8])
    """
    ranges = [(start, end) for start, end in zip(lst, lst[1:]) if start + 1 < end]
    return [[start + 1, end] for start, end in ranges]
