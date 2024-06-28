import json
import os
from datasets import load_dataset
from pathlib import Path



def select_subset(dataset, subset_size, safe:bool):
    count = 0
    subset = []
    for idx_ele in range(len(dataset)):
        if dataset[idx_ele]['is_safe'] == safe:
            count += 1
            if count > subset_size:
                break

            current_row = dataset[idx_ele]
            struct = {
                "input": current_row['prompt'],
                "output": current_row['response']
            }
            subset.append(struct)
        if count > subset_size:
            break

    return subset




# Load the whole dataset
dataset = load_dataset('PKU-Alignment/BeaverTails')
split = '330k_train'
# shuffle the dataset
dataset[split] = dataset[split].shuffle(seed=42)
# select only safe prompts
subset_size = 500



# safe_dataset = select_subset(dataset[split], subset_size, safe=True)
# unsafe_dataset = select_subset(dataset[split], subset_size, safe=False)
#
# # write the safe and unsafe datasets to a json file
# path_to_safe_dataset = './data/datasets/safe-BeaverTails/safe-BeaverTails_train.json'
# path_to_unsafe_dataset = './data/datasets/unsafe-BeaverTails/unsafe-BeaverTails_train.json'
#
#
# with open(path_to_safe_dataset, 'w') as f:
#     json.dump(safe_dataset, f, indent=4, ensure_ascii=False)
# with open(path_to_unsafe_dataset, 'w') as f:
#     json.dump(unsafe_dataset, f, indent=4, ensure_ascii=False)
#



def select_subset_subcategory(dataset, subset_size, category, safe:bool):
    count = 0
    subset = []
    for idx_ele in range(len(dataset)):
        if not dataset[idx_ele]['is_safe']:
            if dataset[idx_ele]['category'][category]:
                count += 1
                if count > subset_size:
                    break

                current_row = dataset[idx_ele]
                struct = {
                    "input": current_row['prompt'],
                    "output": current_row['response']
                }
                subset.append(struct)
        if count > subset_size:
            break

    return subset




categories = ['privacy_violation', 'hate_speech,offensive_language', 'discrimination,stereotype,injustice', 'violence,aiding_and_abetting,incitement']
for category in categories:
    subset_size = 500

    unsafe_dataset = select_subset_subcategory(dataset[split], subset_size, category, safe=False)
    print(f'{category} unsafe dataset size: {len(unsafe_dataset)}')
    name_dataset = f'unsafe-BeaverTails-{category.split(",")[0].split("_")[0]}'
    path = f'./data/datasets/{name_dataset}'
    Path(path).mkdir(parents=True, exist_ok=True)  # check if path exists
    # check if path exists
    path_to_unsafe_dataset = os.path.join(path, f'{name_dataset}_train.json')
    with open(path_to_unsafe_dataset, 'w') as f:
        json.dump(unsafe_dataset, f, indent=4, ensure_ascii=False)


