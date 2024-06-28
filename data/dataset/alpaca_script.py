from datasets import load_dataset
import pandas as pd
import json

dataset = load_dataset('yahma/alpaca-cleaned')
df = dataset['train'].to_csv('./tmp.csv')

df = pd.read_csv('./tmp.csv')

# I want only the instructions
df = df[df.input.isnull()]

# Only the instruction or output less the 150 chars
df['len_instr'] = df['instruction'].apply(lambda x: len(x))
df['len_out'] = df['output'].apply(lambda x: len(x))
df = df[df['len_instr'] < 150]
df = df[df['len_out'] < 150]


new_dataset_size = 500
counter = 0

final_dataset = []
for _, row in df.iterrows():
    counter += 1
    if counter > new_dataset_size:
        break
    final_dataset.append(
        {
            "input": row['instruction'],
            "output": row['output'],
        }
    )


# store dataset
with open('./alpaca_train.json', 'w') as f:
    json.dump(final_dataset, f, indent=4)

final_dataset[0]