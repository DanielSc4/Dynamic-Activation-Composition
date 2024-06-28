import json
import fire
import os

from pathlib import Path
from tqdm import tqdm
from transformers import pipeline

def main(
    original_dataset_name = 'ENi2f',
    new_dataset_name = 'ITi2f',
    src_lang = 'eng_Latn',
    tgt_lang = 'ita_Latn',
    device = 'cuda',
):
    path_to_original = f'./data/datasets/{original_dataset_name}'
    path_to_new = f'./data/datasets/{new_dataset_name}'

    # check if the new dataset folder exists or make it
    Path(path_to_new).mkdir(parents=True, exist_ok=True)

    # load json file
    for tp in ['train']:        #['all', 'eval', 'train']:
        with open(os.path.join(path_to_original, f'{original_dataset_name}_{tp}.json'), 'r') as f:
            data = json.load(f)

        translator = pipeline(
            "translation", 
            model="facebook/nllb-200-distilled-1.3B",
            src_lang=src_lang, tgt_lang=tgt_lang,
            device=device,
        )

        for dd in tqdm(data, total=len(data), desc=f'[{tp}] Translating'):
            dd["output"] = translator(dd["output"])[0]["translation_text"]


        with open(os.path.join(path_to_new, f'{new_dataset_name}_{tp}.json'), 'w') as f:
            json.dump(data, f, indent=4, ensure_ascii=False)


if __name__ == "__main__":
    fire.Fire(main)

