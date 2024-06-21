import fire
import json
from src.utils.eval.multi_token_evaluator import Evaluator
from tqdm import tqdm

from langdetect import detect, detect_langs
from src.utils.prompt_helper import load_dataset
from transformers import pipeline



def eval_lang(all_results: list[dict], lang_to_eval:str):

    for res in tqdm(all_results, desc="[1/3] Evaluating results [lang]", total=len(all_results)):
        outputs_eval = {}   # should contain icl, noicl, edited_alpha_*, ...
        for k, v in res["output"].items():
            # keeping the original output
            eval_struct = {}
            try:
                label = detect(v)
                eval_struct["lang"] = label
                # non so chi si sia inventato questa struttura dati
                # ogni elemento in detect_langs è un NamedTuple con due attributi: lang e prob
                detected_languages = detect_langs(v)
                all_languages = [ele.lang for ele in detected_languages]
                all_scores = [ele.prob for ele in detected_languages]
                if lang_to_eval in all_languages:
                    eval_struct[lang_to_eval] = all_scores[all_languages.index(lang_to_eval)]
                else:
                    eval_struct[lang_to_eval] = 0.0
                
            except:
                print(v)
                print('The above text could not be detected')
                eval_struct["lang"] = 'ERR'
                eval_struct[lang_to_eval] = 0.0
            outputs_eval[k] = eval_struct

        # adding the evaluation outputs
        res["eval_lang"] = outputs_eval

    return all_results


def out_to_eng(all_results: list[dict], source_lang: str):
    translator = pipeline(
        "translation", 
        model="facebook/nllb-200-distilled-1.3B",
        src_lang="eng_Latn", tgt_lang="ita_Latn",
        device="cuda",
    )

    results_new_key = 'output_eng'
    for res in tqdm(all_results, desc="[2/3] Translating outputs", total=len(all_results)):
        res[results_new_key] = {}
        for k, v in res['output'].items():
            # detect language of v
            try:
                lang = detect(v)
            except:
                print(f"Error detecting language of {v}")
                res[results_new_key][k] = v     # keeping the original output (not translated)
                lang = "ERR"
    
            # if not english, translate to english
            if lang != 'en':
                res['output_eng'][k] = translator(
                    v,
                    src_lang=source_lang,
                    tgt_lang='eng_Latn'
                )[0]["translation_text"]
            else:
                res[results_new_key][k] = v
    return all_results

def evaluate_task(all_results: list[dict], evaluator: Evaluator, output_cell: str):
    for res in tqdm(all_results, desc="[x] Evaluating results", total=len(all_results)):
        
        outputs_eval = {}   # should contain icl, noicl, edited_alpha_*, ...
        for k, v in res[output_cell].items():
            # keeping the original output
            if evaluator.model_info["type"] == "classification_formality":
                eval_results = evaluator.get_evaluation(
                    generations=[v],
                )
                outputs_eval[k] = {ele: eval_results[ele][0] for ele in eval_results}
            else:
                eval_results = evaluator.get_evaluation(
                    prompts=[res["prompt"]],
                    generations=[v],
                )
            outputs_eval[k] = {ele: eval_results[ele][0] for ele in eval_results}
        # adding the evaluation outputs
        res["eval_task"] = outputs_eval
    return all_results


# Example usage:
# python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ITA/diff/results_icl4_tok30_ITA-ENG.json --lang_to_eval it --eval_task_requires_translation ita_Latn --dataset_name ITA_train
def main(
    evaluator_model_name: str,    # 'meta-llama/Meta-Llama-Guard-2-8B'
    results_json: str,
    lang_to_eval: str,      # ['en', 'fr', 'es', 'it', 'zh-cn']
    eval_task_requires_translation: str | None = None,      # if None, no translation is done, else translation is done according to the value ['ita_Latn', 'fra_Latn', 'spa_Latn', 'zho_Hans']
    dataset_name: str | None = None,
    ):

    print(f"[-] Loading results from {results_json}")

    with open(results_json, "r") as f:
        all_results = json.load(f)

    _, _, task = load_dataset(dataset_name)

    # eval lang
    all_results = eval_lang(all_results, lang_to_eval=lang_to_eval)

    # translate to english if required
    if eval_task_requires_translation:
        all_results = out_to_eng(all_results, source_lang=eval_task_requires_translation)
        output_cell = "output_eng"
    else:
        output_cell = "output"

    evaluator = Evaluator(
        evaluation_model_name=evaluator_model_name,
        load_in_8bit=True,
        task=task,
    )
    print(task)
    all_results = evaluate_task(all_results, evaluator, output_cell)


    new_name = results_json.replace("results_", "all_evaluated_")
    with open(new_name, "w") as f:
        json.dump(all_results, f, indent=4, ensure_ascii=False)

    
    
if __name__ == "__main__":
    fire.Fire(main)
