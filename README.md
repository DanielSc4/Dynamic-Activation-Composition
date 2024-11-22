# Multi-property Steering of Large Language Models with Dynamic Activation Composition


<div align="center">
<a href="https://www.danielsc4.it/">Daniel Scalena</a>
-
<a href="https://gsarti.com/">Gabriele Sarti</a>
-
<a href="https://malvinanissim.github.io/">Malvina Nissim</a>

<img src="media/Drawing_paper_git.png" alt="drawing" width="500"/>
</div>

This repository contains scripts and notebooks associated to the paper [Multi-property Steering of Large Language Models with Dynamic Activation Composition](https://arxiv.org/abs/2406.17563).

Abstract:
> Activation steering methods were shown to be effective in conditioning language model generation by additively intervening over models' intermediate representations. However, the evaluation of these techniques has so far been limited to single conditioning properties and synthetic settings. In this work, we conduct a comprehensive evaluation of various activation steering strategies, highlighting the property-dependent nature of optimal parameters to ensure a robust effect throughout generation. To address this issue, we propose Dynamic Activation Composition, an information-theoretic approach to modulate the steering intensity of one or more properties throughout generation. Our experiments on multi-property steering show that our method successfully maintains high conditioning while minimizing the impact of conditioning on generation fluency.

## Test Dynamic Activation Composition
You can try the online demo available on [🤗 HuggingFace Spaces](https://TODO.com).

## Citation
```bibtex
@inproceedings{scalena-etal-2024-multi,
    title = "Multi-property Steering of Large Language Models with Dynamic Activation Composition",
    author = "Scalena, Daniel  and
      Sarti, Gabriele  and
      Nissim, Malvina",
    editor = "Belinkov, Yonatan  and
      Kim, Najoung  and
      Jumelet, Jaap  and
      Mohebbi, Hosein  and
      Mueller, Aaron  and
      Chen, Hanjie",
    booktitle = "Proceedings of the 7th BlackboxNLP Workshop: Analyzing and Interpreting Neural Networks for NLP",
    month = nov,
    year = "2024",
    address = "Miami, Florida, US",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2024.blackboxnlp-1.34",
    pages = "577--603",
}
```

## Reproducing the Paper Results

### Single property steering

Single property steering consists of making the difference of internal model activations between two properties (see the paper for more details). The `diff_main.py` script does all this automatically by receiving as input the model (`mistralai/Mistral-7B-Instruct-v0.2`), the datasets and other parameters for generating and/or evaluating the responses.

Language steering:
```bash
echo "making the model speak italian (ITA)"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ITA_train \
    --dataset_B_name ENG_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "making the model speak french (FRA)"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name FRA_train \
    --dataset_B_name ENG_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "making the model speak spanish (SPA)"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name SPA_train \
    --dataset_B_name ENG_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "making the model speak chinese (ZHO)"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ZHO_train \
    --dataset_B_name ENG_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \
```

Safety steering:
```bash
echo "making the model safe"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ENsafe_train \
    --dataset_B_name ENunsafe_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "making the model unsafe"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ENunsafe_train \
    --dataset_B_name ENsafe_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \
```


Formal - Informal steering:
```bash
echo "making the model formal"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ENi2f_train \
    --dataset_B_name ENf2i_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "making the model informal"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ENf2i_train \
    --dataset_B_name ENi2f_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \
```


### Manual multi-property steering
Manual composition combine manually two properties (one behavioral and one language). It is used only for the purpose of providing a baseline for subsequent multi-property settings.
In general it is done by translating a behavioral task (e.g. safe) into the language of the second property (e.g. Italian) to have a result from the ICL that is both e.g Safe and Italian.

Unsafe and Language:
```bash
echo "[manual compositionality] making the model speak italian and unsafe"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ITunsafe_train \
    --dataset_B_name ENsafe_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "[manual compositionality] making the model speak french and unsafe"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name FRunsafe_train \
    --dataset_B_name ENsafe_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "[manual compositionality] making the model speak spanish and unsafe"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name SPunsafe_train \
    --dataset_B_name ENsafe_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "[manual compositionality] making the model speak chinese and unsafe"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ZHunsafe_train \
    --dataset_B_name ENsafe_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \
```


Informal and Language:
```bash
echo "[manual compositionality] making the model speak italian and informal"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name ITf2i_train \
    --dataset_B_name ENi2f_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \

echo "[manual compositionality] making the model speak french and informal"
python diff_main.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit \
    --dataset_A_name FRf2i_train \
    --dataset_B_name ENi2f_train \
    --icl_examples 4 \
    --support 30 \
    --evaluation_size 50 \
    --max_new_tokens 30 \
```

### Multi-property steering
Multi property steering is done by the `compositionality.py` script. The script receives all the datasets needed for the different configurations (noICL, ICL, and the multi-property settings) as well as the Deltas extracted in the previous steps (parameters `diff_path1` and `diff_path2`).

```bash
echo "On dataset ENunsafe composing the output to be Italian and unsafe (ITA + ENunsafe = ITunsafe)"
python compositionality.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit True \
    --source_dataset ENunsafe_train \
    --lang_dataset ITA_train \
    --source_lang_dataset ITunsafe_train \
    --diff_path1 output/Mistral-7B-Instruct-v0.2/ENunsafe/diff/diff_mean_act_icl4_tok30_ENunsafe-ENsafe.pt \
    --diff_path2 output/Mistral-7B-Instruct-v0.2/ITA/diff/diff_mean_act_icl4_tok30_ITA-ENG.pt \
    --ICL_examples 4 \
    --subset_eval 50 \

echo "On dataset ENunsafe composing the output to be French and unsafe (FRA + ENunsafe = FRunsafe)"
python compositionality.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit True \
    --source_dataset ENunsafe_train \
    --lang_dataset FRA_train \
    --source_lang_dataset FRunsafe_train \
    --diff_path1 output/Mistral-7B-Instruct-v0.2/ENunsafe/diff/diff_mean_act_icl4_tok30_ENunsafe-ENsafe.pt \
    --diff_path2 output/Mistral-7B-Instruct-v0.2/FRA/diff/diff_mean_act_icl4_tok30_FRA-ENG.pt \
    --ICL_examples 4 \
    --subset_eval 50 \
    
echo "On dataset ENunsafe composing the output to be Spanish and unsafe (SPA + ENunsafe = SPunsafe)"
python compositionality.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit True \
    --source_dataset ENunsafe_train \
    --lang_dataset SPA_train \
    --source_lang_dataset SPunsafe_train \
    --diff_path1 output/Mistral-7B-Instruct-v0.2/ENunsafe/diff/diff_mean_act_icl4_tok30_ENunsafe-ENsafe.pt \
    --diff_path2 output/Mistral-7B-Instruct-v0.2/SPA/diff/diff_mean_act_icl4_tok30_SPA-ENG.pt \
    --ICL_examples 4 \
    --subset_eval 50 \

echo "On dataset ENunsafe composing the output to be Chinese and unsafe (ZHO + ENunsafe = ZHunsafe)"
python compositionality.py \
    --model_name mistralai/Mistral-7B-Instruct-v0.2 \
    --load_in_8bit True \
    --source_dataset ENunsafe_train \
    --lang_dataset ZHO_train \
    --source_lang_dataset ZHunsafe_train \
    --diff_path1 output/Mistral-7B-Instruct-v0.2/ENunsafe/diff/diff_mean_act_icl4_tok30_ENunsafe-ENsafe.pt \
    --diff_path2 output/Mistral-7B-Instruct-v0.2/ZHO/diff/diff_mean_act_icl4_tok30_ZHO-ENG.pt \
    --ICL_examples 4 \
    --subset_eval 50 \
```


## Evaluation
The scripts above generate according to the specified settings several json files containing the model generations. For the evaluation of the generations (and thus the production of the observed results in the paper) the `evall_all.py` script is employed. This deals with both language evaluation through `langdetect` and safety evaluation (with `meta-llama/Meta-Llama-Guard-2-8B`) or formality evaluation (with `s-nlp/xlmr_formality_classifier`).

### Single property steering

Language:
```bash
echo "eval italian"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ITA/diff/results_icl4_tok30_ITA-ENG.json --lang_to_eval it --eval_task_requires_translation ita_Latn --dataset_name ITA_train

echo "eval french"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/FRA/diff/results_icl4_tok30_FRA-ENG.json --lang_to_eval fr --eval_task_requires_translation fra_Latn --dataset_name FRA_train

echo "eval spanish"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/SPA/diff/results_icl4_tok30_SPA-ENG.json  --lang_to_eval es --eval_task_requires_translation spa_Latn --dataset_name SPA_train

echo "eval chinese"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ZHO/diff/results_icl4_tok30_ZHO-ENG.json  --lang_to_eval zh-cn --eval_task_requires_translation zho_Hans --dataset_name ZHO_train
```

Safety:
```bash
echo "eval safe"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ENsafe/diff/results_icl4_tok30_ENsafe-ENunsafe.json --lang_to_eval en --dataset_name ENsafe_train

echo "eval unsafe"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ENunsafe/diff/results_icl4_tok30_ENunsafe-ENsafe.json  --lang_to_eval en --dataset_name ENunsafe_train
```

Formal - Informal
```bash
echo "eval to informal (en)"
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/ENf2i/diff/results_icl4_tok30_ENf2i-ENi2f.json --lang_to_eval en --dataset_name ENf2i_train

echo "eval to formal (en)"
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/ENi2f/diff/results_icl4_tok30_ENi2f-ENf2i.json --lang_to_eval en --dataset_name ENi2f_train
```


### Manual multi-property steering

Safety:
```bash
echo "eval ITunsafe"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ITunsafe/diff/results_icl4_tok30_ITunsafe-ENsafe.json --lang_to_eval it --eval_task_requires_translation ita_Latn --dataset_name ITunsafe_train

echo "eval FRunsafe"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/FRunsafe/diff/results_icl4_tok30_FRunsafe-ENsafe.json --lang_to_eval fr --eval_task_requires_translation fra_Latn --dataset_name FRunsafe_train

echo "eval SPunsafe"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/SPunsafe/diff/results_icl4_tok30_SPunsafe-ENsafe.json --lang_to_eval es --eval_task_requires_translation spa_Latn --dataset_name SPunsafe_train

echo "eval ZHunsafe"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/ZHunsafe/diff/results_icl4_tok30_ZHunsafe-ENsafe.json --lang_to_eval zh-cn --eval_task_requires_translation zho_Hans --dataset_name ZHunsafe_train
```

Formality:
```bash
echo "eval ITf2i"
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/ITf2i/diff/results_icl4_tok30_ITf2i-ENi2f.json --lang_to_eval it --eval_task_requires_translation ita_Latn --dataset_name ITf2i_train

echo "eval FRf2i"
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/FRf2i/diff/results_icl4_tok30_FRf2i-ENi2f.json --lang_to_eval fr --eval_task_requires_translation fra_Latn --dataset_name FRf2i_train
```




### Multi-property steering

Unsafe:
```bash
######## safe-unsafe
echo "eval ITunsafe comp"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENunsafe/ENunsafe-ITA/results_ENunsafe_ITA_tok30.json --lang_to_eval it --eval_task_requires_translation ita_Latn --dataset_name ITunsafe_train

echo "eval FRunsafe comp"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENunsafe/ENunsafe-FRA/results_ENunsafe_FRA_tok30.json --lang_to_eval fr --eval_task_requires_translation fra_Latn --dataset_name FRunsafe_train

echo "eval SPunsafe comp"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENunsafe/ENunsafe-SPA/results_ENunsafe_SPA_tok30.json --lang_to_eval es --eval_task_requires_translation spa_Latn --dataset_name SPunsafe_train

echo "eval ZHunsafe comp"
python eval_all.py --evaluator_model_name meta-llama/Meta-Llama-Guard-2-8B --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENunsafe/ENunsafe-ZHO/results_ENunsafe_ZHO_tok30.json --lang_to_eval zh-cn --eval_task_requires_translation zho_Hans --dataset_name ZHunsafe_train

```

Formality:
```bash
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENi2f/ENi2f-ITA/results_ENi2f_ITA_tok30.json --lang_to_eval it --dataset_name ITf2i_train
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENi2f/ENi2f-FRA/results_ENi2f_FRA_tok30.json --lang_to_eval fr --dataset_name FRf2i_train

python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENf2i/ENf2i-ITA/results_ENf2i_ITA_tok30.json --lang_to_eval it --dataset_name ITi2f_train
python eval_all.py --evaluator_model_name s-nlp/xlmr_formality_classifier --results_json output/Mistral-7B-Instruct-v0.2/compositionality/ENf2i/ENf2i-FRA/results_ENf2i_FRA_tok30.json --lang_to_eval fr --dataset_name FRi2f_train
```

### Artifacts
Artifacts (produced by the above scripts) can be requested directly by email to (Daniel Scalena: d [dot] scalena [at] rug [dot] nl).
