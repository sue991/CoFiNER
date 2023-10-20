# Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets

This repository contains the code for the paper [Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets](https://arxiv.org/abs/2310.11715), in EMNLP 2023.

## Requirements
    pytorch==2.0.1
    transformers==4.29.2

## Dataset
Because of the license of the datasets, we can only provide the CoNLL'03. You can download the other datasets from the following links:

- [Few-NERD](https://ningding97.github.io/fewnerd/)
- [OntoNote](https://catalog.ldc.upenn.edu/LDC2013T19)

## How to Run
CoFiNER is composed of a training process consisting of 3 steps.

### Step 1: Training a fine-grained model
In the first step, we train a fine-grained model with the low-resource fine-grained dataset. This model serves as an inconsistency filtering.
```bash
python3 main.py --fine_dataset Few-NERD_100 --eval_data Few-NERD --test_data Few-NERD --epochs 30 --overwrite_output_dir  --overwrite_cache  --do_train --do_eval --do_predict --model_type roberta --model_name_or_path roberta-large --suffix roberta-large 
```


### Step 2: Construction of the F2C mapping matrix
The F2C mapping matrix assesses the conditional probability of a coarse-grained entity type given a fine-grained label.

First, we create a model trained on the coarse-grained dataset.
```bash
python3 main.py --fine_dataset CoNLL --eval_data CoNLL --test_data CoNLL --epochs 50 --overwrite_output_dir --do_train --do_eval --do_predict --model_type roberta --model_name_or_path roberta-large --suffix roberta-large

python3 main.py --fine_dataset OntoNote --eval_data OntoNote --test_data OntoNote --epochs 50 --overwrite_output_dir --do_train --do_eval --do_predict --model_type roberta --batch_size 16 --model_name_or_path roberta-large --suffix roberta-large
```

Then, build a F2C mapping matrix:
```bash
python3 make_top_k_mapping.py --fine_dataset Few-NERD_100 --coarse_dataset CoNLL --mapping_top_k 1 --model_type roberta --model_name_or_path roberta-large

python3 make_top_k_mapping.py --fine_dataset Few-NERD_100 --coarse_dataset OntoNote --mapping_top_k 1 --model_type roberta --model_name_or_path roberta-large
```

### Step 3: Jointly training CoFiNER with both datasets
```bash
python3 main.py --fine_dataset Few-NERD_100 --coarse_datasets OntoNote --eval_data Few-NERD --test_data Few-NERD --epochs 30 --overwrite_output_dir --overwrite_cache --do_train --do_eval --do_predict --loss coarseFilter --mapping_top_k 1 --batch_size 16 --model_type roberta --model_name_or_path roberta-large --suffix roberta-large_top1_coarseFilter
```




## Citation(To Be Updated)
```LaTeX
@inproceedings{
anonymous2023enhancing,
title={Enhancing Low-Resource Fine-Grained Named Entity Recognition by Leveraging Coarse-Grained Datasets},
author={Anonymous},
booktitle={The 2023 Conference on Empirical Methods in Natural Language Processing},
year={2023},
url={https://openreview.net/forum?id=nIp7wkMeMP}
}
```
