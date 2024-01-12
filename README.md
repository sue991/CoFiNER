# Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets

This repository contains the code for the paper [Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets](https://aclanthology.org/2023.emnlp-main.197/), in EMNLP 2023.

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




## Citation
```bibtex
@inproceedings{lee-etal-2023-enhancing,
    title = "Enhancing Low-resource Fine-grained Named Entity Recognition by Leveraging Coarse-grained Datasets",
    author = "Lee, Su and Oh, Seokjin and Jung, Woohwan",
    editor = "Bouamor, Houda and Pino, Juan and Bali, Kalika",
    booktitle = "Proceedings of the 2023 Conference on Empirical Methods in Natural Language Processing",
    month = dec,
    year = "2023",
    address = "Singapore",
    publisher = "Association for Computational Linguistics",
    url = "https://aclanthology.org/2023.emnlp-main.197",
    doi = "10.18653/v1/2023.emnlp-main.197",
    pages = "3269--3279",
    abstract = "Named Entity Recognition (NER) frequently suffers from the problem of insufficient labeled data, particularly in fine-grained NER scenarios. Although $K$-shot learning techniques can be applied, their performance tends to saturate when the number of annotations exceeds several tens of labels. To overcome this problem, we utilize existing coarse-grained datasets that offer a large number of annotations. A straightforward approach to address this problem is pre-finetuning, which employs coarse-grained data for representation learning. However, it cannot directly utilize the relationships between fine-grained and coarse-grained entities, although a fine-grained entity type is likely to be a subcategory of a coarse-grained entity type. We propose a fine-grained NER model with a Fine-to-Coarse(F2C) mapping matrix to leverage the hierarchical structure explicitly. In addition, we present an inconsistency filtering method to eliminate coarse-grained entities that are inconsistent with fine-grained entity types to avoid performance degradation. Our experimental results show that our method outperforms both $K$-shot learning and supervised learning methods when dealing with a small number of fine-grained annotations."
}

```
