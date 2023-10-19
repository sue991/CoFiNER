import argparse
import gc
import logging
import os
import random
from datetime import datetime
from copy import deepcopy
from collections import defaultdict
import json
import pandas as pd

import numpy as np
import torch
from torch.nn import CrossEntropyLoss, Softmax
from torch.utils.data import DataLoader
from tqdm import tqdm
from transformers import BertConfig, BertForTokenClassification, BertTokenizer, RobertaTokenizer, RobertaConfig, \
    RobertaForTokenClassification

from engine import get_labels, SingleTaskDataset, MultiTaskDataset, MultitaskBatcher, Collator
# from trainer import Train, evaluate
from util.model import MultiTaskModel

root_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(root_dir)  # absolute path

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    # "dualBert": (BertConfig, BERTWithDualClassification, BertTokenizer),
    "dualBert": (BertConfig, MultiTaskModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)
}


def get_args_parser():
    parser = argparse.ArgumentParser('Making soft mapping matrix with datasets', add_help=False)

    parser.add_argument('--fine_dataset', type=str, required=True,
                        help="Fine dataset")

    parser.add_argument('--coarse_dataset', type=str, required=True)

    parser.add_argument("--mapping_top_k", default=0, type=int,
                        help="Number of top k labels to be considered for soft mapping matrix. if 0, all labels are considered.")

    parser.add_argument('--data_dir', default="./Data", type=str, help="The input data dir.")

    parser.add_argument("--labels", default="label.txt", type=str,
                        help="filename containing labels.")

    parser.add_argument('--batch_size', '-bs', default=32, type=int,
                        help="Train batch size.")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="Path to pre-trained model or shortcut name")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument('--seed', type=int, default=1004)

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")

    parser.add_argument("--hierarchy", default="hierarchy", type=str,
                        help="Filename of hierarchy file for making hard mapping matrix.")

    parser.add_argument('--gpu', default='1', type=str,
                        help="GPU ID")

    return parser


def make_logger(name, TODAY):
    log_dir = f'./logs/'

    if not os.path.exists(log_dir):
        os.makedirs(log_dir)

    logger = logging.getLogger(name)
    logger.setLevel(logging.INFO)
    formatter = logging.Formatter('%(asctime)s [%(levelname)8s] - %(message)s')

    # StreamHandler : console message
    stream_formatter = logging.Formatter('%(asctime)s: %(message)s', "%H:%M:%S")
    stream_handler = logging.StreamHandler()
    stream_handler.setFormatter(stream_formatter)
    logger.addHandler(stream_handler)

    # FileHandler : File message
    file_handler = logging.FileHandler(f'{log_dir}/{TODAY}.log', encoding='utf-8')
    file_handler.setFormatter(formatter)
    logger.addHandler(file_handler)

    return logger


def make_coarse_dict(data_dir, name, label_path):
    data_path = os.path.join(data_dir, name, label_path)
    data_labels = get_labels(data_path)

    label2idx = {c: i for i, c in enumerate(data_labels)}
    idx2label = {i: c for i, c in enumerate(data_labels)}
    print(f"{name} labels {len(data_labels)}: {data_labels}")

    return label2idx, idx2label


def evaluate(args, model, dataloader, label2idx_dict):
    fine_len = len(label2idx_dict[args.fine_dataset])
    coarse_len = len(label2idx_dict[args.coarse_dataset])
    mapping_matrix = np.zeros((fine_len, coarse_len))

    model.eval()
    for step, (batchMetaData, batchData) in enumerate(tqdm(dataloader, desc="Evaluating")):
        datasetName = batchMetaData['datasetName']
        with torch.no_grad():
            # copy true labels of fine-grained dataset in batch
            true = deepcopy(batchData['labels']).detach().cpu().numpy()
            tensor_labels = batchData['labels'].clone().detach()
            # mask true labels because we don't need to calculate loss
            tensor_labels = tensor_labels.masked_fill(tensor_labels > 0, 0)
            batchData['labels'] = tensor_labels

            outputs = model(**batchData)
            _, logits = outputs.loss, outputs.logits
            pred = torch.argmax(logits, dim=2)

            # counting predictions for each fine-grained label
            for i in range(true.shape[0]):
                for j in range(true.shape[1]):
                    if true[i, j] != args.pad_token_label_id:
                        mapping_matrix[true[i, j]][pred[i, j]] += 1
    print("done")
    return mapping_matrix


def make_top_k_mapping_matrix(k, mapping_matrix):
    prob_mapping_matrix = np.zeros(mapping_matrix.shape)
    for i in range(mapping_matrix.shape[0]):
        top_k = np.argsort(mapping_matrix[i, :])[-k:]
        prob_mapping_matrix[i, top_k] = mapping_matrix[i, top_k] / np.sum(mapping_matrix[i, top_k])
    return prob_mapping_matrix


def main(args):
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    torch.set_float32_matmul_precision('high')

    coarse_path = os.path.join(args.data_dir, args.coarse_dataset, args.labels)
    labels = get_labels(coarse_path)  # coarse labels

    num_labels = len(labels)

    label2idx_dict, idx2label_dict = {}, {}
    mapping_datasets = [args.coarse_dataset, args.fine_dataset]
    for dataset in mapping_datasets:
        label2idx, idx2label = make_coarse_dict(args.data_dir, dataset, args.labels)
        label2idx_dict[dataset] = label2idx
        idx2label_dict[dataset] = idx2label

    _, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=True)

    ## Load Coarse-grained model
    best_checkpoint = os.path.join(f"./output/{args.coarse_dataset}_{args.model_type}-large", str(args.seed),
                                   "checkpoint-best")

    model = model_class.from_pretrained(best_checkpoint)
    model.to(args.device)
    # model = torch.compile(model)

    ## Make Fine-grained dataset to inference
    args.pad_token_label_id = CrossEntropyLoss().ignore_index  # -100
    datasets = SingleTaskDataset(args, args.fine_dataset, tokenizer, labels, args.pad_token_label_id, label2idx_dict,
                                 'train')

    dataset = MultiTaskDataset([datasets])
    batcher = MultitaskBatcher(args, dataset, shuffleDataset=False, shuffleBatch=False)

    test_loader = DataLoader(dataset, batch_sampler=batcher, collate_fn=Collator(args).collate_fn)

    mapping_matrix = evaluate(args, model, test_loader, label2idx_dict)

    # Make probability mapping matrix for top_k labels
    prob_mapping_matrix = make_top_k_mapping_matrix(args.mapping_top_k, mapping_matrix)

    # find zero probability coarse label -> not predicted
    zero_prob_coarse = []

    for i, s in enumerate(np.sum(prob_mapping_matrix, 0)):  # sum of each column
        if s == 0:
            zero_prob_coarse.append(idx2label_dict[args.coarse_dataset][i])
            print(f"coarse label {idx2label_dict[args.coarse_dataset][i]} has no prediction")

    # save top_k mapping labels
    mapping_labels = []
    print(f"save new top-{args.mapping_top_k} mapping labels")

    mapping_labels_dir = os.path.join(args.data_dir, args.coarse_dataset, args.model_type, args.model_name_or_path)
    if not os.path.exists(mapping_labels_dir):
        os.makedirs(mapping_labels_dir)

    with open(os.path.join(mapping_labels_dir, f"label_{args.fine_dataset}_top{args.mapping_top_k}.txt"), 'w') as f:
        for label in labels:
            if label not in zero_prob_coarse:
                mapping_labels.append(label)
                f.write(label + "\n")

    df = pd.DataFrame(prob_mapping_matrix, index=list(idx2label_dict[args.fine_dataset].values()), columns=labels)
    df = df[mapping_labels]


    mapping_matrix_dir = os.path.join(args.data_dir, args.coarse_dataset, f"mapping_matrix_{args.model_type}", args.model_name_or_path)
    if not os.path.exists(mapping_matrix_dir):
        os.makedirs(mapping_matrix_dir)
    # save probability mapping matrix

    save_path = os.path.join(mapping_matrix_dir,
                             f"{args.fine_dataset}_top{args.mapping_top_k}.csv")
    print(f"save mapping matrix :", save_path)

    df.to_csv(save_path, index=True)
    return 0


if __name__ == '__main__':
    gc.collect()
    torch.cuda.empty_cache()

    now = datetime.now()
    today = now.strftime('%Y.%m.%d')

    parser = argparse.ArgumentParser('NER model training and evaluation script', parents=[get_args_parser()])
    args = parser.parse_args()

    # Logger
    logger = make_logger(__name__, today)
    args.logger = logger

    # Device to training/test
    device = torch.device(f"cuda:{args.gpu}" if torch.cuda.is_available() else "cpu")
    args.device = device

    main(args)
