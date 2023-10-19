import argparse
import gc
import logging
import os
import random
from datetime import datetime
import json
import numpy as np
import torch
from torch.nn import CrossEntropyLoss
from transformers import BertConfig, BertForTokenClassification, BertTokenizer, RobertaTokenizer, RobertaConfig, \
    RobertaForTokenClassification

from engine import get_labels, make_mask_matrix, make_data_handlers
# from trainer import Train, evaluate
from train import Trainer
# from util.model import MultiTaskModel
from util.summary import TensorboardPlotter

root_dir = os.path.dirname(os.path.realpath(__file__))
os.chdir(root_dir)  # absolute path

MODEL_CLASSES = {
    "bert": (BertConfig, BertForTokenClassification, BertTokenizer),
    # "dualBert": (BertConfig, MultiTaskModel, BertTokenizer),
    'roberta': (RobertaConfig, RobertaForTokenClassification, RobertaTokenizer)
}


def get_args_parser():
    parser = argparse.ArgumentParser('NER model training and evaluation script', add_help=False)

    parser.add_argument('--fine_dataset', type=str, required=True, default='Few-NERD',
                        help="Fine-grained Train dataset Name [OntoNote, Few-NERD, CoNLL]")

    parser.add_argument('--coarse_datasets', nargs='+', type=str, default=[],
                        help="Coarse-grained Train datasets Name [OntoNote, Few-NERD, CoNLL]")

    parser.add_argument("--mapping_top_k", default=0, type=int, 
                        help="Number of top_k labels to be considered for mapping matrix. if 0, all labels are considered.")

    parser.add_argument('--eval_data', nargs='+', type=str, required=True,
                        help="Evaluation Datasets Name [OntoNote, Few-NERD, CoNLL]")

    parser.add_argument('--test_data', nargs='+', type=str, required=True,
                        help="Test Datasets Name [OntoNote, Few-NERD, CoNLL]")

    parser.add_argument('--data_dir', default="./Data", type=str,
                        help="The input data dir.")

    parser.add_argument("--labels", default="label", type=str,
                        help="filename containing fine-grained labels.")

    parser.add_argument("--coarse_labels", default="label", type=str,
                        help="filename containing coarse-grained labels.")

    parser.add_argument('--batch_size', '-bs', default=16, type=int,
                        help="Train batch size.")

    parser.add_argument("--model_type", default="bert", type=str,
                        help="Model type selected in the list: " + ", ".join(MODEL_CLASSES.keys()))

    parser.add_argument("--config_name", default="", type=str,
                        help="Pretrained config name or path if not the same as model_name")

    parser.add_argument("--model_name_or_path", default="bert-base-uncased", type=str,
                        help="Path to pre-trained model or shortcut name")

    parser.add_argument("--tokenizer_name", default="", type=str,
                        help="Pretrained tokenizer name or path if not the same as model_name")

    parser.add_argument("--output_dir", default="./output", type=str,
                        help="The output directory where the model predictions and checkpoints will be written.")

    parser.add_argument("--do_train", action="store_true",
                        help="Whether to run training.")

    parser.add_argument("--do_eval", action="store_true",
                        help="Whether to run evaluation on the dev set.")

    parser.add_argument("--do_predict", action="store_true",
                        help="Whether to run predictions on the test set.")

    parser.add_argument("--max_seq_length", default=256, type=int,
                        help="The maximum total input sequence length after tokenization. Sequences longer "
                             "than this will be truncated, sequences shorter will be padded.")

    parser.add_argument("--max_steps", default=-1, type=int,
                        help="If > 0: set total number of training steps to perform.")

    parser.add_argument("--epochs", default=30, type=int,
                        help="# training epochs.")

    parser.add_argument("--overwrite_cache", action="store_true",
                        help="Overwrite the cached training and evaluation sets")

    parser.add_argument("--overwrite_output_dir", action="store_true",
                        help="Overwrite the content of the output directory")

    parser.add_argument('--seed', default=1004, type=int,
                        help="Random Seed for pytorch, numpy ...")

    parser.add_argument("--suffix", default="",
                        help="Add suffix in load Dataset and Model save path.")

    parser.add_argument("--alpha", default=1e-5, type=float,
                        help="hyper param. e.g. (alpha)*KLDivLoss + CELoss")

    parser.add_argument("--save_predictions", action="store_true",
                        help="Save prediction results.")

    parser.add_argument('--gpu', default='0', type=str,
                        help="GPU ID")

    ## Optimizer
    parser.add_argument("--loss", default="CustomCELoss", type=str,
                        help="Method of calculate loss. [LogSoftmax, LSE]")

    parser.add_argument("--warmup_steps", default=0, type=int,
                        help="Linear warmup over warmup_steps.")

    parser.add_argument("--learning_rate", default=2e-5, type=float,
                        help="The initial learning rate for Adam.")

    parser.add_argument("--weight_decay", default=0.0, type=float,
                        help="Weight decay if we apply some.")

    parser.add_argument("--adam_epsilon", default=1e-8, type=float,
                        help="Epsilon for Adam optimizer.")

    parser.add_argument('--eval_steps', default=1, type=int)
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


def make_output_dir(args):
    if args.coarse_datasets:
        coarse_datasets = '+'.join(args.coarse_datasets)
        using_data = "+".join([coarse_datasets, args.fine_dataset])
    else:
        using_data = args.fine_dataset
    if args.suffix:
        using_data = f"{using_data}_{args.suffix}"
    output_dir = os.path.join(args.output_dir, using_data, str(args.seed))
    os.makedirs(output_dir, exist_ok=True)
    return using_data, output_dir


def main(args):
    ## seed 고정
    torch.manual_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    torch.cuda.manual_seed(args.seed)
    # torch.cuda.manual_seed_all(args.seed)  # if use multi-GPU
    
    # torch.backends.cudnn.deterministic = True
    # torch.backends.cudnn.benchmark = False

    # Make output_dir
    writer_path, args.output_dir = make_output_dir(args)
    if os.path.exists(args.output_dir) and os.listdir(
            args.output_dir) and args.do_train and not args.overwrite_output_dir:
        raise ValueError(
            "Output directory ({}) already exists and is not empty. Use --overwrite_output_dir to overcome.".format(
                args.output_dir))

    # Make Tensorboard
    summary = TensorboardPlotter(f'runs/{writer_path}')
    args.logger.info(f"model output directory : {args.output_dir}")
    args.logger.info(f"Device : {args.device}")

    '''
        # labels : fine-grained labels
        # num_labels : number of fine-grained labels
        # mapping_dict : {"Few-NERD" :  67x67,
        #                 "OntoNote" : 67*11}
        # coarse_dict : {dataset name : {label : label index}}
        # mapping : {fine-grained label : dataset label}
    '''
    fine_path = os.path.join(args.data_dir, args.fine_dataset, f"{args.labels}.txt")
    labels = get_labels(fine_path)  # fine labels
    num_labels = len(labels)
    # args.logger.info(f'{num_labels} {labels}')
    mapping_dict, label2idx_dict, mapping = {}, {}, {}

    mapping_datasets = list(set([args.fine_dataset] + args.coarse_datasets + args.eval_data + args.test_data))

    for dataset in mapping_datasets:
        label2idx, mapping_matrix = make_mask_matrix(args, labels, dataset)
        mapping_dict[dataset] = torch.Tensor(mapping_matrix).to(args.device)
        label2idx_dict[dataset] = label2idx

    args.pad_token_label_id = CrossEntropyLoss().ignore_index  # -100

    # Load pretrained model and tokenizer
    config_class, model_class, tokenizer_class = MODEL_CLASSES[args.model_type]
    config = config_class.from_pretrained(args.config_name if args.config_name else args.model_name_or_path,
                                          num_labels=num_labels
                                          )

    tokenizer = tokenizer_class.from_pretrained(args.tokenizer_name if args.tokenizer_name else args.model_name_or_path,
                                                do_lower_case=True)
    # Load datasets
    train_dataset, _, train_loader = make_data_handlers(args, tokenizer, labels, args.pad_token_label_id,
                                                        label2idx_dict,
                                                        mode="train")
    _, _, dev_loader = make_data_handlers(args, tokenizer, labels, args.pad_token_label_id, label2idx_dict, mode="dev")
    _, _, test_loader = make_data_handlers(args, tokenizer, labels, args.pad_token_label_id, label2idx_dict,
                                           mode="test")

    if args.model_type == "dualBert":
        model = model_class(config, label2idx_dict)
    else:  # bert
        model = model_class.from_pretrained(args.model_name_or_path, config=config)
    model.to(args.device)
    # model = torch.compile(model)  # for pytorch 2.0

    args.logger.info("Training/evaluation parameters %s", args)

    with open(os.path.join(args.output_dir, 'arguments.txt'), 'w') as f:
        args_list = { key: str(args.__dict__[key]) for key in args.__dict__ if key != "logger"}
        f.write(json.dumps(args_list, indent=2))
        del args_list

    ## Declare pre-trained filtering model
    if "Filter" in args.loss:  # ["coarseFilter", "dualFilter", 'focalFilter', 'maxFilter']
        best_checkpoint = os.path.join(f"./output/{args.fine_dataset}_{args.model_name_or_path}", str(args.seed),
                                       "checkpoint-best") 

        if args.model_type == "dualBert":
            prebert = BertForTokenClassification.from_pretrained(best_checkpoint)
        else:
            prebert = model_class.from_pretrained(best_checkpoint)

    if  "Filter" in args.loss:
        ner_train = Trainer(summary, args, tokenizer, train_loader, dev_loader, test_loader, labels, label2idx_dict,
                            args.pad_token_label_id, mapping_dict, prebert=prebert)
    else:
        ner_train = Trainer(summary, args, tokenizer, train_loader, dev_loader, test_loader, labels, label2idx_dict,
                            args.pad_token_label_id, mapping_dict)

    # Training
    if args.do_train:
        if summary.clear_dir(f'runs/{writer_path}'):
            args.logger.info(f"clear summary dir: runs/{writer_path}")

        # Train
        global_step, train_loss = ner_train.train(model)
        args.logger.info("global_step = %s, average loss = %s", global_step, train_loss)

    best_checkpoint = os.path.join(args.output_dir, "checkpoint-best")

    if args.do_train and not os.path.exists(best_checkpoint):
        os.makedirs(best_checkpoint)
        args.logger.info("Saving model checkpoint to %s", best_checkpoint)
        # Save a trained model, configuration and tokenizer using `save_pretrained()`.
        # They can then be reloaded using `from_pretrained()`
        model_to_save = model.module if hasattr(model, "module") else model
        model_to_save.save_pretrained(best_checkpoint)
        tokenizer.save_pretrained(best_checkpoint)
        torch.save(args, os.path.join(best_checkpoint, "training_args.bin"))

    # Evaluation
    if args.do_eval:
        ## f1 best_checkpoint
        args.logger.info("Evaluate the following checkpoints: %s", best_checkpoint)

        if args.model_type == "dualBert":
            model = model_class.from_pretrained(best_checkpoint, label2idx_dict)
        else:
            model = model_class.from_pretrained(best_checkpoint)

        model.to(args.device)

        if "Filter" in args.loss:  # ["coarseFilter", "dualFilter", 'focalFilter', 'maxFilter']
            results, eval_losses = ner_train.evaluate(model, dev_loader, prefix="Best Checkpoint", mode="eval",
                                                      prebert=prebert)

        else:
            results, eval_losses = ner_train.evaluate(model, dev_loader, prefix="Best Checkpoint", mode="eval")

        output_eval_file = os.path.join(args.output_dir, "eval_results.txt")

        with open(output_eval_file, "w") as writer:
            for name, metric in results.items():
                writer.write("{} = {}\n".format(name, str(metric)))

    # Test
    if args.do_predict:
        # f1_best-checkpoint
        args.logger.info("Predict the following checkpoints: %s", best_checkpoint)

        if args.model_type == "dualBert":
            model = model_class.from_pretrained(best_checkpoint, label2idx_dict)
        else:
            model = model_class.from_pretrained(best_checkpoint)

        model.to(args.device)

        if "Filter" in args.loss:  # ["coarseFilter", "dualFilter", 'focalFilter', 'maxFilter']
            test_results, _ = ner_train.evaluate(model, test_loader, prefix="Best Checkpoint", mode="test",
                                                 prebert=prebert, save_path=args.output_dir)

        else:
            test_results, _ = ner_train.evaluate(model, test_loader, prefix="Best Checkpoint", mode="test",
                                                 save_path=args.output_dir)

        ## Save results
        output_test_results_file = os.path.join(args.output_dir, "test_results.txt")
        with open(output_test_results_file, "w") as writer:
            for name, metric in test_results.items():
                writer.write("{} = {}\n".format(name, str(metric)))

    summary.close()


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
