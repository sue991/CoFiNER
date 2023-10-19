import os

import pickle
import numpy as np
import torch
from torch.nn import NLLLoss, Softmax, CrossEntropyLoss
from tqdm import tqdm
from transformers import AdamW, get_linear_schedule_with_warmup, get_cosine_schedule_with_warmup
from engine import *
from util.loss import TotalLoss
from util.metric import Metrics
from collections import defaultdict


class Trainer:
    def __init__(self, summary, args, tokenizer, train_loader, dev_loader, test_loader, labels, coarse_dict,
                 pad_token_label_id, mask_dict, **kwargs):
        args.logger.info('Initializing Task')
        self._no_decay = ["bias", "LayerNorm.weight"]

        self._summary = summary
        self._args = args
        self._tokenizer = tokenizer
        self._train_loader = train_loader
        self._dev_loader = dev_loader
        self._test_loader = test_loader
        self._labels = labels
        self._coarse_dict = coarse_dict
        self._pad_token_label_id = pad_token_label_id
        self._mask_dict = mask_dict

        # Loss function
        self.criterion = TotalLoss(self._args.loss, self._args.coarse_datasets)

        # self.__dict__.update(locals())
        self.__dict__.update(**kwargs)

        self.prebert = self.prebert.to(self._args.device) if "prebert" in kwargs else None

        if args.max_steps > 0:
            self.t_total = args.max_steps
        else:
            self.train_len_loader = len(self._train_loader)
            self.t_total = self.train_len_loader * args.epochs

    def train(self, model):
        if self._args.max_steps > 0:
            self._args.epochs = self._args.max_steps // self.train_len_loader + 1
        self._args.logger.info("***** Running training *****")
        self._args.logger.info("  Train data loader length = %d", self.train_len_loader)
        self._args.logger.info("  Num Epochs = %d", self._args.epochs)
        self._args.logger.info("  Total train batch size = %d", self._args.batch_size)
        self._args.logger.info("  Total optimization steps = %d", self.t_total)

        # Train configurations
        optimizer_grouped_parameters = [
            {"params": [p for n, p in model.named_parameters() if not any(nd in n for nd in self._no_decay)],
             "weight_decay": self._args.weight_decay},
            {"params": [p for n, p in model.named_parameters() if any(nd in n for nd in self._no_decay)],
             "weight_decay": 0.0}
        ]
        optimizer = AdamW(optimizer_grouped_parameters, lr=self._args.learning_rate, eps=self._args.adam_epsilon,
                          no_deprecation_warning=True)
        scheduler = get_cosine_schedule_with_warmup(optimizer, num_warmup_steps=self._args.warmup_steps,
                                                    num_training_steps=self.t_total)
        global_step = 0
        tr_loss = 0.0
        best_metric = -999

        model.zero_grad()
        model.train()
        for epoch in range(self._args.epochs):
            self._args.logger.info(f'Training {epoch + 1} epoch..')
            train_loss = 0.0
            train_step = 0
            for batchMetaData, batchData in tqdm(self._train_loader, desc="Training"):
                dataset_name = batchMetaData['datasetName']
                # inference
                if self._args.model_type == "dualBert":
                    _, logits, coarse_logits = model(batchMetaData, batchData)
                else:
                    outputs = model(**batchData)
                    _, logits = outputs.loss, outputs.logits
                    coarse_logits = None

                ignore_index_mask = batchData['labels'] != self._args.pad_token_label_id  # -100

                if self.prebert and dataset_name in self._args.coarse_datasets:  # Filtering Coarse datasets
                    self.prebert.eval()
                    with torch.no_grad():
                        f_outputs = self.prebert(**batchData)
                        _, mask_logits = f_outputs.loss, f_outputs.logits
                        del f_outputs
                else:
                    mask_logits = None

                if self._args.save_predictions and dataset_name in self._args.coarse_datasets and epoch == 0:
                    metric = Metrics()
                    preds_list, out_label_list = metric.get_pred_label_list(dataset_name, mask_logits, batchData['labels'],
                                                                            self._mask_dict, self._coarse_dict,
                                                                            self._args.pad_token_label_id)

                    metric.save_predictions(preds_list, out_label_list, self._tokenizer, batchData['input_ids'], dataset_name, save_dir=f"{self._args.output_dir}/predictions")

                # Loss
                loss = self.criterion.calculate(dataset_name, logits, batchData['labels'],
                                                coarse_logits=coarse_logits,
                                                mask_matrix=self._mask_dict[dataset_name],
                                                ignore_index_mask=ignore_index_mask,
                                                mask_logits=mask_logits,
                                                alpha=self._args.alpha)

                # Backpropagation
                optimizer.zero_grad()
                loss.backward()
                optimizer.step()
                scheduler.step()

                # Metadata update
                tr_loss += loss.item()
                train_loss += loss.item()
                global_step += 1
                train_step += 1

                if 0 < self._args.max_steps < global_step:
                    break

                del batchMetaData, batchData, logits

            # add information to summary
            train_loss = train_loss / train_step
            self._args.logger.info(f'train loss after one epoch : {train_loss}')
            self._summary.scalar_plot("Loss", 'train', train_loss, global_step)

            # Evaluate
            # if self._args.do_eval and global_step % self._args.logging_steps == 0:
            if self._args.do_eval and (epoch + 1) % self._args.eval_steps == 0:
                # Save model checkpoint
                train_checkpoint = os.path.join(self._args.output_dir, "train_checkpoint")
                if not os.path.exists(train_checkpoint):
                    os.makedirs(train_checkpoint)
                    self._args.logger.info("Saving train checkpoint to %s", train_checkpoint)

                # evaluate
                results, eval_loss = self.evaluate(model, self._dev_loader, mode="dev", train_epoch=epoch+1,
                                                   prefix=global_step)

                # add information to summary
                self._summary.dict_plot("Loss/eval", eval_loss, global_step)

                f1 = []
                for result in results:
                    f1.append(results[result]['f1'])
                mean_f1 = np.mean(f1)

                self._args.logger.info(f"Evaluation Loss : {eval_loss}")

                # Save model checkpoint
                if mean_f1 > best_metric:
                    best_metric = mean_f1

                    output_dir = os.path.join(self._args.output_dir, "checkpoint-best")
                    if not os.path.exists(output_dir):
                        os.makedirs(output_dir)

                    model_to_save = model.module if hasattr(model,
                                                            "module") else model  # Take care of distributed/parallel training
                    model_to_save.save_pretrained(output_dir)
                    torch.save(self._args, os.path.join(output_dir,
                                                        "training_args.bin"))
                    self._tokenizer.save_pretrained(output_dir)
                    self._args.logger.info("Saving model checkpoint to %s", output_dir)

            if 0 < self._args.max_steps < global_step:
                break

        return global_step, tr_loss / global_step

    def evaluate(self, model, dataloader, prefix, mode, **kwargs):
        if self.prebert is not None:
            self.prebert.to(self._args.device).eval()

        metric = Metrics()
        class_cnt = defaultdict(list)
        prob_dict = defaultdict(list)
        results, total_eval_loss = {}, {}

        eval_loss = 0.0
        nb_eval_steps = 0

        model.eval()
        for step, (batchMetaData, batchData) in enumerate(tqdm(dataloader, desc="Evaluating")):
            dataset_name = batchMetaData['datasetName']
            with torch.no_grad():
                # inference
                if self._args.model_type == "dualBert":
                    _, logits, coarse_logits = model(batchMetaData, batchData)
                else:
                    outputs = model(**batchData)
                    loss, logits = outputs.loss, outputs.logits
                    coarse_logits = None
                    del outputs
            # Loss
            ignore_index_mask = batchData['labels'] != -100
            tmp_eval_loss = self.criterion.calculate(dataset_name, logits, batchData['labels'],
                                                     coarse_logits=coarse_logits,
                                                     mask_matrix=self._mask_dict[dataset_name],
                                                     ignore_index_mask=ignore_index_mask, alpha=self._args.alpha)

            # Metadata update
            eval_loss += tmp_eval_loss.item()
            nb_eval_steps += 1

            # Update class_cnt
            preds_list, out_label_list = metric.get_pred_label_list(dataset_name, logits, batchData['labels'],
                                                                    self._mask_dict, self._coarse_dict,
                                                                    self._args.pad_token_label_id)

            bs_prob_dict = metric.get_probabilities(dataset_name, logits, batchData['labels'],
                                                    self._mask_dict, self._coarse_dict,
                                                    self._args.pad_token_label_id)

            for k, v in bs_prob_dict.items():
                prob_dict[k].extend(v)

            bs_class_cnt = metric.class_f1_score(preds_list, out_label_list)
            for k, v in bs_class_cnt.items():
                if k in class_cnt:
                    class_cnt[k][0] += v[0]  # pred
                    class_cnt[k][1] += v[1]  # label
                    class_cnt[k][2] += v[2]  # corrct
                else:
                    class_cnt[k] = list(v)

            # memory management
            del tmp_eval_loss, logits, batchMetaData, batchData
        # eval_loss, class_cnt
        eval_loss = eval_loss / nb_eval_steps

        total_f1, class_report = metric.get_f1_score(class_cnt)
        result = {"precision": total_f1[0],
                  "recall": total_f1[1],
                  "f1": total_f1[2]}

        name = list(dataloader.dataset.get_dataset().keys())[0]
        self._args.logger.info("***** %s Eval results %s *****", name, prefix)
        for key in sorted(result.keys()):
            self._args.logger.info("  %s = %s", key, str(result[key]))

        #### Save class results
        save_class_path = os.path.join(self._args.output_dir, "class_report", mode)
        if not os.path.exists(save_class_path):
            os.makedirs(save_class_path)

        if mode == "dev":
            if kwargs['train_epoch'] % 10 == 0:  # During training
                class_f1_path = os.path.join(save_class_path, f"{kwargs['train_epoch']}_class_f1_score.csv")
                metric.save_class_report(class_report, class_cnt, path=class_f1_path)
        else:
            self._args.logger.info("Saving class report to %s", save_class_path)
            class_f1_path = os.path.join(save_class_path, f"{mode}_class_f1_score.csv")
            metric.save_class_report(class_report, class_cnt, path=class_f1_path)

        if prefix and prefix != 'Best Checkpoint':
            self._summary.dict_plot(f'Metric/{name}', result, prefix)
        results[name], total_eval_loss[name] = result, eval_loss
        return results, total_eval_loss
