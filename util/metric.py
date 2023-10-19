import itertools
import os
from collections import defaultdict
import torch
from torch.nn import Softmax

import pandas as pd
import matplotlib.pyplot as plt
import numpy as np


class Metrics:
    def __init__(self, ignore_index=-100):
        '''
        word_encoder: Sentence encoder

        You need to set self.cost as your own loss function.
        '''
        self.ignore_index = ignore_index

    def __get_class_span_dict__(self, label, is_string=False):
        '''
        return a dictionary of each class label/tag corresponding to the entity positions in the sentence
        {label:[(start_pos, end_pos), ...]}
        '''
        class_span = {}
        current_label = None
        i = 0
        if not is_string:
            # having labels in [0, num_of_class]
            while i < len(label):
                if label[i] > 0:
                    start = i
                    current_label = label[i]
                    i += 1

                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    assert label[i] == 0
                    i += 1
        else:
            # having tags in string format ['O', 'O', 'person-xxx', ..]
            while i < len(label):
                if label[i] != 'O':
                    # if label[i] not in ["O", "event-election", "other-astronomything", "other-biologything",
                    #                     "other-chemicalthing", "other-disease", "other-educationaldegree", "other-god",
                    #                     "other-livingthing", "other-medical", "product-food"]: # fine labels belonging to coarse label O
                    # if label[i] not in ['O', 'DATE', 'TIME', 'PERCENT', 'QUANTITY', 'ORDINAL', 'CARDINAL']:
                    # if label[i] not in ["O", "event-election", "other-astronomything", "other-biologything", "other-chemicalthing", "other-disease", "other-educationaldegree", "other-god", "other-livingthing", "other-medical", "product-food"]:
                    start = i
                    current_label = label[i]
                    i += 1
                    while i < len(label) and label[i] == current_label:
                        i += 1
                    if current_label in class_span:
                        class_span[current_label].append((start, i))
                    else:
                        class_span[current_label] = [(start, i)]
                else:
                    i += 1
        return class_span

    def __get_intersect_by_entity__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
        return cnt

    def __get_cnt__(self, label_class_span):
        '''
        return the count of entities
        '''
        cnt = 0
        for label in label_class_span:
            cnt += len(label_class_span[label])
        return cnt

    def __get_correct_span__(self, pred_span, label_span):
        '''
        return count of correct entity spans
        '''
        pred_span_list = []
        label_span_list = []
        for pred in pred_span:
            pred_span_list += pred_span[pred]
        for label in label_span:
            label_span_list += label_span[label]
        return len(list(set(pred_span_list).intersection(set(label_span_list))))

    def __get_wrong_within_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span, correct coarse type but wrong finegrained type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            within_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] == coarse:
                    within_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(within_pred_span))))
        return cnt

    def __get_wrong_outer_span__(self, pred_span, label_span):
        '''
        return count of entities with correct span but wrong coarse type
        '''
        cnt = 0
        for label in label_span:
            coarse = label.split('-')[0]
            outer_pred_span = []
            for pred in pred_span:
                if pred != label and pred.split('-')[0] != coarse:
                    outer_pred_span += pred_span[pred]
            cnt += len(list(set(label_span[label]).intersection(set(outer_pred_span))))
        return cnt

    def __get_type_error__(self, pred, label, query):
        '''
        return finegrained type error cnt, coarse type error cnt and total correct span count
        '''
        pred_tag, label_tag = self.__transform_label_to_tag__(pred, query)
        pred_span = self.__get_class_span_dict__(pred_tag, is_string=True)
        label_span = self.__get_class_span_dict__(label_tag, is_string=True)
        total_correct_span = self.__get_correct_span__(pred_span, label_span) + 1e-6
        wrong_within_span = self.__get_wrong_within_span__(pred_span, label_span)
        wrong_outer_span = self.__get_wrong_outer_span__(pred_span, label_span)
        return wrong_within_span, wrong_outer_span, total_correct_span

    def metrics_by_entity_(self, pred, label):
        '''
        return entity level count of total prediction, true labels, and correct prediction
        '''
        pred_class_span = self.__get_class_span_dict__(pred, is_string=True)
        label_class_span = self.__get_class_span_dict__(label, is_string=True)
        pred_cnt = self.__get_cnt__(pred_class_span)
        label_cnt = self.__get_cnt__(label_class_span)
        correct_cnt = self.__get_intersect_by_entity__(pred_class_span, label_class_span)
        return pred_cnt, label_cnt, correct_cnt

    def metrics_by_entity(self, pred, label):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0
        for i in range(len(pred)):
            p_cnt, l_cnt, c_cnt = self.metrics_by_entity_(pred[i], label[i])
            pred_cnt += p_cnt
            label_cnt += l_cnt
            correct_cnt += c_cnt

        print(f"# of pred, label, correct : {pred_cnt}, {label_cnt}, {correct_cnt}")
        precision = round(correct_cnt / (pred_cnt + 1e-8), 5)
        recall = round(correct_cnt / (label_cnt + 1e-8), 5)
        f1 = round(2 * precision * recall / (precision + recall + 1e-8), 5)
        return precision, recall, f1

    def __get_class_cnt__(self, label_class_span):
        '''
        return the count of entities per class
        '''
        cnt = {}
        for label in label_class_span:
            cnt[label] = len(label_class_span[label])
        return cnt

    def __get_intersect_by_entity_class__(self, pred_class_span, label_class_span):
        '''
        return the count of correct entity per class
        '''
        cnt = {}
        for label in label_class_span:
            cnt[label] = len(list(set(label_class_span[label]).intersection(set(pred_class_span.get(label, [])))))
        return cnt

    def class_f1_score_(self, pred, label):
        per_pred_class_span = self.__get_class_span_dict__(pred, is_string=True)
        per_label_class_span = self.__get_class_span_dict__(label, is_string=True)
        per_pred_cnt = self.__get_class_cnt__(per_pred_class_span)
        per_label_cnt = self.__get_class_cnt__(per_label_class_span)
        per_correct_cnt = self.__get_intersect_by_entity_class__(per_pred_class_span, per_label_class_span)

        return per_pred_cnt, per_label_cnt, per_correct_cnt

    def get_pred_label_list(self, dataset_name, logits, labels, mask_dict, coarse_dict, pad_token_label_id):
        ### coarse data를 예측한경우 그 결과를 fine data에 mapping 하여 preds_list, out_label_list를 만든다.
        softmax = Softmax(dim=2)
        sm_logits = softmax(logits)
        coarse_softmax = torch.matmul(sm_logits, mask_dict[dataset_name])
        coarse_softmax = coarse_softmax.detach().cpu().numpy()

        coarse_pred = np.argmax(coarse_softmax, axis=2)
        out_label_ids = labels.detach().cpu().numpy()

        label_map = {i: label for label, i in coarse_dict[dataset_name].items()}  # i: idx, label: class name

        out_label_list = [[] for _ in range(out_label_ids.shape[0])]
        preds_list = [[] for _ in range(out_label_ids.shape[0])]

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    out_label_list[i].append(label_map[out_label_ids[i][j]])
                    preds_list[i].append(label_map[coarse_pred[i][j]])
        return preds_list, out_label_list

    def get_probabilities(self, dataset_name, logits, labels, mask_dict, coarse_dict, pad_token_label_id):

        softmax = Softmax(dim=2)
        fine_prob = softmax(logits) # bs, seq_len, fine_labels

        coarse_prob = torch.matmul(fine_prob, mask_dict[dataset_name]) # bs, seq_len, coarse_labels
        coarse_proprobb = coarse_prob.detach().cpu().numpy()

        coarse_prob_dict = {label: [] for label in coarse_dict[dataset_name].keys()}
        label_map = {i: label for label, i in coarse_dict[dataset_name].items()}
        # coarse_pred = np.argmax(coarse_softmax, axis=2)
        out_label_ids = labels.detach().cpu().numpy()

        for i in range(out_label_ids.shape[0]):
            for j in range(out_label_ids.shape[1]):
                if out_label_ids[i, j] != pad_token_label_id:
                    coarse_prob_dict[label_map[out_label_ids[i][j]]].append(coarse_prob[i][j].tolist())

        return coarse_prob_dict



    def get_f1_score(self, class_cnt):
        pred_cnt = 0
        label_cnt = 0
        correct_cnt = 0

        class_metric = {}

        for label, value in class_cnt.items():
            pred_cnt += value[0]
            label_cnt += value[1]
            correct_cnt += value[2]

            try:
                precision = value[2] / (value[0] + 1e-8)
                recall = value[2] / (value[1] + 1e-8)
                f1 = 2 * precision * recall / (precision + recall)

            except Exception as e:  # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
                precision = 0
                recall = 0
                f1 = 0
                print("ERROR : ", e)

            # print(f"{label} pred, label, correct: {value[0]}, {value[1]}, {value[2]}")
            class_metric[label] = (precision, recall, f1)

        print(f"# of pred, label, correct : {pred_cnt}, {label_cnt}, {correct_cnt}")

        precision = round(correct_cnt / (pred_cnt + 1e-8), 5)
        recall = round(correct_cnt / (label_cnt + 1e-8), 5)
        f1 = round(2 * precision * recall / (precision + recall + 1e-8), 5)

        total_f1 = (precision, recall, f1)
        return total_f1, class_metric

    def class_f1_score(self, pred, label):

        pred_class_cnt = defaultdict(int)
        label_class_cnt = defaultdict(int)
        correct_class_cnt = defaultdict(int)

        for i in range(len(pred)):
            p_cnt, l_cnt, c_cnt = self.class_f1_score_(pred[i], label[i])

            for k, v in p_cnt.items():
                pred_class_cnt[k] += v

            for k, v in l_cnt.items():
                label_class_cnt[k] += v

            for k, v in c_cnt.items():
                correct_class_cnt[k] += v

        # class_metric = {}
        class_cnt = {}
        for label in label_class_cnt:
            # try:
            #     precision = correct_class_cnt[label] / (pred_class_cnt[label] + 1e-8)
            #     recall = correct_class_cnt[label] / (label_class_cnt[label] + 1e-8)
            #     f1 = 2 * precision * recall / (precision + recall)
            # except Exception as e:  # 모든 예외의 에러 메시지를 출력할 때는 Exception을 사용
            #     precision = 0
            #     recall = 0
            #     f1 = 0
            #     print("ERROR : ", e)

            # print(f"{label} pred, label, correct: {pred_class_cnt[label]}, {label_class_cnt[label]}, {correct_class_cnt[label]}")
            class_cnt[label] = (pred_class_cnt[label], label_class_cnt[label], correct_class_cnt[label])
            # class_metric[label] = (precision, recall, f1)
        return class_cnt

    def save_class_report(self, class_report, class_cnt, path=None):
        if path:
            report = pd.DataFrame(class_report, index=['precision', 'recall', 'f1']).T
            cnt = pd.DataFrame(class_cnt, index=['#pred', '#label', '#correct']).T

            result = pd.merge(report, cnt, left_index=True, right_index=True, how='outer')
            result.to_csv(path, index=True)

    def plot_confusion_matrix(self, con_mat, labels, save_path, title='Confusion Matrix', cmap=plt.cm.get_cmap('Blues'),
                              normalize=False, ):

        plt.figure(figsize=(25, 25))
        plt.imshow(con_mat, interpolation='nearest', cmap=cmap)
        plt.title(title)
        plt.colorbar()
        marks = np.arange(len(labels))
        nlabels = []
        for k in range(len(con_mat)):
            n = sum(con_mat[k])
            # nlabel = '{0}(n={1})'.format(labels[k], n)
            nlabel = '{0}'.format(labels[k])
            nlabels.append(nlabel)
        plt.xticks(marks, labels)
        plt.yticks(marks, nlabels)

        thresh = con_mat.max() / 2.
        if normalize:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, '{0}%'.format(con_mat[i, j] * 100 / n), horizontalalignment="center",
                         color="white" if con_mat[i, j] > thresh else "black")
        else:
            for i, j in itertools.product(range(con_mat.shape[0]), range(con_mat.shape[1])):
                plt.text(j, i, con_mat[i, j], horizontalalignment="center",
                         color="white" if con_mat[i, j] > thresh else "black")
        plt.tight_layout()
        plt.ylabel('True label')
        plt.xlabel('Predicted label')

        output_matrix_save_file = os.path.join(save_path, "confusion_matrix.png")
        plt.savefig(output_matrix_save_file)


    def save_predictions(self, preds_list, out_label_list, tokenizer, input_ids_list, dataset_name, save_dir):
        os.makedirs(save_dir, exist_ok=True)
        save_path = os.path.join(save_dir, f"{dataset_name}_predictions.txt")

        for i in range(len(preds_list)):
            preds = preds_list[i]
            out_label = out_label_list[i]
            input_ids = input_ids_list[i]

            with open(save_path, "a") as f:
                input_words = tokenizer.convert_ids_to_tokens(input_ids, skip_special_tokens=True)

                for j in range(len(input_words)):
                    if preds[j] == out_label[j]:
                        f.write(f"{input_words[j]}\t{out_label[j]}\n")
                    else:
                        f.write(f"{input_words[j]}\tO\n")
                f.write("\n")


            # save if pred and label are not same
