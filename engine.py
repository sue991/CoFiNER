import json
import os
import pickle
import random

import numpy as np
import pandas as pd
import torch
from torch.utils.data import DataLoader, Dataset, BatchSampler

from util.reader import Reader


def open_pickle(path, name):
    with open(f'{path}.pkl', 'rb') as fr:
        h = pickle.load(fr)
    hierarchy = h[name]
    return hierarchy


def open_json(path, name):
    name = name.split("_")[0]
    with open(f'{path}.json', 'r') as f:
        h = json.load(f)
    hierarchy = h[name]
    return hierarchy

def get_labels(path):
    if path:
        with open(path, "r") as f:
            labels = f.read().splitlines()

        return labels
    else:
        return ["O", "B-MISC", "I-MISC", "B-PER", "I-PER", "B-ORG", "I-ORG", "B-LOC", "I-LOC"]


def mask_check(matrix, fine, coarse):
    for i, col in enumerate(matrix.columns):
        assert col == coarse[i], 'coarse label error'

    for i, row in enumerate(matrix.index):
        assert row == fine[i], 'fine label error'


def make_mask_matrix(args, labels, name):
    """
    top_k mapping matrix
    :return: coarse_label_idx -> dict, {fine:coarse} -> dict, mask_matrix
    """
    num_labels = len(labels)  # fine

    if name in args.coarse_datasets:  # coarse dataset

        label_path = os.path.join(args.data_dir, name, args.model_type, args.model_name_or_path, f"{args.coarse_labels}_{args.fine_dataset}_top{args.mapping_top_k}.txt")

    else:  # fine, dev, test datasets
        label_path = os.path.join(args.data_dir, name, f"{args.labels}.txt")
    data_labels = get_labels(label_path)

    label2idx = {c: i for i, c in enumerate(data_labels)}  # coarse_index

    args.logger.info(
        f"{name} coarse label {len(data_labels)}: {data_labels}")

    # load mapping matrix if name is coarse dataset
    if name in args.coarse_datasets:
        matrix_path = os.path.join(args.data_dir, name, f"mapping_matrix_{args.model_type}", args.model_name_or_path, f"{args.fine_dataset}_top{args.mapping_top_k}.csv")
        matrix_df = pd.read_csv(matrix_path, index_col=0)
        mapping_matrix = matrix_df.reindex(labels)
        mapping_matrix = mapping_matrix[data_labels]
        mask_check(mapping_matrix, labels, data_labels)
        mapping_matrix = mapping_matrix.to_numpy()

    else: # if name in fine, dev, test datasets, make symmetric matrix
        mapping_matrix = np.zeros((num_labels, len(data_labels)))  # #fine, #coarse
        for i, label in enumerate(labels):  # fine
            mapping_matrix[i, label2idx[label]] = 1


    return label2idx, mapping_matrix


class SingleTaskDataset(Dataset):
    '''
        Dataset of single task
        input arguments:
            args: arguments
            dataset: dataset name (string)
            tokenizer: tokenizer
            labels: list of labels
            pad_token_label_id: pad token id
            coarse_dict: coarse label dict
            mode: train, dev, test
    '''

    def __init__(self, args, dataset, tokenizer, labels, pad_token_label_id, coarse_dict, mode):
        super().__init__()
        self._args = args
        self._dataset_name = dataset
        self._tokenizer = tokenizer
        self._labels = labels
        self._pad_token_label_id = pad_token_label_id
        self._coarse_dict = coarse_dict
        self._mode = mode

        self.data = self.make_dataset()

    def get_dataset_name(self):
        return self._dataset_name

    def make_dataset(self):
        data_dir = os.path.join(self._args.data_dir, self._dataset_name)
        data_form = open_json(
            f"{self._args.data_dir}/format", self._dataset_name)

        cached_feature_file = os.path.join(data_dir, "{}_{}_{}_{}".format(self._mode,
                                                                          list(
                                                                              filter(None,
                                                                                     self._args.model_name_or_path.split(
                                                                                         "/"))).pop(),
                                                                          str(self._args.max_seq_length),
                                                                          f"top{str(self._args.mapping_top_k)}"))

        if os.path.exists(cached_feature_file) and not self._args.overwrite_cache:
            self._args.logger.info(
                "Loading features from cached file %s", cached_feature_file)
            features = torch.load(cached_feature_file)
        else:
            self._args.logger.info(
                "Creating features from dataset file at %s", data_dir)

            reader = Reader()
            if self._dataset_name != "OntoNote":  # .txt
                examples = reader.read_txt(
                    data_dir, data_form, self._dataset_name, self._mode)
            else:
                exist_label = self._coarse_dict[self._dataset_name].keys()
                examples = reader.read_conll(data_dir, data_form, self._mode, exist_label)

            features = reader.convert_examples_to_features(examples, self._labels,
                                                           self._coarse_dict[self._dataset_name],
                                                           self._args.max_seq_length, self._tokenizer,
                                                           self._args.logger, self._dataset_name,

                                                           cls_token_at_end=bool(
                                                               self._args.model_type in ["xlnet"]),
                                                           # xlnet has a cls token at the end
                                                           cls_token=self._tokenizer.cls_token,
                                                           cls_token_segment_id=2 if self._args.model_type in [
                                                               "xlnet"] else 0,  # 0
                                                           sep_token=self._tokenizer.sep_token,
                                                           sep_token_extra=bool(
                                                               self._args.model_type in ["roberta"]),
                                                           # False
                                                           # roberta uses an extra separator b/w pairs of sentences, cf. github.com/pytorch/fairseq/commit/1684e166e3da03f5b600dbb7855cb98ddfcd0805
                                                           pad_on_left=bool(self._args.model_type in [
                                                               "xlnet"]),  # False
                                                           # pad on the left for xlnet
                                                           pad_token=self._tokenizer.convert_tokens_to_ids(
                                                               [self._tokenizer.pad_token])[0],
                                                           pad_token_segment_id=4 if self._args.model_type in [
                                                               "xlnet"] else 0,  # 0
                                                           pad_token_label_id=self._pad_token_label_id
                                                           )
            self._args.logger.info(
                "Saving features into cached file %s", cached_feature_file)
            torch.save(features, cached_feature_file)

        data = {
            'input_ids': [f.input_ids for f in features],
            'attention_mask': [f.attention_mask for f in features],
            'token_type_ids': [f.token_type_ids for f in features],
            'labels': [f.labels for f in features],
        }
        return data

    def __len__(self):
        return len(self.data['input_ids'])

    def __getitem__(self, idx):
        sample = {
            'input_ids': self.data['input_ids'][idx],
            'attention_mask': self.data['attention_mask'][idx],
            'token_type_ids': self.data['token_type_ids'][idx],
            'labels': self.data['labels'][idx],
        }
        return sample


class MultiTaskDataset(Dataset):
    """
        input arguments:
            datasets : dictionary of {key : dataset name, value : dataset}
    """

    def __init__(self, datasets):
        self._datasets = datasets
        self._dataset_name_2_dataset_dic = {}

        for dataset in self._datasets:
            dataset_name = dataset.get_dataset_name()
            self._dataset_name_2_dataset_dic[dataset_name] = dataset

    def get_dataset(self):
        return self._dataset_name_2_dataset_dic

    def __len__(self):
        return sum([len(dataset) for dataset in self._datasets])

    def __getitem__(self, idx):
        dataset_name, sample_idx = idx
        out = {
            'dataset_name': dataset_name,
            'sample': self._dataset_name_2_dataset_dic[dataset_name][sample_idx]
        }
        return out


class MultitaskBatcher(BatchSampler):
    def __init__(self, args, datasetObj, shuffleDataset=True, shuffleBatch=True, seed=42):

        """
            input arguments:
                args : literally args
                datasetObj : MultitaskDataset object
                shuffleDataset : shuffle dataset or not (bool)
                shuffleBatch : shuffle batch or not (bool)
                seed : random seed (int)
        """

        self._args = args
        self._multitaskDatasetData = datasetObj.get_dataset()

        self._batchSize = self._args.batch_size

        # to shuffle the indices 'in' a batch
        self._shuffleBatch = shuffleBatch
        # to shuffle the samples picked up among all the tasks -> 어떤식으로 작동하는지는 이해해야겠지?
        self._shuffleDataset = shuffleDataset
        self._seed = seed

        # self._multitaskDatasetDataBatchIdxs = []
        # self._batchDatasetNames = []

    def make_batches(self, dataSize):

        # make batch indices
        batchIndices = list(range(dataSize))
        if self._shuffleBatch:
            random.seed(self._seed)
            random.shuffle(batchIndices)

        batchIdxs = [batchIndices[i:i + self._batchSize] for i in range(0, len(batchIndices), self._batchSize)]

        return batchIdxs

    def make_task_idxs(self, multitaskDatasetDataBatchIdxs):
        '''
        This fn makes task indices for which a corresponding batch is created
        eg. [0, 0, 1, 3, 0, 2, 3, 1, 1, ..] if task ids are 0,1,2,3
        '''
        taskIdxs = []
        for i in range(len(multitaskDatasetDataBatchIdxs)):
            taskIdxs += [i] * len(multitaskDatasetDataBatchIdxs[i])
        if self._shuffleDataset:
            random.seed(self._seed)
            random.shuffle(taskIdxs)
        return taskIdxs

    # overriding BatchSampler functions to generate iterators for all tasks
    # and iterate
    def __len__(self):
        # length of the batch sampler is the sum of the lengths of all the tasks
        return sum([len(dataset) // self._batchSize + (1 if len(dataset) % self._batchSize != 0 else 0) for dataset in
                    self._multitaskDatasetData.values()])
        # return sum([len(batchIdxs) for batchIdxs in self._multitaskDatasetDataBatchIdxs])

    def __iter__(self):
        # make batch indices for each task
        batchDatasetNames = []
        multitaskDatasetDataBatchIdxs = []

        for datasetName, dataset in self._multitaskDatasetData.items():
            batchDatasetNames.append(datasetName)
            multitaskDatasetDataBatchIdxs.append(self.make_batches(len(dataset)))

        allDatasetIters = [iter(item)
                           for item in multitaskDatasetDataBatchIdxs]
        # all_iters = [iter(item) for item in self._train_data_list]
        allIdxs = self.make_task_idxs(multitaskDatasetDataBatchIdxs)
        for datasetIdx in allIdxs:
            # this batch belongs to a specific task id
            batchDatasetName = batchDatasetNames[datasetIdx]
            batch = next(allDatasetIters[datasetIdx])
            yield [(batchDatasetName, sampleIdx) for sampleIdx in batch]


class Collator:
    '''
    This class is supposed to perform function which will help complete the batch data
    when DataLoader creates batch using allTasksDataset and Batcher.
    Main function would be
    1. A function to make get the various components of input in batch samples and make them into
    Pytorch Tensors like token_id, type_ids, masks.
    2. Collater function :- This function will use the above function to convert the batch into
    pytorch tensor inputs. As converting all the data into pytorch tensors before might not be a good
    idea due to space, hence this custom function will be used to convert the batches into tensors on the fly
    by acting as custom collater function to DataLoader
    '''

    def __init__(self, args):
        self._args = args
        self.maxSeqLen = self._args.max_seq_length

        # self.dropout = dropout

    def check_samples_len(self, batch):
        # function to check whether all samples are having the maxSeqLen mentioned
        for idx in range(len(batch['input_ids'])):
            assert len(
                batch['input_ids'][idx]) == self.maxSeqLen, "input_ids length is not equal to maxSeqLen"
            assert len(batch['attention_mask'][idx]
                       ) == self.maxSeqLen, "attention_mask length is not equal to maxSeqLen"
            assert len(batch['token_type_ids'][idx]
                       ) == self.maxSeqLen, "token_type_ids length is not equal to maxSeqLen"
            assert len(
                batch['labels'][idx]) == self.maxSeqLen, "labels length is not equal to maxSeqLen"

    def add_padding(self, batch):
        # length of all the samples in the batch
        # sample has 'input_ids', 'attention_mask', 'token_type_ids', 'labels'
        lens = [len(sample['input_ids']) for sample in batch]
        max_len = max(lens)
        self.maxSeqLen = max_len

        batchData = {
            'input_ids': [],
            'attention_mask': [],
            'token_type_ids': [],
            'labels': []
        }
        for sample in batch:
            input_ids = sample['input_ids']
            attention_mask = sample['attention_mask']
            token_type_ids = sample['token_type_ids']
            labels = sample['labels']

            padding_len = max_len - len(input_ids)
            input_ids += [0] * padding_len
            attention_mask += [0] * padding_len
            token_type_ids += [0] * padding_len
            labels += [-100] * padding_len

            batchData['input_ids'].append(input_ids)
            batchData['attention_mask'].append(attention_mask)
            batchData['token_type_ids'].append(token_type_ids)
            batchData['labels'].append(labels)

        self.check_samples_len(batchData)
        return batchData

    def make_batch_to_input_tensor(self, batch):
        # add padding to the batch
        batchData = self.add_padding(batch)

        # meta deta will store more things like task id, task type etc.
        batchMetaData = {"input_ids": 0, "attention_mask": 1,
                         "token_type_ids": 2, "labels": 3}

        # convert to tensors
        for key in batchData.keys():
            batchData[key] = torch.tensor(
                batchData[key], dtype=torch.long).to(self._args.device)
        return batchMetaData, batchData

    def collate_fn(self, batch):
        '''
        This function will be used by DataLoader to return batches
        '''
        datasetName = batch[0]["dataset_name"]

        orgBatch = []
        label_ids = []
        for samp in batch:
            assert samp["dataset_name"] == datasetName
            orgBatch.append(samp["sample"])

        batch = orgBatch
        # making tensor batch data
        batchMetaData, batchData = self.make_batch_to_input_tensor(batch)
        batchMetaData['datasetName'] = datasetName
        return batchMetaData, batchData


def make_data_handlers(args, tokenizer, labels, pad_token_label_id, coarse_dict, mode):
    datasets = []
    if mode == "train":
        # Adding coarse-grained datasets
        if args.coarse_datasets:
            for coarseDatasetName in args.coarse_datasets:
                datasets.append(SingleTaskDataset(
                    args, coarseDatasetName, tokenizer, labels, pad_token_label_id, coarse_dict, mode))

        # Adding fine-grained dataset
        datasets.append(
            SingleTaskDataset(args, args.fine_dataset, tokenizer, labels, pad_token_label_id, coarse_dict, mode))

        dataset = MultiTaskDataset(datasets)
        batcher = MultitaskBatcher(args, dataset, shuffleDataset=False)
        loader = DataLoader(dataset, batch_sampler=batcher,
                            collate_fn=Collator(args).collate_fn)
    elif mode == "dev":
        for datasetName in args.eval_data:
            datasets.append(SingleTaskDataset(
                args, datasetName, tokenizer, labels, pad_token_label_id, coarse_dict, mode))
        dataset = MultiTaskDataset(datasets)
        batcher = MultitaskBatcher(
            args, dataset, shuffleDataset=False, shuffleBatch=False)
        loader = DataLoader(dataset, batch_sampler=batcher,
                            collate_fn=Collator(args).collate_fn)
    else:  # test
        for datasetName in args.test_data:
            datasets.append(SingleTaskDataset(
                args, datasetName, tokenizer, labels, pad_token_label_id, coarse_dict, mode))
        dataset = MultiTaskDataset(datasets)
        batcher = MultitaskBatcher(
            args, dataset, shuffleDataset=False, shuffleBatch=False)
        loader = DataLoader(dataset, batch_sampler=batcher,
                            collate_fn=Collator(args).collate_fn)

    return dataset, batcher, loader
