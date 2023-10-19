import os


class InputExample(object):
    """A single training/test example for token classification."""

    def __init__(self, guid, words, labels):
        """Constructs a InputExample.

        Args:
            guid: Unique id for the example.
            words: list. The words of the sequence.
            labels: (Optional) list. The labels for each word of the sequence. This should be
            specified for train and dev examples, but not for test examples.
        """
        self.guid = guid
        self.words = words
        self.labels = labels


class InputFeatures(object):
    """A single set of features of data."""

    def __init__(self, input_ids, attention_mask, token_type_ids, labels):
        self.input_ids = input_ids
        self.attention_mask = attention_mask
        self.token_type_ids = token_type_ids
        self.labels = labels


class Reader:
    def read_conll(self, data_dir, data_form, mode, exist_label):

        """
            [OntoNote 5.0]  conllx-format
            idx: splits[0]
            word: splits[1]
            POS: splits[3]
            head: int(splits[6])
            dep_label = splits[7]
            label: splits[10]
        """
        word_id = data_form['word']
        label_id = data_form['label']
        file_path = os.path.join(data_dir, "{}.sd.conllx".format(mode))
        guid_index = 1
        examples = []

        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or not line.strip():  # "" # 문서 시작 or 문장
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                     words=words,
                                                     labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    word = splits[word_id]
                    label = splits[label_id]

                    if label[2:] not in exist_label:  # if not in exist_label
                        label = 'O'

                    if word.strip():
                        words.append(word)
                        if len(splits) > 10:
                            labels.append(label.replace("\n", ""))
                        else:
                            # Examples could have no label for mode = "test"
                            labels.append("O")
            if words:
                examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                             words=words,
                                             labels=labels))

        return examples

    def read_txt(self, data_dir, data_form, dataset, mode):

        word_id = data_form['word']
        label_id = data_form['label']
        file_path = os.path.join(data_dir, "{}.txt".format(mode))
        guid_index = 1
        examples = []

        with open(file_path, encoding="utf-8") as f:
            words = []
            labels = []
            for line in f:
                if line.startswith("-DOCSTART-") or not line.strip():  # "" # 문서 시작 or 문장
                    if words:
                        examples.append(InputExample(guid="{}-{}".format(mode, guid_index),
                                                     words=words,
                                                     labels=labels))
                        guid_index += 1
                        words = []
                        labels = []
                else:
                    splits = line.split()
                    if splits[word_id].strip():
                        words.append(splits[word_id])

                        # # "inference"
                        # if mode == 'inference':
                        #     labels.append("O")

                        if (dataset == 'CoNLL' and len(splits) > 3) or (dataset != 'CoNLL' and len(splits) > 1):
                            labels.append(splits[label_id].replace("\n", ""))
                        else:
                            # Examples could have no label for mode = "inference"
                            labels.append("O")
            if words:
                examples.append(InputExample(guid="%s-%d".format(mode, guid_index),
                                             words=words,
                                             labels=labels))
        return examples

    def convert_examples_to_features(self,
                                     examples,
                                     labels,
                                     coarse_dict,
                                     max_seq_length,
                                     tokenizer,
                                     logger,
                                     name,  # dataset
                                     cls_token_at_end=False,
                                     cls_token="[CLS]",
                                     cls_token_segment_id=1,
                                     sep_token="[SEP]",
                                     sep_token_extra=False,
                                     pad_on_left=False,
                                     pad_token=0,
                                     pad_token_segment_id=0,
                                     pad_token_label_id=-1,
                                     sequence_a_segment_id=0,
                                     mask_padding_with_zero=True):

        label_dict = coarse_dict

        features = []

        for (ex_index, example) in enumerate(examples):
            if ex_index % 10000 == 0:
                logger.info("Writing example %d of %d", ex_index, len(examples))
            tokens = []
            labels = []

            for word, label in zip(example.words, example.labels):
                word_tokens = tokenizer.tokenize(word)

                # 'O' labeling
                if label.startswith('B-') or label.startswith('I-'):
                    label = label[2:]

                if word_tokens:
                    tokens.extend(word_tokens)
                    # Use the real label id for the first token of the word, and padding ids for the remaining tokens
                    # labels.extend([label_dict[label]] + [pad_token_label_id] * (len(word_tokens) - 1))
                    labels.extend([label_dict[label]] * len(word_tokens))

            # Account for [CLS] and [SEP] with "- 2" and with "- 3" for RoBERTa.
            special_tokens_count = 3 if sep_token_extra else 2
            if len(tokens) > max_seq_length - special_tokens_count:
                tokens = tokens[:(max_seq_length - special_tokens_count)]
                labels = labels[:(max_seq_length - special_tokens_count)]

            """
            # The convention in BERT is:
            # (a) For sequence pairs:
            #  tokens:   [CLS] is this jack ##son ##ville ? [SEP] no it is not . [SEP]
            #  type_ids:   0   0  0    0    0     0       0   0   1  1  1  1   1   1
            # (b) For single sequences:
            #  tokens:   [CLS] the dog is hairy . [SEP]
            #  type_ids:   0   0   0   0  0     0   0
            #
            # Where "type_ids" are used to indicate whether this is the first
            # sequence or the second sequence. The embedding vectors for `type=0` and
            # `type=1` were learned during pre-training and are added to the wordpiece
            # embedding vector (and position vector). This is not *strictly* necessary
            # since the [SEP] token unambiguously separates the sequences, but it makes
            # it easier for the model to learn the concept of sequences.
            #
            # For classification tasks, the first vector (corresponding to [CLS]) is
            # used as as the "sentence vector". Note that this only makes sense because
            # the entire model is fine-tuned.
            """
            tokens += [sep_token]
            labels += [pad_token_label_id]

            # roberta uses an extra separator b/w pairs of sentences
            if sep_token_extra:
                tokens += [sep_token]
                labels += [pad_token_label_id]

            token_type_ids = [sequence_a_segment_id] * len(tokens)

            if cls_token_at_end:
                tokens += [cls_token]
                labels += [pad_token_label_id]
                token_type_ids += [cls_token_segment_id]
            else:
                tokens = [cls_token] + tokens
                labels = [pad_token_label_id] + labels
                token_type_ids = [cls_token_segment_id] + token_type_ids

            input_ids = tokenizer.convert_tokens_to_ids(tokens)

            # The mask has 1 for real tokens and 0 for padding tokens. Only real
            # tokens are attended to.
            attention_mask = [1 if mask_padding_with_zero else 0] * len(input_ids)

            if ex_index < 3:
                logger.info("*** Example ***")
                logger.info("guid: %s", example.guid)
                logger.info("tokens: %s", " ".join([str(x) for x in tokens]))
                logger.info("input_ids: %s", " ".join([str(x) for x in input_ids]))
                logger.info("input_mask: %s", " ".join([str(x) for x in attention_mask]))
                logger.info("token_type_ids: %s", " ".join([str(x) for x in token_type_ids]))
                logger.info("labels: %s", " ".join([str(x) for x in labels]))

            features.append(
                InputFeatures(input_ids=input_ids,
                              attention_mask=attention_mask,
                              token_type_ids=token_type_ids,
                              labels=labels))
        return features
