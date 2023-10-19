import torch
from torch import nn
from transformers import BertPreTrainedModel, BertModel, RobertaModel
from torch.nn import CrossEntropyLoss


class BERTWithDualClassification(BertPreTrainedModel):

    def __init__(self, config, coarse_dict):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.bert = BertModel.from_pretrained("bert-base-uncased", config=config, add_pooling_layer=False)

        # self.model = RobertaModel.from_pretrained("roberta-base")
        # self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # fine labels

        self.classifiers = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, len(coarse_dict[key])) for key in coarse_dict})

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(
            self,
            input_ids,
            name=None,
            attention_mask=None,
            token_type_ids=None,
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            labels=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None):

        outputs = self.bert(
            input_ids,
            attention_mask=attention_mask,
            token_type_ids=token_type_ids,
            position_ids=position_ids,
            head_mask=head_mask,
            inputs_embeds=inputs_embeds,
            output_attentions=output_attentions,
            output_hidden_states=output_hidden_states,
            return_dict=return_dict,
        )

        sequence_output = outputs[0]

        sequence_output = self.dropout(sequence_output)

        if name != "Few-NERD":  # not fine
            logits = self.classifiers['Few-NERD'](sequence_output)
            coarse_logits = self.classifiers[name](sequence_output)
        else:  # Few-NERD
            logits = self.classifiers[name](sequence_output)
            coarse_logits = None

        loss = None
        if labels is not None:
            loss_fct = CrossEntropyLoss()
            loss = loss_fct(logits.view(-1, self.num_labels), labels.view(-1))

        # if not return_dict:
        #     output = (logits,) + outputs[2:]
        #     return ((loss,) + output) if loss is not None else output

        return loss, logits, coarse_logits


class MultiTaskModel(BertPreTrainedModel):

    def __init__(self, config, coarse_dict):
        super().__init__(config)

        self.num_labels = config.num_labels

        self.encoder = BertModel.from_pretrained("bert-base-uncased", config=config, add_pooling_layer=False)

        # self.model = RobertaModel.from_pretrained("roberta-base")
        # self.bert = BertModel(config, add_pooling_layer=False)

        classifier_dropout = (
            config.classifier_dropout if config.classifier_dropout is not None else config.hidden_dropout_prob
        )
        self.dropout = nn.Dropout(classifier_dropout)

        self.relu = nn.ReLU()
        self.tanh = nn.Tanh()

        # self.classifier = nn.Linear(config.hidden_size, config.num_labels)  # fine labels

        self.classifiers = nn.ModuleDict(
            {key: nn.Linear(config.hidden_size, len(coarse_dict[key])) for key in coarse_dict})

        # Initialize weights and apply final processing
        # self.post_init()

    def forward(self, batchMetaData, batchData):

        outputs = self.encoder(
            input_ids=batchData['input_ids'],
            attention_mask=batchData['attention_mask'],
            token_type_ids=batchData['token_type_ids'],
            position_ids=None,
            head_mask=None,
            inputs_embeds=None,
            output_attentions=None,
            output_hidden_states=None,
            return_dict=None,
        )

        sequence_output = outputs[0]
        sequence_output = self.dropout(sequence_output)

        if batchMetaData['datasetName'].split('_')[0] != "Few-NERD":  # not fine
            logits = self.classifiers['Few-NERD'](sequence_output)
            coarse_logits = self.classifiers[batchMetaData['datasetName']](sequence_output)
        else:  # Few-NERD
            logits = self.classifiers[batchMetaData['datasetName']](sequence_output)
            coarse_logits = None

        loss = None

        return loss, logits, coarse_logits
