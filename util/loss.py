import torch
from torch import nn
from torch.nn import Softmax, KLDivLoss, NLLLoss, CrossEntropyLoss, MSELoss


class TotalLoss:
    """
        __init__ function will initialize the loss function based on the loss_type
    """

    def __init__(self, loss_type, coarse_datasets=None):
        """
            input arguments :
                    loss_type: str
        """
        self._loss_type = loss_type
        self._coarse_datasets = coarse_datasets

        self.fine_loss_function = CrossEntropyLoss()
        
        if self._loss_type == 'Max':
            self.loss_function = MaxLogits()

        elif self._loss_type == "Mix":
            self.loss_function = MixLoss()

        elif self._loss_type == "CustomCELoss":  # Default
            self.loss_function = CustomCELoss()

        elif self._loss_type == "coarseFilter":
            self.loss_function = coarseFilter()

        elif self._loss_type == 'maxFilter':
            self.loss_function = maxFilter()

        elif self._loss_type == "Focal":
            self.loss_function = FocalLoss(gamma=2)

        elif self._loss_type == "focalFilter":
            self.loss_function = focalFilter(gamma=2)

        else:
            self.loss_function = OnlyCrossEntropy()

    def calculate(self, dataset_name, logits, labels, **kwargs):

        if dataset_name in self._coarse_datasets:
            loss = self.loss_function(logits, labels, **kwargs)
        else:  # train_fine, eval, test
            loss = self.fine_loss_function(logits.transpose(1, 2), labels)  # CELoss

        return loss


class LogSoftmaxF2C(nn.Module):  # logits -> coarse -> log_softmax
    def __init__(self):
        super(LogSoftmaxF2C, self).__init__()
        self.softmax = Softmax(dim=2)

    def forward(self, logits, mask_matrix):
        p_fine = self.softmax(logits)
        p_coarse = torch.matmul(p_fine, mask_matrix)
        logp_coarse = torch.log(p_coarse)
        return logp_coarse


class OnlyCrossEntropy(nn.Module):
    def __init__(self):
        super().__init__()
        self.loss_function = CrossEntropyLoss()

    def forward(self, logits, labels, **kwargs):
        coarse_logits = kwargs['coarse_logits'] if 'coarse_logits' in kwargs else None

        loss = self.loss_function(coarse_logits.transpose(1, 2), labels)
        return loss


class MaxLogits(nn.Module):
    def __init__(self):
        super().__init__()
        # self.nll_criterion = NLLLoss()
        self.ce_criterion = CrossEntropyLoss()

    @staticmethod
    def mapping_fine_indices(c_label, mask_matrix):
        fine_indices = mask_matrix[:, c_label].nonzero(as_tuple=True)[0].detach().to('cpu').numpy()
        return fine_indices

    def forward(self, logits, labels, **kwargs):
        mask_matrix = kwargs['mask_matrix'] if 'mask_matrix' in kwargs else None

        coarse_logits = []
        # total_max = torch.max(logits, 2).values

        for i in range(mask_matrix.size()[-1]):  # #coarse
            coarse_logits.append(
                # torch.max(logits[:, :, self.mapping_fine_indices(i, mask_matrix)], 2).values - total_max)
                torch.max(logits[:, :, self.mapping_fine_indices(i, mask_matrix)], 2).values)

        coarse_max_logits = torch.stack(coarse_logits, 2)
        # loss = self.nll_criterion(coarse_max_logits.transpose(1, 2), labels)
        loss = self.ce_criterion(coarse_max_logits.transpose(1, 2), labels)

        return loss


class maxFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=2)
        self.fine_ce_criterion = CrossEntropyLoss()
        self.ce_criterion = CrossEntropyLoss(reduction='none')

    def filtering(self, logits, labels, mask_matrix):
        p_fine = self.softmax(logits)
        p_f2c = torch.matmul(p_fine, mask_matrix)
        pred = torch.argmax(p_f2c, 2)

        mask_labels = labels == pred
        return mask_labels

    @staticmethod
    def mapping_fine_indices(c_label, mask_matrix):
        fine_indices = mask_matrix[:, c_label].nonzero(as_tuple=True)[0].detach().to('cpu').numpy()
        return fine_indices

    def forward(self, logits, labels, **kwargs):
        mask_matrix = kwargs['mask_matrix'] if 'mask_matrix' in kwargs else None
        mask_logits = kwargs['mask_logits'] if 'mask_logits' in kwargs else None

        coarse_logits = []
        mask_labels = self.filtering(mask_logits, labels, mask_matrix)

        for i in range(mask_matrix.size()[-1]):  # #coarse
            coarse_logits.append(
                torch.max(logits[:, :, self.mapping_fine_indices(i, mask_matrix)], 2).values)
        coarse_max_logits = torch.stack(coarse_logits, 2)

        losses = self.ce_criterion(coarse_max_logits.transpose(1, 2), labels)
        masked_loss = losses * mask_labels
        loss = torch.sum(masked_loss) / torch.count_nonzero(masked_loss)  # sum(mask_labels)

        return loss


class MixLoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.max_logits = MaxLogits()
        self.custom_celoss = CustomCELoss()

    def forward(self, logits, labels, **kwargs):
        loss = 0.5 * (self.max_logits(logits, labels, **kwargs) + self.custom_celoss(logits, labels, **kwargs))
        return loss


class CustomCELoss(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=2)
        self.nll_criterion = NLLLoss()

    def forward(self, logits, labels, **kwargs):
        mask_matrix = kwargs['mask_matrix'] if 'mask_matrix' in kwargs else None

        p_fine = self.softmax(logits)
        p_coarse = torch.matmul(p_fine, mask_matrix)
        logp_coarse = torch.log(p_coarse)

        loss = self.nll_criterion(logp_coarse.transpose(1, 2), labels)
        return loss


class coarseFilter(nn.Module):
    def __init__(self):
        super().__init__()
        self.softmax = Softmax(dim=2)
        self.nll_criterion = NLLLoss(reduction='none')
        self.log_sm = LogSoftmaxF2C()

    def filtering(self, logits, labels, mask_matrix):
        p_fine = self.softmax(logits)
        p_f2c = torch.matmul(p_fine, mask_matrix)
        pred = torch.argmax(p_f2c, 2)

        mask_labels = labels == pred
        return mask_labels

    def forward(self, logits, labels, **kwargs):
        mask_matrix = kwargs['mask_matrix'] if 'mask_matrix' in kwargs else None
        mask_logits = kwargs['mask_logits'] if 'mask_logits' in kwargs else None

        mask_labels = self.filtering(mask_logits, labels, mask_matrix)

        logp_coarse = self.log_sm(logits, mask_matrix)

        losses = self.nll_criterion(logp_coarse.transpose(1, 2), labels)
        masked_loss = losses * mask_labels
        loss = torch.sum(masked_loss) / torch.count_nonzero(masked_loss)  # sum(mask_labels)

        return loss


class FocalLoss(nn.Module):
    def __init__(self, gamma=2, reduction=True):
        super().__init__()
        self.gamma = gamma
        self.reduce = reduction

        self.log_sm = LogSoftmaxF2C()

    def forward(self, logits, labels, **kwargs):
        mask_matrix = kwargs['mask_matrix'] if 'mask_matrix' in kwargs else None


        logp_coarse = self.log_sm(logits, mask_matrix)
        ce_loss = NLLLoss(reduction='none')(logp_coarse.transpose(1, 2), labels)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if self.reduce:
            return torch.sum(focal_loss) / torch.count_nonzero(focal_loss)  # pad_token_id 제외
        else:
            return focal_loss


class focalFilter(nn.Module):
    def __init__(self, gamma=2):
        super().__init__()
        self.gamma = gamma
        self.softmax = Softmax(2)

        self.log_sm = LogSoftmaxF2C()

    def filtering(self, logits, labels, mask_matrix):
        p_fine = self.softmax(logits)
        p_f2c = torch.matmul(p_fine, mask_matrix)
        pred = torch.argmax(p_f2c, 2)

        mask_labels = labels == pred
        return mask_labels

    def forward(self, logits, labels, **kwargs):
        mask_matrix = kwargs['mask_matrix'] if 'mask_matrix' in kwargs else None
        mask_logits = kwargs['mask_logits'] if 'mask_logits' in kwargs else None


        logp_coarse = self.log_sm(logits, mask_matrix)
        ce_loss = NLLLoss(reduction='none')(logp_coarse.transpose(1, 2), labels)

        pt = torch.exp(-ce_loss)
        focal_loss = (1 - pt) ** self.gamma * ce_loss

        if mask_logits is not None:
            mask_labels = self.filtering(mask_logits, labels, mask_matrix)
            focal_loss = focal_loss * mask_labels

        return torch.sum(focal_loss) / torch.count_nonzero(focal_loss)