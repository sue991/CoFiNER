from tensorboardX import SummaryWriter
import torch
import os
import shutil

class TensorboardPlotter(object):
    def __init__(self, log_dir):
        self.writer = SummaryWriter(log_dir=log_dir)  # log가 저장될 경로

    def clear_dir(self, log_dir):
        flag = False
        if os.path.exists(log_dir):
            flag = True
            shutil.rmtree(log_dir)
        return flag

    def scalar_plot(self, group, tag, scalar_value, global_step):
        self.writer.add_scalar(f'{group}/{tag}', torch.tensor(scalar_value), global_step)

    def dict_plot(self, tag, tag_scalar_dict, global_step):
        self.writer.add_scalars(tag, tag_scalar_dict, global_step)

    # def histogram_plot(self, tag, values, global_step):
    #     self.writer.add_histogram(tag, values.grad.data.cpu().numpy(), global_step)

    #
    def close(self):
        self.writer.close()