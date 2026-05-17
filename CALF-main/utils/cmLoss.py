import torch
import torch.nn.functional as F
import torch.nn as nn
from .similar_utils import *
from copy import deepcopy

from .losses import mape_loss, mase_loss, smape_loss

loss_dict = {
    "l1": nn.L1Loss(),
    "smooth_l1": nn.SmoothL1Loss(),
    "ce": nn.CrossEntropyLoss(),
    "mse": nn.MSELoss(),
    "smape": smape_loss(),
    "mape": mape_loss(),
    "mase": mase_loss(),
}


class cmLoss(nn.Module):
    def __init__(self, feature_loss, output_loss, task_loss, task_name, feature_w=0.01, output_w=1.0, task_w=1.0):
        super(cmLoss, self).__init__()
        self.task_w = task_w
        self.output_w = output_w
        self.feature_w = feature_w

        self.feature_loss = loss_dict[feature_loss]
        self.output_loss = loss_dict[output_loss]
        self.task_loss = loss_dict[task_loss]
        
        self.task_name = task_name

    def forward(self, outputs, batch_y, in_sample=None, freq_map=None, batch_y_mark=None):
        outputs_text, outputs_time, intermidiate_feat_time, intermidiate_feat_text = (
            outputs["outputs_text"],
            outputs["outputs_time"],
            outputs["intermidiate_time"],
            outputs["intermidiate_text"],
        )
        
        # feture regularization loss
        feature_loss = sum(
            [
                (0.8**idx) * self.feature_loss(feat_time, feat_text)
                for idx, (feat_time, feat_text) in enumerate(
                    zip(intermidiate_feat_time[::-1], intermidiate_feat_text[::-1])
                )
            ]
        )

        # output consistency loss
        if self.task_name == "long_term_forecast":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "short_term_forecast":
            output_loss = self.output_loss(in_sample, freq_map, outputs_time, outputs_text, batch_y_mark)
        elif self.task_name == "classification":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "imputation":
            output_loss = self.output_loss(outputs_time, outputs_text)
        elif self.task_name == "anomaly_detection":
            output_loss = self.output_loss(outputs_time, outputs_text)
            

        batch_y = batch_y.to(output_loss.device)
        
        # supervised task loss — 两个分支都算，取平均
        if self.task_name == "long_term_forecast":
            task_loss_time = self.task_loss(outputs_time, batch_y)
            task_loss_text = self.task_loss(outputs_text, batch_y)
            task_loss = (task_loss_time + task_loss_text) / 2.0
        elif self.task_name == "short_term_forecast":
            task_loss = self.task_loss(in_sample, freq_map, outputs_time, batch_y, batch_y_mark)
        elif self.task_name == "classification":
            task_loss_time = self.task_loss(outputs_time, batch_y)
            task_loss_text = self.task_loss(outputs_text, batch_y)
            task_loss = (task_loss_time + task_loss_text) / 2.0
        elif self.task_name == "imputation":
            task_loss_time = self.task_loss(outputs_time, batch_y)
            task_loss_text = self.task_loss(outputs_text, batch_y)
            task_loss = (task_loss_time + task_loss_text) / 2.0
        elif self.task_name == "anomaly_detection":
            task_loss_time = self.task_loss(outputs_time, batch_y)
            task_loss_text = self.task_loss(outputs_text, batch_y)
            task_loss = (task_loss_time + task_loss_text) / 2.0

        total_loss = self.task_w * task_loss + self.output_w * output_loss + self.feature_w * feature_loss
        return {
            'total_loss': total_loss,
            'task_loss': task_loss.detach(),
            'output_loss': output_loss.detach(),
            'feature_loss': feature_loss.detach(),
        }
