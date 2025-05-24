import torch
import torch.nn as nn
import torch.nn.init as init
import torch.nn.functional as F
import os
import sys

sys.path.append(os.path.dirname(os.path.abspath(__file__)))
from Transfer_Classification.cla_network.ResNet import generate_model, generate_model_NoMaxPool, generate_model_SkipConv


class Classify_Normal(nn.Module):
    def __init__(self, configs):
        super(Classify_Normal, self).__init__()

        self.encoder_net = 'Conv'

        self.image_encoder = generate_model(configs.image_model_depth, **configs.image_encoder_params)

        bottom_channel = configs.model['bottom_channel']
        n_class = configs.model['n_class']
        self.avgpool = nn.AdaptiveAvgPool3d(1)

        assert configs.model['hidden_layer'] is None or configs.model['hidden_layer'] in ['None', 'Linear', 'Conv']

        if configs.model['hidden_layer'] == 'None' or configs.model['hidden_layer'] is None:
            self.has_hidden = False
            self.class_head = nn.Linear(bottom_channel, n_class)
        else:
            self.has_hidden = True
            assert configs.model['hidden_layer'] == 'Linear' or configs.model['hidden_layer'] == 'Conv'
            if configs.model['hidden_layer'] == 'Linear':
                self.hidden_layer = nn.Linear(bottom_channel, bottom_channel)
            else:
                self.hidden_layer = nn.Conv1d(bottom_channel, bottom_channel, kernel_size=1, stride=1, padding=0,
                                              bias=False)
            self.relu = nn.ReLU()
            self.class_head = nn.Linear(bottom_channel, n_class)
        if 'initialize' in configs.model:
            assert configs.model['initialize'] == True
            self._initialize_weights()

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                init.kaiming_normal_(m.weight.data, mode='fan_in', nonlinearity='relu')
                if m.bias is not None:
                    init.constant_(m.bias.data, 0)

    def forward(self, x):
        x_enc = self.image_encoder(x)

        b, c, h, w, d = x_enc.shape
        x_enc = self.avgpool(x_enc)
        x_enc = x_enc.reshape(b, c)

        if self.has_hidden:
            x_enc = self.hidden_layer(x_enc)
            x_enc = self.relu(x_enc)

        x_enc = self.class_head(x_enc)
        return x_enc


if __name__=="__main__":
    pass


