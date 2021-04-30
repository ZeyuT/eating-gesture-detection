import torch
import torchvision
import torch.nn as nn
from constants import LABEL_NUM

class TimeDistributed(nn.Module):
    def __init__(self, module, batch_first=True):
        super(TimeDistributed, self).__init__()
        self.module = module
        self.batch_first = batch_first
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        input = x.view(batch_size * time_steps, C, H, W)
        output = self.module(input)
        output = output.view(batch_size, time_steps, -1)
        if self.batch_first is False:
            output = output.permute(1, 0, 2)
        return output

    
class Spatial_Encoder(nn.Module):
    def __init__(self,basemodel='resnet34'):
        super(Spatial_Encoder, self).__init__()
        self._prepare_basemodel(basemodel)
    def forward(self, x):
        batch_size, time_steps, C, H, W = x.size()
        x = x.view(batch_size * time_steps, C, H, W)
        x = self.net(x)
        new_C, new_H, new_W = x.size()[-3:]
        output = x.view(batch_size, time_steps, new_C, new_H, new_W)
        return output
    def _prepare_basemodel(self,basemodel):
        if basemodel == "resnet34":
            model = torchvision.models.resnet34(pretrained=True)
        if basemodel == "resnet50":
            model = torchvision.models.resnet50(pretrained=True)
        module_list = list(model.children())
        del module_list[-1]
        self.net = nn.Sequential(*module_list)
        
class RES_LSTM(nn.Module):
    def __init__(self,seq_len=16,basemodel='resnet34'):
        super(RES_LSTM, self).__init__()
        self.encoder = Spatial_Encoder(basemodel)
        if basemodel=='resnet34':
            encoder_size = 512
        if basemodel=='resnet50':
            encoder_size = 2048
        self.lstm = nn.LSTM(input_size=encoder_size,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        self.batch_norm = nn.BatchNorm1d(num_features=seq_len)
        self.flatten = nn.Flatten(start_dim=2,end_dim=-1)
        self.fc = nn.Sequential(nn.Linear(128, LABEL_NUM),
                                nn.ReLU())
        self.act = nn.Softmax(dim=-1)
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.batch_norm(x)
        self.lstm.flatten_parameters() 
        x,(hn, cn) = self.lstm(x)
        x = self.fc(x)
        output = self.act(x)
        return output