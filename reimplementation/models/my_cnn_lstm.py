import torch
import torchvision
import torch.nn as nn

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
        if basemodel == "resnet101":
            model = torchvision.models.resnet101(pretrained=True)
        module_list = list(model.children())
        del module_list[-1]
        self.net = nn.Sequential(*module_list)

class CNN_LSTM(nn.Module):
    def __init__(self,seq_len=16,num_classes=3,basemodel='resnet50'):
        super(CNN_LSTM, self).__init__()
        self.encoder = Spatial_Encoder(basemodel)
        if basemodel=='resnet34':
            encoder_size = 512
        if basemodel=='resnet50' or basemodel == "resnet101":
            encoder_size = 2048
        self.lstm = nn.LSTM(input_size=encoder_size,
                            hidden_size=128,
                            num_layers=2,
                            batch_first=True)
        for name, param in self.lstm.named_parameters():
            if 'bias' in name:
                 nn.init.constant_(param, 0.0)
            elif 'weight_ih' in name:
                 nn.init.kaiming_normal_(param)
            elif 'weight_hh' in name:
                 nn.init.orthogonal_(param)
                 
        self.batch_norm = nn.BatchNorm1d(num_features=seq_len)
        self.flatten = nn.Flatten(start_dim=2,end_dim=-1)
        self.fc = nn.Sequential(nn.Linear(128, num_classes),
                                nn.ReLU())                                
        for name, param in self.fc.named_parameters():
            if 'weight' in name:
                nn.init.kaiming_normal_(param)
            elif 'bias' in name:
                nn.init.constant_(param, 0.0)   
        self.act = nn.Softmax(dim=-1)
        
    def forward(self, x):
        x = self.encoder(x)
        x = self.flatten(x)
        x = self.batch_norm(x)
        self.lstm.flatten_parameters()
        x,_ = self.lstm(x)
        x = self.fc(x)
        output = self.act(x)
        return output
            
def generate_model(seq_len=16, network='lstm-r34', num_classes=3):
    if network=='my-lstm-r34':
        resnet_version = 'resnet34'
    elif network=='my-lstm-r50':
        resnet_version = 'resnet50'
    elif network=='my-lstm-r101':
        resnet_version = 'resnet101'        
    model = CNN_LSTM(seq_len, num_classes, resnet_version)
    return model