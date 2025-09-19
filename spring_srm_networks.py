
import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

# RNN part for outputing preferences (blue blocks on figure 2).
class Decoder_RNN(nn.Module):
    def __init__(self, x_size, h_size, out_size, dr = 0.2, num_rnn_layers = 3):
        super(Decoder_RNN, self).__init__()
        self.lstm = nn.GRU(x_size, h_size, num_layers = num_rnn_layers, dropout = dr)
        self.fc = torch.nn.Sequential(
            nn.Linear(h_size * 2, 512),
            nn.LeakyReLU(0.1),
            nn.Dropout(dr),
            nn.Linear(512, out_size)
        )

    def forward(self, x, h, z, force_sm = False):
        o, h = self.lstm(x.to(torch.float), h)
        o = F.leaky_relu(o)
        o = self.fc(torch.cat([o, z], dim = -1))
        if self.training and not force_sm:
            o = F.log_softmax(o.squeeze()).unsqueeze(0)
        else:
            o = F.softmax(o.squeeze()).unsqueeze(0)
        return o, h





# Var encoder part (red blocks on figure 2).
class Sub_Encoder(nn.Module):
    def __init__(self, in_size, z_size):
        super(Sub_Encoder, self).__init__()
        self.fc = nn.Linear(in_features = in_size, out_features = z_size)

    def forward(self, x):
        x = F.relu(self.fc(x))
        return x



# Scene encoder.
class Super_Encoder(nn.Module):
    def __init__(self, z_size, droprate = 0.2):
        super(Super_Encoder, self).__init__()
        self.fe = torch.nn.Sequential(*(list(models.resnet18(weights=models.ResNet18_Weights.IMAGENET1K_V1).children())[:-1]))
        self.append_dropout(self.fe, droprate)
        self.fc = nn.Linear(in_features = 512, out_features = z_size)

    def append_dropout(self, model, rate = 0.2):
        for name, module in model.named_children():
            if len(list(module.children())) > 0:
                self.append_dropout(module)
            if isinstance(module, nn.ReLU):
                new_model = nn.Sequential(module, nn.Dropout2d(p=rate, inplace=False))
                setattr(model, name, new_model)

    def forward(self, img):
        x = torch.flatten(self.fe(img), 1)
        x = F.relu(self.fc(x))
        return x
