import torch
import torch.nn as nn
import torch.nn.functional as F

from torch_geometric.nn.conv import GatedGraphConv

torch.manual_seed(2020)


def get_conv_mp_out_size(in_size, last_layer, mps):
    size = in_size

    for mp in mps:
        size = round((size - mp["kernel_size"]) / mp["stride"] + 1)

    size = size + 1 if size % 2 != 0 else size

    return int(size * last_layer["out_channels"])


def init_weights(m):
    if type(m) == nn.Linear or type(m) == nn.Conv1d:
        torch.nn.init.xavier_uniform_(m.weight)


class Conv(nn.Module):

    def __init__(self, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size):
        super(Conv, self).__init__()
        self.conv1d_1_args = conv1d_1
        self.conv1d_1 = nn.Conv1d(**conv1d_1)
        self.conv1d_2 = nn.Conv1d(**conv1d_2)
        #print(1, conv1d_1, conv1d_2, maxpool1d_1, maxpool1d_2, fc_1_size, fc_2_size)
        #fc1_size = get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        fc1_size = 1500
        fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        self.fc1_size = fc1_size #get_conv_mp_out_size(fc_1_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        self.fc2_size = get_conv_mp_out_size(fc_2_size, conv1d_2, [maxpool1d_1, maxpool1d_2])
        # Dense layers
        print('new fc1', fc1_size, 'new fc2 size', fc2_size)
        self.fc1 = nn.Linear(fc1_size, 1)
        self.fc2 = nn.Linear(fc2_size, 1)

        # Dropout
        self.drop = nn.Dropout(p=0.2)

        self.mp_1 = nn.MaxPool1d(**maxpool1d_1)
        self.mp_2 = nn.MaxPool1d(**maxpool1d_2)

    def forward(self, hidden, x):
        #print('\n\nConv Class, forward method')
        #print('1 hidden', hidden.size())
        #print('2 x size', x.size())
        concat = torch.cat([hidden, x], 1)
        #print('2b concat hidden and x', concat.size())
        concat_size = hidden.shape[1] + x.shape[1]
        concat = concat.view(-1, self.conv1d_1_args["in_channels"], concat_size)
        #print('3 concat size', concat.size())
        
        Z = self.mp_1(F.relu(self.conv1d_1(concat)))
        Z = self.mp_2(self.conv1d_2(Z))
        #print('4 Z', Z.shape)

        hidden = hidden.view(-1, self.conv1d_1_args["in_channels"], hidden.shape[1])

        Y = self.mp_1(F.relu(self.conv1d_1(hidden)))
        Y = self.mp_2(self.conv1d_2(Y))
        #print('5 Y', Y.shape)
        
        #print('6 Z', Z.shape[-1], 'Y', Y.shape[-1])
        
        Z_flatten_size = int(Z.shape[1] * Z.shape[-1])
        Y_flatten_size = int(Y.shape[1] * Y.shape[-1])
        if Z_flatten_size != self.fc1_size:
            print('Dimensions are incompatble, change fc1 size')
        elif Y_flatten_size != self.fc2_size:
            print('Dimensions are incompatble, change fc2 size')
        #print('7 Z flatten', Z_flatten_size, 'Y flatten', Y_flatten_size)

        Z = Z.view(-1, Z_flatten_size)
        Y = Y.view(-1, Y_flatten_size)
        res = self.fc1(Z) * self.fc2(Y)
        res = self.drop(res)
        # res = res.mean(1)
        # print(res, mean)
        sig = torch.sigmoid(torch.flatten(res))
        return sig


class Net(nn.Module):

    def __init__(self, gated_graph_conv_args, conv_args, emb_size, device):
        super(Net, self).__init__()
        self.ggc = GatedGraphConv(**gated_graph_conv_args).to(device)
        self.conv = Conv(**conv_args,
                         fc_1_size=gated_graph_conv_args["out_channels"] + emb_size,
                         fc_2_size=gated_graph_conv_args["out_channels"]).to(device)
        # self.conv.apply(init_weights)

    def forward(self, data):
        #print('\n\nNET Class')
        x, edge_index = data.x, data.edge_index
        #print('batch size', x.size())
        x = self.ggc(x, edge_index)
        #print('ggc output size', x.size())
       # print('conv data.x', data.x.size())
        x = self.conv(x, data.x)

        return x

    def save(self, path):
        torch.save(self.state_dict(), path)

    def load(self, path):
        self.load_state_dict(torch.load(path))
