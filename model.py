import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.nn import MultiheadAttention
import torch.optim as optim
import numpy as np
from scipy.stats import truncnorm

LEARNING_RATE = 1e-3

def hidden_init(layer):
    fan_in = layer.weight.data.size()[0]
    lim = 1./np.sqrt(fan_in)
    return(-lim,lim)
def truncated_normal(size, threshold=1):
    values = truncnorm.rvs(-threshold, threshold, size=size)
    return values
class Net(nn.Module):
    def __init__(self,in_shape,out_shape = 1,fc1_units = 256,fc2_units = 128,fc3_units = 16): #256-128-16 dropout0.5
        super(Net, self).__init__()
        self.fc1 = nn.Linear(in_shape, fc1_units)
        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.fc3 = nn.Linear(fc2_units, fc3_units)
        self.fc4 = nn.Linear(fc3_units,out_shape)
        # self.bn1 = nn.BatchNorm1d(fc1_units)
        # self.bn2 = nn.BatchNorm1d(fc2_units)
        # self.bn3 = nn.BatchNorm1d(fc3_units)
        self.reset_parameters()
        self.dropout = nn.Dropout(p=0.5)
        # add dropout?
    def forward(self,x):
        # x = F.relu(self.fc1(x))
        # x = F.relu(self.fc2(x))
        # x = F.relu(self.fc3(x))
        x = self.dropout(F.relu(self.fc1(x)))
        x = self.dropout(F.relu(self.fc2(x)))
        x = self.dropout(F.relu(self.fc3(x)))
        x = self.fc4(x)
        # self.reset_parameters()
        # return x
        return x.double()
    def reset_parameters(self):
        self.fc1.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc1.in_features))
        self.fc2.weight.data.normal_(mean=0, std=1 / np.sqrt(self.fc2.in_features))
        self.fc3.weight.data.normal_(mean=0, std=1 / np.sqrt(self.fc3.in_features))
        self.fc4.weight.data.normal_(mean=0, std=1 / np.sqrt(self.fc3.in_features))
        # self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(-3e-3,3e-3)
    def print_weight(self):
        print(self.fc1.weight)
class Model_v2(nn.Module):
    def __init__(self, in_shape, out_shape=1, hidden=[512,256,128,64,16], dropout = 0.1):  # 256-128-16 dropout0.5
        super(Model_v2, self).__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        ### Initilize Hidden unit
        fc1_units, fc2_units, fc3_units, fc4_units, fc5_units = hidden

        ### Hidden laters
        self.fc1 = nn.Linear(in_shape,fc1_units)
        self.bn1 = nn.BatchNorm1d(fc1_units)

        self.fc2 = nn.Linear(fc1_units,fc2_units)
        self.bn2 = nn.BatchNorm1d(fc2_units)

        self.fc3 = nn.Linear(fc2_units,fc3_units)
        self.bn3 = nn.BatchNorm1d(fc3_units)

        self.fc4 = nn.Linear(fc3_units,fc4_units)
        self.bn4 = nn.BatchNorm1d(fc4_units)

        self.fc5 = nn.Linear(fc4_units,fc5_units)
        self.bn5 = nn.BatchNorm1d(fc5_units)

        self.fc6 = nn.Linear(fc5_units,out_shape)

        ### Others
        self.reset_parameters()
        self.dropout = nn.Dropout(p=dropout)
        # add dropout?
    def forward(self,x):
        # x = F.leaky_relu(self.fc1(x))
        # x = F.leaky_relu(self.fc2(x))
        # x = F.leaky_relu(self.fc3(x))
        x = F.leaky_relu(self.bn1(self.dropout(self.fc1(x))))
        x = F.leaky_relu(self.bn2(self.dropout(self.fc2(x))))
        x = F.leaky_relu(self.bn3(self.dropout(self.fc3(x))))
        x = F.leaky_relu(self.bn4(self.dropout(self.fc4(x))))
        x = F.leaky_relu(self.bn5(self.dropout(self.fc5(x))))
        # x = self.dropout(F.leaky_relu(self.fc1(x)))
        # x = self.dropout(F.leaky_relu(self.fc2(x)))
        # x = self.dropout(F.leaky_relu(self.fc3(x)))
        x = self.fc6(x)

        return x.double()
    def reset_parameters(self):
        self.fc1.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc1.in_features))
        self.fc2.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc2.in_features))
        self.fc3.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc3.in_features))
        self.fc4.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc4.in_features))
        self.fc5.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc5.in_features))
        self.fc6.weight.data.normal_(mean=0, std=1/np.sqrt(self.fc6.in_features))
        # self.fc1.weight.data.uniform_(*hidden_init(self.fc1))
        # self.fc2.weight.data.uniform_(*hidden_init(self.fc2))
        # self.fc3.weight.data.uniform_(-3e-3,3e-3)
    def print_weight(self):
        print(self.fc1.weight)
        print(self.fc2.weight)
        print(self.fc3.weight)
        print(self.fc4.weight)
        print(self.fc5.weight)
        print(self.fc6.weight)
class Model_210805(nn.Module):
    '''
    With skip connection
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[512,256,128,64,16],dropout=0.5):  # 256-128-16 dropout0.5
        super(Model_210805, self).__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        ### Parameters
        self.dropout_rate = dropout

        ### Hidden laters
        # self.layers = self._make_layer(in_shape, out_shape,hidden)

        self.layers = self._make_layer(in_shape, hidden[2],hidden[:2])
        self.layers2 = self._make_layer(in_shape + hidden[2],out_shape,hidden[3:])

        self.activation = nn.Tanh()

        ### Others
        self.reset_parameters()

    def forward(self,x):
        h = self.layers(x)
        h = torch.cat((x, h), 1) #skip connection
        h = self.layers2(h)
        h = self.activation(h)

        # # Output is going to be between -1 and 7
        # h = h * 4
        # h = h + 3
        # Should I clip x to range(-1,7) so the algo will be stabilized?
        return h.double()

    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def _make_skip_layer(self,x, in_shape, out_shape, hidden =[10]):
        layers = []
        x_tmp = x
        in_layer = in_shape

        # Add hidden layers
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2ecf15f8a7f87fa56e784e0504136e9daf6b93d6/models/networks.py#L259
        for hidden_layer in hidden:
            x = nn.Linear(in_layer,hidden_layer)(x)
            in_layer = hidden_layer

        x = torch.cat((x,x_tmp), 1)
        x = nn.BatchNorm1d(hidden_layer)(x)
        x = nn.LeakyReLU(x)

        x = nn.Linear(in_shape + hidden_layer, out_shape)(x)
        x = nn.BatchNorm1d(hidden_layer)(x)
        x = nn.LeakyReLU(x)

        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)

    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)
class Model_fullyconnected_211218(nn.Module):
    '''
    Simple fully connected neural network 
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[256],dropout=0.5):  # 256-128-16 dropout0.5
        
        super(Model_fullyconnected_211218, self).__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        ### Parameters
        self.dropout_rate = dropout

        ### Hidden laters
        self.layers = self._make_layer(in_shape, out_shape,hidden)
        self.activation = nn.Tanh()

        ### Others
        self.reset_parameters()

    def forward(self,x):
        h = self.layers(x)
        h = self.activation(h)

        return h.double()

    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)

    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)
class Model_skip_211229(nn.Module):
    '''
    With skip connection
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[2000,2000,2000,1000,100],dropout=0.5): 
        '''
        in_shape: input shape
        out_shape: output shape
        hidden: list of hidden layers
        '''
        
        super(Model_skip_211229, self).__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        ### Parameters
        self.dropout_rate = dropout

        ### Hidden laters
        # self.layers = self._make_layer(in_shape, out_shape,hidden)

        self.layers = self._make_layer(in_shape, hidden[2],hidden[:2])
        self.layers2 = self._make_layer(in_shape + hidden[2],out_shape,hidden[2:])

        self.activation = nn.Tanh()

        ### Others
        self.reset_parameters()

    def forward(self,x):
        h = self.layers(x)
        h = torch.cat((x, h), 1) #skip connection
        h = self.layers2(h)
        # h = self.activation(h)

        # # Output is going to be between -1 and 7
        # h = h * 4
        # h = h + 3
        # Should I clip x to range(-1,7) so the algo will be stabilized?
        return h.double()

    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def _make_skip_layer(self,x, in_shape, out_shape, hidden =[10]):
        layers = []
        x_tmp = x
        in_layer = in_shape

        # Add hidden layers
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2ecf15f8a7f87fa56e784e0504136e9daf6b93d6/models/networks.py#L259
        for hidden_layer in hidden:
            x = nn.Linear(in_layer,hidden_layer)(x)
            in_layer = hidden_layer

        x = torch.cat((x,x_tmp), 1)
        x = nn.BatchNorm1d(hidden_layer)(x)
        x = nn.LeakyReLU(x)

        x = nn.Linear(in_shape + hidden_layer, out_shape)(x)
        x = nn.BatchNorm1d(hidden_layer)(x)
        x = nn.LeakyReLU(x)

        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)

    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)
class Model_skip_211229_toy(nn.Module):
    '''
    With skip connection
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[2000,2000,2000,1000,100],dropout=0.5):  # 256-128-16 dropout0.5
        super(Model_skip_211229_toy, self).__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        ### Parameters
        self.dropout_rate = dropout

        ### Hidden laters
        # self.layers = self._make_layer(in_shape, out_shape,hidden)

        self.layers = self._make_layer(in_shape, hidden[2],hidden[:2])
        self.layers2 = self._make_layer(in_shape + hidden[2],out_shape,hidden[2:])

        self.activation = nn.Tanh()

        ### Others
        self.reset_parameters()

    def forward(self,x):
        h = self.layers(x)
        h = torch.cat((x, h), 1) #skip connection
        h = self.layers2(h)
        # h = self.activation(h)

        # # Output is going to be between -1 and 7
        # h = h * 4
        # h = h + 3
        # Should I clip x to range(-1,7) so the algo will be stabilized?
        return h.double()

    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)

    def _make_skip_layer(self,x, in_shape, out_shape, hidden =[10]):
        layers = []
        x_tmp = x
        in_layer = in_shape

        # Add hidden layers
        # https://github.com/junyanz/pytorch-CycleGAN-and-pix2pix/blob/2ecf15f8a7f87fa56e784e0504136e9daf6b93d6/models/networks.py#L259
        for hidden_layer in hidden:
            x = nn.Linear(in_layer,hidden_layer)(x)
            in_layer = hidden_layer

        x = torch.cat((x,x_tmp), 1)
        x = nn.BatchNorm1d(hidden_layer)(x)
        x = nn.LeakyReLU(x)

        x = nn.Linear(in_shape + hidden_layer, out_shape)(x)
        x = nn.BatchNorm1d(hidden_layer)(x)
        x = nn.LeakyReLU(x)

        return x

    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)

    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)

class Model_fullyconnected_220923(nn.Module):
    '''
    Simple fully connected neural network 
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[256],dropout=0.5):  # 256-128-16 dropout0.5
        
        super().__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        ### Parameters
        self.dropout_rate = dropout

        ### Hidden laters
        self.inshape = in_shape
        self.in_shape_flatten = in_shape[0] * in_shape[1] 
        self.layers = self._make_layer(self.in_shape_flatten, out_shape,hidden)
        self.activation = nn.Tanh()

        ### Others
        self.reset_parameters()
    def forward(self,x):
        x = torch.flatten(x, 1)
        h = self.layers(x)
        h = self.activation(h)
        return h.double()
    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)
    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)

# class Model_Attention_221008(nn.Module):
    '''
    Simple fully connected neural network 
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[256],dropout=0.5):  # 256-128-16 dropout0.5
        
        super().__init__()
        # print(in_shape)

        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs

        # Self Attention
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        
        ### Parameters
        self.dropout_rate = dropout

        ### Hidden laters
        self.inshape = in_shape
        self.in_shape_flatten = in_shape[0] * in_shape[1] 
        self.layers = self._make_layer(self.in_shape_flatten, out_shape,hidden)
        self.activation = nn.Tanh()

        # # Attention layer
        # self.self_attn = MultiheadAttention(in_shape[0], num_heads=3)


        ### Others
        self.reset_parameters()
    def forward(self,x):
        # x = MultiheadAttention(x)
        print(x.shape)
        x = torch.flatten(x, 1)
        h = self.layers(x)
        h = self.activation(h)
        return h.double()
    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)
    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)
class Model_Attention_221029(nn.Module):
    '''
    Simple fully connected neural network 
    '''
    def __init__(self, in_shape, out_shape=1, hidden=[256],dropout=0.5,output_attention_weight=False):  # 256-128-16 dropout0.5
        super().__init__()
        # print(in_shape)
        # Todo This is the way to make layers dynamic
        # https://www.youtube.com/watch?v=DkNIBBBvcPs
        # Self Attention
        # https://uvadlc-notebooks.readthedocs.io/en/latest/tutorial_notebooks/tutorial6/Transformers_and_MHAttention.html
        
        ### Parameters
        self.dropout_rate = dropout
        self.output_attention_weight = output_attention_weight

        ### Hidden laters
        self.inshape = in_shape
        self.in_shape_flatten = in_shape[0] * in_shape[1] 
        self.layers = self._make_layer(self.in_shape_flatten, out_shape,hidden)
        self.activation = nn.Tanh()

        # # Attention layer
        self.self_attn = MultiheadAttention(in_shape[1], num_heads=1,batch_first=True)

        ### Others
        self.reset_parameters()
    def forward(self,x):
        x, attention_weight = self.self_attn(x,x,x)
        x = torch.flatten(x, 1)
        h = self.layers(x)
        h = self.activation(h)
        if self.output_attention_weight:
            return h.double(), attention_weight
        else:
            return h.double()
    def _make_layer(self, in_shape, out_shape, hidden):
        layers = []
        in_layer = in_shape

        # Add hidden layers
        for hidden_layer in hidden:
            layers.append(nn.Linear(in_layer,hidden_layer))
            layers.append(nn.Dropout(self.dropout_rate))
            layers.append(nn.BatchNorm1d(hidden_layer))
            layers.append(nn.LeakyReLU())
            in_layer = hidden_layer

        # Add the output layer
        layers.append(nn.Linear(in_layer, out_shape))
        # layers.append(nn.Tanh())

        return nn.Sequential(*layers)
    def reset_parameters(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, mean=0, std=1.0)
    def print_weight(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                print(m.weight)


# net = Net()
# optimizer = optim.Adam(net.parameters(), lr=LEARNING_RATE)
# loss = nn.NLLLoss() #loss = F.mse_loss(Q_                                cxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxexpected, Q_targets)
if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    model = Model_Attention_221029([36,12]).to(device)
    print(model)
    # print('hi')
