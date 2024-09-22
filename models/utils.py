import torch
from torch import nn
import torch.nn.functional as F

# A lot of modules are copied from
# https://github.com/Lightning-AI/lightning-bolts/blob/70ce46bbed9ae01c413934ffd0423d6cf05e1d2d/pl_bolts/models/autoencoders/components.py#L125

def get_dropout(drop_p):
    """ Getting a dropout layer """
    if(drop_p):
        drop = nn.Dropout(p=drop_p)
    else:
        drop = nn.Identity()
    return drop

def get_act(act_name):
    """ Gettign activation given name """
    assert act_name in ["ReLU", "Sigmoid", "Tanh"]
    activation = getattr(nn, act_name)

    return activation()

class LinearBlock(nn.Module):
    """Applies Linear Transforms to the Input"""
    def __init__(self, input_dim, sizes=[128], act="ReLU", dropout=0, **kwargs):
        super().__init__()
        self.input_dim = input_dim
        self.sizes = sizes
        self.act = act
        self.dropout = dropout
        self.linear_block = self._create_block()
    
    def _create_block(self):
        layers = []
        in_size = self.input_dim

        for size in self.sizes:
            layers.append(nn.Linear(in_features=in_size, out_features=size))
            if self.act:
                layers.append(get_act(self.act))
            layers.append(get_dropout(self.dropout))
            in_size = size
        
        linear_block = nn.Sequential(*layers)
        return linear_block
    
    def forward(self, x):
        return self.linear_block(x)

def init_state(cell_type, n_cells, rnn_dim, b_size, x):
    
    # type_as lightning thing
    # refer to https://pytorch-lightning.readthedocs.io/en/latest/accelerators/accelerator_prepare.html
    states = []
    for _ in range(n_cells):
        if cell_type == "LSTM":
            h = torch.zeros(b_size, rnn_dim).type_as(x)
            c = torch.zeros(b_size, rnn_dim).type_as(x)
            state = (h, c)
        else:
            state = torch.zeros(b_size, rnn_dim).type_as(x)
    
        states.append(state)
    
    return states

def init_state_hm(n_cells, rnn_ch, x):
    """
    Method for getting initial satates for convLSTM module
    """
    # type_as lightning thing
    # refer to https://pytorch-lightning.readthedocs.io/en/latest/accelerators/accelerator_prepare.html
    b_size, _, _, height, width = x.shape

    states = []
    for _ in range(n_cells):
        h = torch.zeros(b_size, rnn_ch, height, width).type_as(x)
        c = torch.zeros(b_size, rnn_ch, height, width).type_as(x)
        state = (h, c)
        states.append(state)

    
    return states


class ConvLSTMCell(nn.Module):
    """
    Implementation taken from:
    https://github.com/here-to-learn0/VideoFramePrediction/blob/master/scripts/conv_lstm.py
    """

    def __init__(self, input_dim, hidden_dim, kernel_size, bias=True, mode="zeros"):
        super(ConvLSTMCell, self).__init__()

        self.input_dim = input_dim
        self.hidden_dim = hidden_dim

        self.kernel_size = kernel_size
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.bias = bias
        self.mode = mode

        self.conv = nn.Conv2d(in_channels=self.input_dim + self.hidden_dim,
                              out_channels=4 * self.hidden_dim,
                              kernel_size=self.kernel_size,
                              padding=self.padding,
                              bias=self.bias)
    
    def forward(self, input_tensor, cur_state):
        h_cur, c_cur = cur_state

        combined = torch.cat([input_tensor, h_cur], dim=1)  # concatenate along channel axis

        combined_conv = self.conv(combined)
        cc_i, cc_f, cc_o, cc_g = torch.split(combined_conv, self.hidden_dim, dim=1)
        i = torch.sigmoid(cc_i)
        f = torch.sigmoid(cc_f)
        o = torch.sigmoid(cc_o)
        g = torch.tanh(cc_g)

        c_next = f * c_cur + i * g
        h_next = o * torch.tanh(c_next)

        return h_next, c_next


def conv3x3(in_planes, out_planes, stride=1):
    """3x3 convolution with padding."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride, padding=1, bias=False)

def conv1x1(in_planes, out_planes, stride=1):
    """1x1 convolution."""
    return nn.Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)


class Interpolate(nn.Module):
    """nn.Module wrapper for F.interpolate."""

    def __init__(self, size=None, scale_factor=None):
        super().__init__()
        self.size, self.scale_factor = size, scale_factor

    def forward(self, x):
        return F.interpolate(x, size=self.size, scale_factor=self.scale_factor)


def resize_conv3x3(in_planes, out_planes, scale=1):
    """upsample + 3x3 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv3x3(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv3x3(in_planes, out_planes))

def resize_conv1x1(in_planes, out_planes, scale=1):
    """upsample + 1x1 convolution with padding to avoid checkerboard artifact."""
    if scale == 1:
        return conv1x1(in_planes, out_planes)
    return nn.Sequential(Interpolate(scale_factor=scale), conv1x1(in_planes, out_planes))


class EncoderBlock(nn.Module):
    """ResNet block, copied from https://github.com/pytorch/vision/blob/master/torchvision/models/resnet.py#L35."""

    expansion = 1

    def __init__(self, inplanes, planes, stride=1, downsample=None):
        super().__init__()
        self.conv1 = conv3x3(inplanes, planes, stride)
        self.bn1 = nn.BatchNorm2d(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes)
        self.bn2 = nn.BatchNorm2d(planes)
        self.downsample = downsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class DecoderBlock(nn.Module):
    """ResNet block, but convs replaced with resize convs, and channel increase is in second conv, not first."""

    expansion = 1

    def __init__(self, inplanes, planes, scale=1, upsample=None):
        super().__init__()
        self.conv1 = resize_conv3x3(inplanes, inplanes)
        self.bn1 = nn.BatchNorm2d(inplanes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = resize_conv3x3(inplanes, planes, scale)
        self.bn2 = nn.BatchNorm2d(planes)
        self.upsample = upsample

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.upsample is not None:
            identity = self.upsample(x)

        out += identity
        out = self.relu(out)

        return out


class ResNetEncoder(nn.Module):
    def __init__(self, block, planes_list, n_blocks_per_layer=2):
        super().__init__()
        self.inplanes = 17

        self.planes_list = planes_list
        self.layers = [n_blocks_per_layer] * len(planes_list)

        encoder_layers = []

        for i, planes in enumerate(planes_list):
            stride = 1 if i==0 else 2
            encoder_layers.append(
                self._make_layer(block, planes, self.layers[i], stride=stride)
            )
        

        self.encoder_layers = nn.Sequential(*encoder_layers)


    
    def _make_layer(self, block, planes, blocks, stride=1):
        downsample = None
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.encoder_layers(x)


class ResNetDecoder(nn.Module):
    """Resnet Enconder in reverse order."""
    def __init__(self, block, planes_list, in_planes, n_blocks_per_layer=2):
        super().__init__()
        self.expansion = block.expansion
        self.inplanes = in_planes
        self.planes_list = planes_list
        self.layers = [n_blocks_per_layer] * len(planes_list)

        decoder_layers = []

        for i, planes in enumerate(planes_list):
            scale = 1 if i == len(planes_list) - 1 else 2
            decoder_layers.append(
                self._make_layer(block, planes, self.layers[i], scale=scale)
            )
        

        self.decoder_layers = nn.Sequential(*decoder_layers)
    

    def _make_layer(self, block, planes, blocks, scale=1):
        upsample = None
        if scale != 1 or self.inplanes != planes * block.expansion:
            upsample = nn.Sequential(
                resize_conv1x1(self.inplanes, planes * block.expansion, scale),
                nn.BatchNorm2d(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, scale, upsample))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes))

        return nn.Sequential(*layers)
    
    def forward(self, x):
        return self.decoder_layers(x)