import torch
import torch.nn as nn
import math

class DenseBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, bias=True, activation='prelu', norm='batch'):
        super(DenseBlock, self).__init__()
        self.fc = torch.nn.Linear(input_dim, output_dim, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm1d(output_dim)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm1d(output_dim)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_dim)
        elif self.norm == 'spectral':
            self.fc = nn.utils.spectral_norm(self.fc)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None and self.norm is not 'spectral':
            out = self.bn(self.fc(x))
        else:
            out = self.fc(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ConvBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=0, dilation=1, groups=1, bias=True, activation='prelu', norm=None):
        super(ConvBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv = torch.nn.Conv2d(input_dim, output_dim, kernel_size, stride, padding, dilation, groups, bias=bias)

        self.norm = norm
        if self.norm =='batch':
            self.bn = torch.nn.BatchNorm2d(output_dim)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_dim)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_dim)
        elif self.norm == 'spectral':
            self.conv = nn.utils.spectral_norm(self.conv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None and self.norm is not 'spectral':
            out = self.bn(self.conv(x))
        else:
            out = self.conv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class DeconvBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=4, stride=2, padding=1, bias=True, activation='prelu', norm=None):
        super(DeconvBlock, self).__init__()

        self.input_dim = input_dim
        self.output_dim = output_dim

        self.deconv = torch.nn.ConvTranspose2d(input_dim, output_dim, kernel_size, stride, padding, bias=bias)

        self.norm = norm
        if self.norm == 'batch':
            self.bn = torch.nn.BatchNorm2d(output_dim)
        elif self.norm == 'instance':
            self.bn = torch.nn.InstanceNorm2d(output_dim)
        elif self.norm == 'group':
            self.bn = torch.nn.GroupNorm(32, output_dim)
        elif self.norm == 'spectral':
            self.deconv = nn.utils.spectral_norm(self.deconv)

        self.activation = activation
        if self.activation == 'relu':
            self.act = torch.nn.ReLU(True)
        elif self.activation == 'prelu':
            self.act = torch.nn.PReLU()
        elif self.activation == 'lrelu':
            self.act = torch.nn.LeakyReLU(0.2, True)
        elif self.activation == 'tanh':
            self.act = torch.nn.Tanh()
        elif self.activation == 'sigmoid':
            self.act = torch.nn.Sigmoid()

    def forward(self, x):
        if self.norm is not None and self.norm is not 'spectral':
            out = self.bn(self.deconv(x))
        else:
            out = self.deconv(x)

        if self.activation is not None:
            return self.act(out)
        else:
            return out


class ResidualBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResidualBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.Conv2d(input_dim, output_dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, kernel_size, stride=1, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(output_dim)

        if input_dim != output_dim or stride != 1:
            self.skip_layer = nn.Sequential(
                nn.Conv2d(input_dim, output_dim, 1, stride=stride, padding=0, bias=bias),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.skip_layer = None

        self.act2 = nn.ReLU(True)


    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        return self.act2(x + residual)


class ResidualUpBlock(torch.nn.Module):
    def __init__(self, input_dim, output_dim, kernel_size=3, stride=1, padding=1, bias=True):
        super(ResidualUpBlock, self).__init__()
        self.input_dim = input_dim
        self.output_dim = output_dim

        self.conv1 = nn.ConvTranspose2d(input_dim, output_dim, 4, stride=2, padding=1, bias=bias)
        self.bn1 = nn.BatchNorm2d(output_dim)
        self.act1 = nn.ReLU(True)
        self.conv2 = nn.Conv2d(output_dim, output_dim, 3, stride=1, padding=1, bias=bias)
        self.bn2 = nn.BatchNorm2d(output_dim)

        if input_dim != output_dim or stride != 1:
            self.skip_layer = nn.Sequential(
                nn.ConvTranspose2d(input_dim, output_dim, 4, stride=2, padding=1, bias=bias),
                nn.BatchNorm2d(output_dim)
            )
        else:
            self.skip_layer = None

        self.act2 = nn.ReLU(True)


    def forward(self, x):
        if self.skip_layer is not None:
            residual = self.skip_layer(x)
        else:
            residual = x

        x = self.conv1(x)
        x = self.bn1(x)
        x = self.act1(x)
        x = self.conv2(x)
        x = self.bn2(x)
        
        return self.act2(x + residual)


class MargeBlock(torch.nn.Module):
    def __init__(self, dim, kernel_size=3, stride=1, padding=1, bias=True, activation='prelu', norm='batch'):
        super(MargeBlock, self).__init__()
        self.conv1 = nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn1 = nn.BatchNorm2d(dim)
        self.conv2 = nn.Conv2d(dim, dim, kernel_size, stride=stride, padding=padding, bias=bias)
        self.bn2 = nn.BatchNorm2d(dim)

        if activation == 'relu':
            self.act = nn.ReLU(True)

    def forward(self, a, b):
         a = self.bn1(self.conv1(a))
         b = self.bn2(self.conv2(b))

         return self.act(a+b)



class UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(UpBlock, self).__init__()
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class D_UpBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_UpBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        self.up_conv1 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv2 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.up_conv3 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        h0 = self.up_conv1(x)
        l0 = self.up_conv2(h0)
        h1 = self.up_conv3(l0 - x)
        return h1 + h0


class DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, bias=True, activation='prelu', norm=None):
        super(DownBlock, self).__init__()
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0


class D_DownBlock(torch.nn.Module):
    def __init__(self, num_filter, kernel_size=8, stride=4, padding=2, num_stages=1, bias=True, activation='prelu', norm=None):
        super(D_DownBlock, self).__init__()
        self.conv = ConvBlock(num_filter*num_stages, num_filter, 1, 1, 0, bias=bias, activation=activation, norm=norm)
        self.down_conv1 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv2 = DeconvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)
        self.down_conv3 = ConvBlock(num_filter, num_filter, kernel_size, stride, padding, bias=bias, activation=activation, norm=norm)

    def forward(self, x):
        x = self.conv(x)
        l0 = self.down_conv1(x)
        h0 = self.down_conv2(l0)
        l1 = self.down_conv3(h0 - x)
        return l1 + l0