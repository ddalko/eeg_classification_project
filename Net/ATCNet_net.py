import torch
import torch.nn as nn
import torch.nn.functional as F

class MultiHeadAttentionBlock(nn.Module):
    def __init__(self, embed_dim, num_heads, dropout=0.5):
        super(MultiHeadAttentionBlock, self).__init__()
        self.attention = nn.MultiheadAttention(embed_dim=embed_dim, num_heads=num_heads, dropout=dropout)
        self.layer_norm = nn.LayerNorm(embed_dim)
        self.dropout = nn.Dropout(0.3)
    
    def forward(self, x):
        x = self.layer_norm(x)
        q, k, v = x, x, x
        x, _ = self.attention(q, k, v)
        x = x + self.dropout(x)
        return x

class ConvBlock(nn.Module):
    def __init__(self, in_chans, F1=16, D=2, kern_length=64, pool_size=8, dropout=0.1):
        super(ConvBlock, self).__init__()
        F2 = F1 * D

        self.conv1 = nn.Conv2d(
            in_channels=1,
            out_channels=F1,
            kernel_size=(1, kern_length),
            padding='same',
            bias=False
        )
        self.bn1 = nn.BatchNorm2d(F1)

        self.depthwise_conv = nn.Conv2d(
            in_channels=F1,
            out_channels=F2,
            kernel_size=(in_chans, 1),
            groups=F1,
            bias=False
        )
        self.bn2 = nn.BatchNorm2d(F2)
        self.pool1 = nn.AvgPool2d(kernel_size=(1, pool_size))
        self.dropout = nn.Dropout(dropout)

        self.conv2 = nn.Conv2d(
            in_channels=F2,
            out_channels=F2,
            kernel_size=(1, 16),
            groups=F2,
            bias=False,
            padding='same'
        )
        self.bn3 = nn.BatchNorm2d(F2)
        self.pool2 = nn.AvgPool2d(kernel_size=(1, 8))
    
    def forward(self, x):
        x = self.conv1(x)
        x = self.bn1(x)
        x = F.elu(x)
        x = self.depthwise_conv(x)
        x = self.bn2(x)
        x = F.elu(x)
        x = self.pool1(x)
        x = self.dropout(x)
        x = self.conv2(x)
        x = self.bn3(x)
        x = F.elu(x)
        x = self.pool2(x)
        x = self.dropout(x)
        return x


class ResidualBlock(nn.Module):
    def __init__(self, in_channels, out_channels, kernel_size, dilation, dropout):
        super(ResidualBlock, self).__init__()
        self.kernel_size = kernel_size
        self.dilation = dilation
        self.conv1 = nn.Conv1d(
            in_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size - 1) * dilation,
            dilation=dilation
        )
        self.bn1 = nn.BatchNorm1d(out_channels)
        self.conv2 = nn.Conv1d(
            out_channels,
            out_channels,
            kernel_size=kernel_size,
            padding=(kernel_size-1) * dilation,
            dilation=dilation
        )
        self.bn2 = nn.BatchNorm1d(out_channels)
        self.dropout = nn.Dropout(dropout)

    def forward(self, x):
        residual = x
        x = self.conv1(x)[:, :, :-(self.kernel_size-1)*self.dilation]
        x = self.bn1(x)
        x = F.elu(x)
        x = self.dropout(x)
        
        x = self.conv2(x)[:, :, :-(self.kernel_size-1)*self.dilation]
        x = self.bn2(x)
        x = F.elu(x)
        x = self.dropout(x)

        x += residual
        return x

class TemporalConvNet(nn.Module):
    def __init__(self, input_channels, num_filters, kernel_size, depth, dropout):
        super(TemporalConvNet, self).__init__()
        self.blocks = nn.ModuleList()
        for i in range(depth):
            dilation = 2 ** i
            in_channels = input_channels if i == 0 else num_filters
            self.blocks.append(ResidualBlock(in_channels, num_filters, kernel_size, dilation, dropout))
        
    def forward(self, x):
        for block in self.blocks:
            x = block(x)
        return x

class ATCNet(nn.Module):
    def __init__(self, n_classes=4, n_windows=5, in_chans=22, eegn_F1=16, eegn_D=2, eegn_kernel_size=64, eegn_pool_size=7, eegn_dropout=0.3, tcn_depth=2, tcn_kernel_size=4, tcn_filters=32, tcn_dropout=0.3, fuse='average'):
        super(ATCNet, self).__init__()
        self.n_windows = n_windows
        self.fuse = fuse

        self.conv_block = ConvBlock(
            in_chans=in_chans,
            F1=eegn_F1,
            D=eegn_D,
            kern_length=eegn_kernel_size,
            pool_size=eegn_pool_size,
            dropout=eegn_dropout
        )

        self.attention = MultiHeadAttentionBlock(
            embed_dim=eegn_F1 * eegn_D,
            num_heads=2,
        )

        self.temporal_cnn = TemporalConvNet(
            input_channels=eegn_F1 * eegn_D,
            num_filters=tcn_filters,
            kernel_size=tcn_kernel_size,
            depth=tcn_depth,
            dropout=tcn_dropout
        )

        self.classifier = nn.Linear(tcn_filters, n_classes)
    
    def forward(self, x):
        x = self.conv_block(x)
        x = x.squeeze()

        outputs = []
        for i in range(self.n_windows):
            start = i
            end = x.size(-1) - self.n_windows + i + 1
            window = x[:, :, start:end]
            window = window.permute(2, 0, 1)
            window = self.attention(window)
            window = window.permute(1, 2, 0)
            tcn_output = self.temporal_cnn(window)
            last_step = tcn_output[:, :, -1]
            output = self.classifier(last_step)
            outputs.append(output)
        x = torch.stack(outputs, dim=0).mean(dim=0)
        return x
