from torch import nn
from tcn import TemporalConvNet


class TCN(nn.Module):
    def __init__(self, input_size, output_size, num_channels, kernel_size, dropout):
        '''

        Args:
            input_size: int,  输入通道数或者特征数
            output_size:int, 输出通道数或者特征数
            num_channels:list, 每层的hidden_channel数. 例如[5,12,3], 代表有3个block,
                                block1的输出channel数量为5;
                                block2的输出channel数量为12;
                                block3的输出channel数量为3.
            kernel_size: int, 卷积核尺寸
            dropout: float, drop_out比率
        '''
        super(TCN, self).__init__()
        self.tcn = TemporalConvNet(input_size, num_channels, kernel_size=kernel_size, dropout=dropout)
        self.linear = nn.Linear(num_channels[-1], output_size)
        self.init_weights()

    def init_weights(self):
        self.linear.weight.data.normal_(0, 0.01)

    def forward(self, x):
        y1 = self.tcn(x)
        return self.linear(y1[:, :, -1])