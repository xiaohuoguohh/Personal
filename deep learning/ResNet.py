import torch
from torch import nn
from torch.nn import functional as F

class ResBlk(nn.Module):
    """
    resnet block
    """
    def __init__(self, chan_in, chan_out, stride=1):
        """
        :param chan_in:
        :param chan_out:
        """
        super(ResBlk, self).__init__()

        self.conv1 = nn.Conv2d(chan_in, chan_out, kernel_size=3, stride=stride, padding=1)
        self.bn1 = nn.BatchNorm2d(chan_out)
        self.conv2 = nn.Conv2d(chan_out, chan_out, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(chan_out)

        self.extra = nn.Sequential()
        if chan_in != chan_out:
            self.extra = nn.Sequential(
                nn.Conv2d(chan_in, chan_out, kernel_size=1, stride=stride),
                nn.BatchNorm2d(chan_out)
            )

    def forward(self, x):
        """
        :param x: [b, chan, h, w]
        :return:
        """
        out = F.relu(self.bn1(self.conv1(x)))
        out = self.bn2(self.conv2(out))
        # shortcut
        # [b, chan_in, h, w] ==> [b, chan_out, h, w]
        # element-wise add:

        out = self.extra(x) + out

        return out

class ResNet18(nn.Module):
    def __init__(self):
        super(ResNet18, self).__init__()

        self.conv1 = nn.Sequential(
            nn.Conv2d(3, 64, kernel_size=3, stride=3, padding=0),
            nn.BatchNorm2d(64)
        )
        # followed 4 blocks
        # [b, 64, h, w] ==> [b, 128, h, w]
        self.blk1 = ResBlk(64, 128, stride=2)
        # [b, 128, h, w] ==> [b, 256, h, w]
        self.blk2 = ResBlk(128, 256, stride=2)
        # [b, 256, h, w] ==> [b, 512, h, w]
        self.blk3 = ResBlk(256, 512, stride=2)
        # [b, 512, h, w] ==> [b, 512, h, w]
        self.blk4 = ResBlk(512, 512, stride=2)

        self.outlayer = nn.Linear(512*1*1, 10)

    def forward(self, x):
        """

        :param x:
        :return:
        """
        x = F.relu(self.conv1(x))

        # [b, 64, h, w] ==> [b, 512, h, w]
        x = self.blk1(x)
        x = self.blk2(x)
        x = self.blk3(x)
        x = self.blk4(x)
        #print('after conv:', x.shape)
        x = F.adaptive_max_pool2d(x, [1, 1])
        #print('after pooling:', x.shape)
        x = x.view(x.size(0), -1)

        x = self.outlayer(x)

        return x

#def main():
#    blk = ResBlk(64, 128, stride=2)
#    tmp = torch.randn(2, 64, 32, 32)
#    out = blk(tmp)
#    print('block :', out.shape)

#    x = torch.randn(2, 3, 32, 32)
#    model = ResNet18()
#    out = model(x)
#    print('resnet:', out.shape)

#if __name__ == '__main__':
#    main()






