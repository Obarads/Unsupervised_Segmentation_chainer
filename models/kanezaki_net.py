import os, sys
sys.path.append(os.path.dirname(__file__))

import chainer
import chainer.functions as F
import chainer.links as L

class KanezakiNet(chainer.Chain):
    def __init__(self, input_dim, nChannel=100, nConv=2):
        super(KanezakiNet, self).__init__()
        with self.init_scope():
            self.conv1 = L.Convolution2D(input_dim, nChannel, ksize=3, stride=1, pad=1)
            self.bn1 = L.BatchNormalization(nChannel)
            self.conv2 = chainer.ChainList()
            self.bn2 = chainer.ChainList()
            for i in range(nConv-1):
                self.conv2.append(L.Convolution2D(nChannel,nChannel,ksize=3,stride=1,pad=1))
                self.bn2.append(L.BatchNormalization(nChannel))
            self.conv3 = L.Convolution2D(nChannel,nChannel,ksize=1,stride=1,pad=0)
            self.bn3 = L.BatchNormalization(nChannel)

            self.nChannel = nChannel
        self.nConv = nConv

    def __call__(self,x):
        x = self.bn1(F.relu(self.conv1(x)))
        for i in range(self.nConv-1):
            x = self.bn2[i](F.relu(self.conv2[i](x)))
        x = self.bn3(self.conv3(x))
        return x
