import chainer
import chainer.functions as F
import chainer.links as L

from chainer_codebox.blocks.conv_block import ConvBlock

class KanezakiNet(chainer.Chain):
    def __init__(self, input_dim, nChannel=100, nConv=2):
        super(KanezakiNet, self).__init__()
        with self.init_scope():
            self.conv_b1 = ConvBlock(input_dim,nChannel,ksize=3,stride=1,pad=1,use_bn=True)
            self.conv_b2 = []
            for i in range(nConv-1):
                self.conv_b2.append(ConvBlock(nChannel, nChannel, ksize=3, stride=1, pad=1, use_bn=True))
            self.conv_b3 = ConvBlock(nChannel, nChannel, ksize=1, stride=1, pad=0, use_bn=True)
            self.nChannel = nChannel
            self.nConv = nConv

    def __call__(self,x,y):
        h = self.conv_b1(x)
        for i in range(self.nConv-1):
            h = self.conv_b2[i](h)
        h = self.conv_b3(h)
        return h