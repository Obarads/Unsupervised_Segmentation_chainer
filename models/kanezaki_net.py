import chainer
import chainer.functions as F
import chainer.links as L

from chainer_codebox.blocks.conv_block import ConvBlock

N_CHANNEL = 100

class KanezakiNet(chainer.Chain):
    def __init__(self, input_dim, nChannel=100, nConv=2):
        super(KanezakiNet, self).__init__()
        with self.init_scope():
            self.conv_b1 = ConvBlock(input_dim,nChannel,ksize=3,stride=1,pad=1,use_bn=True)
            for 
            self.conv_b2 = ConvBlock(nChannel, )

