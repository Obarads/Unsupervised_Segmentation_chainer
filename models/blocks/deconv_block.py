import chainer
from chainer import functions
from chainer import links

class DeconvBlock(chainer.Chain):

    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, outsize=None ,initial_bias=None, dilate=1,groups=1,use_bn=True,
                 activation=functions.relu, dropout_ratio=-1, residual=False):
        super(DeconvBlock, self).__init__()
        with self.init_scope():
            self.conv = links.Deconvolution2D(
                in_channels, out_channels, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, outsize=outsize,initialW=initialW, initial_bias=initial_bias)
            if use_bn:
                self.bn = links.BatchNormalization(out_channels)
        self.activation = activation
        self.use_bn = use_bn
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):
        if self.use_bn:
            h = self.bn(self.conv(x))
        else:
            h = self.conv(x)
        if self.activation is not None:
            h = self.activation(h)
        """
        if self.residual:
            from chainerex.functions import residual_add
            h = residual_add(h, x)
        """
        if self.residual:
            raise NotImplementedError('not implemented yet')
        if self.dropout_ratio >= 0:
            h = functions.dropout(h, ratio=self.dropout_ratio)
        return h
