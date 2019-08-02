"""
This code is written based on the following reference.
corochann, "corochann/chainer-pointnet: Chainer implementation of PointNet, PointNet++, KD-Network and 3DContextNework," [Online]. Available: https://github.com/corochann/chainer-pointnet. [Accessed: 30-Jun-2019]

should modify this code (lrn and bn)
"""

import chainer
from chainer import functions as F
from chainer import links as L

class ConvBlock(chainer.Chain):

    # L.Convolution2D argument is same as ConvBlock argument except dlate and groups.
    # do = dropout
    # bn = BatchNormalization
    # lrn = local_response_normalization
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0,
                 nobias=False, initialW=None, initial_bias=None, activation=F.relu, 
                 use_do=False, dropout_ratio=.5, residual=False, use_bn=False, use_lrn=False):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = L.Convolution2D(
                in_channels, out_channels, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias)
            if use_bn:
                self.bn = L.BatchNormalization(out_channels)

        self.activation = activation
        self.use_bn = use_bn
        self.use_lrn = use_lrn
        self.use_do = use_do
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):

        h = self.conv(x)

        if self.use_lrn:
            h = F.local_response_normalization(h)
        if self.use_bn:
            h = self.bn(h)
        
        if self.activation is not None:
            h = self.activation(h)
        
        if self.residual:
            raise NotImplementedError('not implemented yet')

        if self.use_do:
            h = F.dropout(h, ratio=self.dropout_ratio)

        return h
