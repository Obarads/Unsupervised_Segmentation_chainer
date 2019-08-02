"""
This code is written based on the following reference.
corochann, "corochann/chainer-pointnet: Chainer implementation of PointNet, PointNet++, KD-Network and 3DContextNework," [Online]. Available: https://github.com/corochann/chainer-pointnet. [Accessed: 30-Jun-2019]
"""

import chainer
from chainer import functions as F
from chainer import links as L

class LinearBlock(chainer.Chain):

    def __init__(self, in_size, out_size=None, nobias=False,
                 initialW=None, initial_bias=None, activation=F.relu, 
                 use_do=False, dropout_ratio=.5, residual=False, use_bn=False):
        super(LinearBlock, self).__init__()
        with self.init_scope():
            self.linear = L.Linear(
                in_size, out_size=out_size, nobias=nobias,
                initialW=initialW, initial_bias=initial_bias)
            if use_bn:
                self.bn = L.BatchNormalization(out_size)

        self.activation = activation
        self.use_bn = use_bn
        self.use_do = use_do
        self.dropout_ratio = dropout_ratio
        self.residual = residual

    def __call__(self, x):

        h = self.linear(x)

        if self.use_bn:
            h = self.bn(h)

        if self.activation is not None:
            h = self.activation(h)

        if self.residual:
            raise NotImplementedError('not implemented yet')

        if self.use_do:
            h = F.dropout(h, ratio=self.dropout_ratio)

        return h
