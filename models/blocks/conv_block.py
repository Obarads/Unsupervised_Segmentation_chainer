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

        if self.activation is not None:
            h = self.activation(h)

        if self.use_lrn:
            h = F.local_response_normalization(h)
        if self.use_bn:
            h = self.bn(h)
        
        if self.residual:
            raise NotImplementedError('not implemented yet')

        if self.use_do:
            h = F.dropout(h, ratio=self.dropout_ratio)

        return h

import sys

import chainer
from chainer import functions
from chainer import links

class ConvBlock1(chainer.Chain):
    # L.Convolution2D argument is same as ConvBlock argument except dlate and groups.
    # do = dropout
    # bn = BatchNormalization
    # lrn = local_response_normalization
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, processing_order=["bn","act"], nobias=False, initialW=None, initial_bias=None, activation=functions.relu, dropout_ratio=.5):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = links.Convolution2D(
                in_channels, out_channels, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias)

            processing_list = chainer.ChainList()
            for po in processing_order:
                if po == "bn":
                    processing_list.append(links.BatchNormalization(out_channels))
                elif po == "do":
                    processing_list.append(self.__dropout)
                elif po == "act":
                    processing_list.append(activation)
                else:
                    print('Error: processing_order contains exception.')
                    sys.exit(1)
            self.processing_list = processing_list

        self.activation = activation
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = self.conv(x)
        for pl in self.processing_list:
            h = pl(h)
        return h

    def __dropout(self,h):
        return functions.dropout(h, ratio=self.dropout_ratio)
