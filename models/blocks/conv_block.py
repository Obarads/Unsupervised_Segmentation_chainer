import sys

import chainer
from chainer import functions
from chainer import links
from chainer import Sequential

class ConvBlock(chainer.Chain):
    # L.Convolution2D argument is same as ConvBlock argument except dlate and groups.
    # do = dropout
    # bn = BatchNormalization
    def __init__(self, in_channels, out_channels, ksize=None, stride=1, pad=0, processing_order=["bn","act"], nobias=False, initialW=None, initial_bias=None, activation=functions.relu, dropout_ratio=.5):
        super(ConvBlock, self).__init__()
        with self.init_scope():
            self.conv = links.Convolution2D(
                in_channels, out_channels, ksize=ksize, stride=stride, pad=pad,
                nobias=nobias, initialW=initialW, initial_bias=initial_bias)

            self.processing_list = Sequential()
            for po in processing_order:
                if po == "bn":
                    process = links.BatchNormalization(out_channels)
                elif po == "do":
                    process = self.__dropout
                elif po == "act":
                    process = activation
                else:
                    print('Error: processing_order contains exception.')
                    sys.exit(1)
                self.processing_list.append(process)

        self.activation = activation
        self.dropout_ratio = dropout_ratio

    def __call__(self, x):
        h = self.conv(x)
        h = self.processing_list(h)
        return h

    def __dropout(self,h):
        return functions.dropout(h, ratio=self.dropout_ratio)
