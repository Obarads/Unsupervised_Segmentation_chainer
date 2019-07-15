import chainer
from chainer import optimizers
from chainer import iterators
from chainer import functions as F

import numpy as np
import cupy as cp
import argparse
import os
import cv2
from skimage import segmentation
from distutils.util import strtobool

from models.kanezaki_net import KanezakiNet

def main():
    # Arguments
    parser = argparse.ArgumentParser(description='train model.')
    #parser.add_argument('--config', 'c', type=str, default=None)
    parser.add_argument('--batchsize', '-b', type=int, default=32)
    parser.add_argument('--gpu', '-g', type=int, default=-1)
    parser.add_argument('--out', '-o', type=str, default='result')
    parser.add_argument('--epoch', '-e', type=int, default=1000)
    parser.add_argument('--seed', '-s', type=int, default=777)
    parser.add_argument('--save_file_name','-sfn', type=str, default='model_ae.npz')
    parser.add_argument('--use_bn', '-ub', type=strtobool, default='true')
    parser.add_argument('--use_do', '-ud', type=strtobool, default='false')
    parser.add_argument('--dropout_ratio', '-dr', type=float, default=0)
    parser.add_argument('--residual', '-r', type=strtobool, default='false')
    parser.add_argument('--num_superpixels', '-ns', metavar='K', default=10000, type=int, help='number of superpixels')
    parser.add_argument('--compactness', '-c', metavar='C', default=100, type=float, help='compactness of superpixels')
    parser.add_argument('--visualize', '-v', metavar='1 or 0', default=1, type=int, help='visualization flag')
    parser.add_argument('--minLabels', '-ml', metavar='minL', default=3, type=int, help='minimum number of labels')

    args = parser.parse_args()

    batchsize = args.batchsize
    gpu = args.gpu
    epoch = args.epoch
    seed = args.seed
    save_file_name = args.save_file_name
    use_bn = args.use_bn
    use_do = args.use_do
    dropout_ratio = args.dropout_ratio
    residual = args.residual
    compactness = args.compactness
    num_superpixels = args.num_superpixels
    visualize = args.visualize
    minLabels = args.minLabels

    #load image no pro
    im = cv2.imread("data/trials/108004.jpg")
    if gpu >= 0:
        xp = cp
    else:
        xp = np
    data = xp.array([im.transpose( (2, 0, 1) ).astype('float32')/255.])
    data = chainer.Variable(data)

    #slic
    labels = segmentation.slic(im, compactness=compactness, n_segments=num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    u_labels = np.unique(labels)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])

    #train
    print(data.shape)
    model = KanezakiNet(data.shape[1])
    if(gpu >= 0):
        print('using gpu {}'.format(gpu))
        chainer.backends.cuda.get_device_from_id(gpu).use()
        model.to_gpu()
    optimizer = optimizers.SGD(lr=0.1).setup(model)
    loss_fn = F.softmax_cross_entropy

    label_colours = np.random.randint(255,size=(100,3))

    for i in range(epoch):
        output = model.encoder(data)
        output = output.transpose(1,2,0)
        target = F.max(output, 1)
        im_target = target.array
        nLabels = len(xp.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            cv2.imshow( "output", im_target_rgb )
            cv2.waitKey(10)
        
        # superpixel refinement
        # TODO: use Torch Variable instead of numpy for faster calculation
        for i in range(len(l_inds)):
            labels_per_sp = im_target[ l_inds[ i ] ]
            u_labels_per_sp = np.unique( labels_per_sp )
            hist = np.zeros( len(u_labels_per_sp) )
            for j in range(len(hist)):
                hist[ j ] = len( np.where( labels_per_sp == u_labels_per_sp[ j ] )[ 0 ] )
            im_target[ l_inds[ i ] ] = u_labels_per_sp[ np.argmax( hist ) ]
        target = chainer.Variable( im_target )
        loss = loss_fn(output, target)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        print (i, '/', epoch, ':', nLabels, loss.data)
        # for pytorch 1.0
        # print (batch_idx, '/', args.maxIter, ':', nLabels, loss.item())
        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break

    # save output image
    if not visualize:
        output = model( data )[ 0 ]
        output = output.transpose(1,2,0)
        target = F.max(output, 1)
        im_target = target.array
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite( "output.png", im_target_rgb )


if __name__ == "__main__":
    main()