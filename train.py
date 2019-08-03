import chainer
from chainer import optimizers
from chainer import iterators
from chainer import functions as F

import numpy as np
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
    parser.add_argument('--nChannel', '-nc', metavar='N', default=100, type=int, help='number of channels')
    parser.add_argument('--num_superpixels', '-ns', metavar='K', default=10000, type=int, help='number of superpixels')
    parser.add_argument('--compactness', '-c', metavar='C', default=100, type=float, help='compactness of superpixels')
    parser.add_argument('--visualize', '-v', metavar='1 or 0', default=1, type=int, help='visualization flag')
    parser.add_argument('--minLabels', '-ml', metavar='minL', default=3, type=int, help='minimum number of labels')
    parser.add_argument('--use_colab', '-uc', default='false', type=strtobool, help='when you use colab')
    parser.add_argument('--input', metavar='FILENAME', help='input image file name', required=True)
    args = parser.parse_args()

    batchsize = args.batchsize # disabled value
    gpu = args.gpu
    epoch = args.epoch
    seed = args.seed # disabled value
    save_file_name = args.save_file_name # disabled value
    use_bn = args.use_bn # disabled value
    use_do = args.use_do # disabled value
    dropout_ratio = args.dropout_ratio # disabled value
    residual = args.residual # disabled value
    nChannel = args.nChannel
    compactness = args.compactness
    num_superpixels = args.num_superpixels
    visualize = args.visualize
    minLabels = args.minLabels
    use_colab = args.use_colab
    target_image = args.input

    if gpu >= 0:
        import cupy as cp
        xp = cp
    else:
        xp = np

    #load image no pro
    im = cv2.imread(target_image)
    print("A:",im)
    data = xp.array([im.transpose( (2, 0, 1) ).astype('float32')/255.])
    print("B:",data.shape)
    data = chainer.Variable(data)

    #slic
    labels = segmentation.slic(im, compactness=compactness, n_segments=num_superpixels)
    labels = labels.reshape(im.shape[0]*im.shape[1])
    print("C:",labels.shape)
    u_labels = np.unique(labels)
    print("D:",u_labels.shape)
    l_inds = []
    for i in range(len(u_labels)):
        l_inds.append(np.where(labels == u_labels[i])[0])
    print("E:",len(l_inds))

    #train
    model = KanezakiNet(data.shape[1],nChannel=nChannel)
    if(gpu >= 0):
        print('using gpu {}'.format(gpu))
        model.to_gpu(gpu)
    optimizer = optimizers.MomentumSGD(lr=0.1,momentum=0.9).setup(model)
    loss_fn = F.softmax_cross_entropy

    label_colours = np.random.randint(255,size=(100,3))
    print("F:",len(label_colours))

    for epoch_now in range(epoch):
        output = model.encoder(data)[0]
        print("G:",output.shape)
        output = F.reshape(F.transpose(output,axes=(1,2,0)),(-1,nChannel))
        print("H:",output.shape)
        target = F.argmax(output, 1)
        print("I:",target.shape)

        im_target = target.array
        if gpu >= 0:
            im_target = cp.asnumpy(im_target)
        nLabels = len(xp.unique(im_target))
        if visualize:
            im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])    
            im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
            if use_colab:
                from google.colab.patches import cv2_imshow
                cv2_imshow(im_target_rgb)
            else:
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
        if gpu >= 0:
            im_target = cp.asarray(im_target).to_gpu(gpu)
        target = chainer.Variable( im_target )
        loss = loss_fn(output, target)
        model.cleargrads()
        loss.backward()
        optimizer.update()

        print (epoch_now, '/', epoch, ':', nLabels, loss.data)
        if nLabels <= minLabels:
            print ("nLabels", nLabels, "reached minLabels", minLabels, ".")
            break

    # save output image
    if not visualize:
        output = model( data )[ 0 ]
        output = F.transpose(output, axes=(1,2,0))
        target = F.max(output, 1)
        im_target = target.array
        im_target_rgb = np.array([label_colours[ c % 100 ] for c in im_target])
        im_target_rgb = im_target_rgb.reshape( im.shape ).astype( np.uint8 )
    cv2.imwrite( "output.png", im_target_rgb )


if __name__ == "__main__":
    main()