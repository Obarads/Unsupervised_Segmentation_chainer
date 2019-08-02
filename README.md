# unsupervised_segmentation_chainer
## About
This repository is a chainer implementation of the method proposed by Kanezaki[1].

## How to use
### Install
Please install following library.  
```bash
pip install chainer
pip install opencv-python
```

If GPU is usable in your environment, Please install following library.
```bash
pip install cupy-cuda90
```

### Execute
Following command execute learning and demo.
```bash
python train.py
```

Option  
- use gpu : `-g 0`
- use colab : `-uc true`

## Reference
1. [Asako Kanezaki. Unsupervised Image Segmentation by Backpropagation. IEEE International Conference on Acoustics, Speech and Signal Processing (ICASSP), 2018.](https://kanezaki.github.io/pytorch-unsupervised-segmentation/ICASSP2018_kanezaki.pdf)