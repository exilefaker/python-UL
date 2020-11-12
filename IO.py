from array import array
import struct
import numpy as np

def load_mnist(dataset='train'):
        
    if dataset == 'train':
        fn = 'datasets/MNIST/train-images-idx3-ubyte'
        fn2 = 'datasets/MNIST/train-labels-idx1-ubyte'
    elif dataset == 'test':
        fn = 'datasets/MNIST/t10k-images-idx3-ubyte'
        fn2 = 'datasets/MNIST/t10k-labels-idx1-ubyte'

    print("Loading MNIST dataset...")
    #parse the matrix file into temporary files for each header item
    with open(fn,'rb') as f:
        print("Loading images...")
        magic_nr, N, rows, cols = struct.unpack(">IIII", f.read(16))
        print("Magic number: %s; N: %s; rows: %s; cols: %s" % (magic_nr, N, rows, cols))
        img = array("B", f.read())

    with open(fn2, 'rb') as g:
        print("Loading labels...")
        magic_nr, size = struct.unpack(">II", g.read(8))
        print("Magic number: %s; N: %s" % (magic_nr, size))
        lbl = array("b", g.read())

    ind = [k for k in range(N)]

    images = np.zeros((N, rows*cols), dtype='uint8')
    labels = np.zeros((N), dtype='int8')

    for i in range(len(ind)):
        images[i] = np.array(img[ind[i]*rows*cols:(ind[i]+1)*rows*cols])
        labels[i] = lbl[ind[i]]

    return images, labels