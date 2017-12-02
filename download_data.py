import os
import sys
import gzip
import shutil
from six.moves import urllib

# mnist dataset
HOMEPAGE = "http://yann.lecun.com/exdb/mnist/"
MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"

# fashion-mnist dataset
HOMEPAGE = "http://fashion-mnist.s3-website.eu-central-1.amazonaws.com/"
FASHION_MNIST_TRAIN_IMGS_URL = HOMEPAGE + "train-images-idx3-ubyte.gz"
FASHION_MNIST_TRAIN_LABELS_URL = HOMEPAGE + "train-labels-idx1-ubyte.gz"
FASHION_MNIST_TEST_IMGS_URL = HOMEPAGE + "t10k-images-idx3-ubyte.gz"
FASHION_MNIST_TEST_LABELS_URL = HOMEPAGE + "t10k-labels-idx1-ubyte.gz"


def download_and_uncompress_zip(URL, dataset_dir, force=False):
    '''
    Args:
        URL: the download links for data
        dataset_dir: the path to save data
        force: re-download data
    '''
    filename = URL.split('/')[-1]
    filepath = os.path.join(dataset_dir, filename)
    if not os.path.exists(dataset_dir):
        os.mkdir(dataset_dir)
    extract_to = os.path.splitext(filepath)[0]

    def download_progress(count, block_size, total_size):
        sys.stdout.write("\r>> Downloading %s %.1f%%" % (filename, float(count * block_size) / float(total_size) * 100.))
        sys.stdout.flush()

    if not force and os.path.exists(filepath):
        print("file %s already exist" % (filename))
    else:
        filepath, _ = urllib.request.urlretrieve(URL, filepath, download_progress)
        print()
        print('Successfully Downloaded', filename)

    # with zipfile.ZipFile(filepath) as fd:
    with gzip.open(filepath, 'rb') as f_in, open(extract_to, 'wb') as f_out:
        print('Extracting ', filename)
        shutil.copyfileobj(f_in, f_out)
        print('Successfully extracted')
        print()


def start_download(dataset, save_to, force):
    if not os.path.exists(save_to):
        os.mkdir(save_to)
    if dataset == 'mnist':
        download_and_uncompress_zip(MNIST_TRAIN_IMGS_URL, save_to, force)
        download_and_uncompress_zip(MNIST_TRAIN_LABELS_URL, save_to, force)
        download_and_uncompress_zip(MNIST_TEST_IMGS_URL, save_to, force)
        download_and_uncompress_zip(MNIST_TEST_LABELS_URL, save_to, force)
    elif dataset == 'fashion-mnist':
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_IMGS_URL, save_to, force)
        download_and_uncompress_zip(FASHION_MNIST_TRAIN_LABELS_URL, save_to, force)
        download_and_uncompress_zip(FASHION_MNIST_TEST_IMGS_URL, save_to, force)
        download_and_uncompress_zip(FASHION_MNIST_TEST_LABELS_URL, save_to, force)
    else:
        raise Exception("Invalid dataset name! please check it: ", dataset)

if __name__ == '__main__':
    import argparse
    parser = argparse.ArgumentParser('Script for automatically downloading datasets')
    parser.add_argument("--dataset", default='mnist', choices=['mnist', 'fashion-mnist', 'smallNORB'])
    save_to = os.path.join('data', 'mnist')
    parser.add_argument("--save_to", default=save_to)
    parser.add_argument("--force", default=False, type=bool)
    args = parser.parse_args()
    start_download(args.dataset, args.save_to, args.force)
