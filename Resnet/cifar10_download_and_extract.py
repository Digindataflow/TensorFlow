import tensorflow as tf

import argparse
import os
import sys
import tarfile

import urllib

DATA_URL = 'https://www.cs.toronto.edu/~kriz/cifar-10-binary.tar.gz'

parser = argparse.ArgumentParser()

parser.add_argument('--data_dir', type=str, default='/tmp/cifar10_data', help='directory for data')

def main(unused_argv):
    
    if not os.path.exists(FLAGS.data_dir):
        os.makedirs(FLAGS.data_dir)
        
    filename = DATA_URL.split('/')[-1]
    filepath = os.path.join(FLAGS.data_dir, filename)
    
    if not os.path.exists(filepath):
        def _progress(count, block_size, total_size):
            sys.stdout.write('\r>> Downloading %s %.1f%%' %(filename, 100.0*count*block_size/total_size))
            sys.stdout.flush()
        
        filepath, _ = urllib.request.urlretrieve(DATA_URL, filepath, _progress)
        
        print()
        statinfo = os.stat(filepath)
        
        print('Successfully downloaded', filename, statinfo.st_size, 'bytes.')
        
    tarfile.open(filepath, 'r:gz').extractall(FLAGS.data_dir)
    

if __name__ == '__main__':
    FLAGS, unparsed = parser.parse_known_args()
    tf.app.run(main=main, argv=[sys.argv[0]]+unparsed)
        
        
    