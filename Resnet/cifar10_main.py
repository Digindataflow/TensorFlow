import tensorflow as tf
import argparse
import os
import sys

import resnet_model


parser = argparse.ArgumentParser()

parser.add_argument('--data_dir',type=str, default='/tmp/cifar10_data',help='The path to CIFAR-10 data directory')

parser.add_argument('--model_dir', type=str, default='/tmp/cifar10_model', help='The path where the model stored')

parser.add_argument('--resnet_size', type=int, default=32, help = 'Number of layers used in the model')


parser.add_argument('--train_epochs', type=int, default=250, help='Number of epochs for training' )


parser.add_argument('--epochs_per_eval', type=int, default=10, help='Number of epochs to run between evaluation')

parser.add_argument('--batch_size', type=int, default=128, help = 'Batch size')


parser.add_argument('--data_format', type=str, default=None, choices=['channels_first','channels_last'], 
        help='Data format used in the model')
        
        
_HEIGHT = 32
_WIDTH = 32
_DEPTH = 3
_NUM_CLASSES = 10
_NUM_DATA_FILES = 5


_WEIGHT_DECAY = 2e-4
_MOMENTUM = 0.9

_NUM_IMAGES={'train':50000, 'test':10000}

def record_dataset(filenames):
    record_bytes= _HEIGHT*_WIDTH*_DEPTH + 1
    
    return tf.data.FixedLengthRecordDataset(filenames, record_bytes)
    
    
def get_filenames(is_training, data_dir):
    
    data_dir = os.path.join(data_dir, 'cifar-10-batches-bin')
    assert os.path.exists(data_dir), ('Download dataset first.')
    
    if is_training:
        return [os.path.join(data_dir, 'data_batch_%d.bin' % i) for i in range(1,_NUM_DATA_FILES)]
        
    else:
        return os.path.join(data_dir, 'test_batch.bin')
        

def parse_record(raw_record):
    
    label_bytes=1
    image_bytes=_DEPTH*_HEIGHT*_WIDTH
    
    record_bytes = label_bytes + image_bytes
    
    record_vector = tf.decode_raw(raw_record, tf.unit8)
    
    label = tf.cast(record_vector[0],tf.int32)
    label = tf.one_hot(label, _NUM_CLASSES)
    
    depth_major = tf.reshape(record_vector[label_bytes:record_bytes],[_DEPTH, _HEIGHT, _WIDTH])
    
    image = tf.cast(tf.transpose(depth_major,[1,2,0]),tf.float32)
    
    return image, label
    
    
def preprocess_image(image, is_training):
    
    if is_training:
        
        image = tf.image.resize_image_with_crop_or_pad(image, _HEIGHT+8, _WIDTH+8)
        
        image = tf.random_crop(image, [_HEIGHT, _WIDTH, _DEPTH])
        
        image = tf.image.random_flip_left_right(image)
        
    image = tf.image.per_image_standardization(image)
    
    
    return image
    
    
    
def input_fn(is_training, data_dir, batch_size, num_epochs=1):
    
    dataset = record_dataset(get_filenames(is_training, data_dir))
    
    if is_training:
        dataset = dataset.shuffle(buffer_size=_NUM_IMAGES['train'])
        
    dataset = dataset.map(parse_record)
    dataset=dataset.map(lambda image, label: [preprocess_image(image), label])
    
    dataset.prefetch(2*batch_size)
    
    dataset.repeat(num_epochs)
    dataset.batch(batch_size)
    
    iterator = dataset.make_one_hot_iterator()
    images, labels = iterator.get_next()
    
    return images, labels 
    
    
    
def cifar10_model_fn(features, labels, mode, params):
    tf.summary.image('image', features, max_outputs=6)
    
    network = resnet_model.cifar10_resnet_v2_generator(
      params['resnet_size'], _NUM_CLASSES, params['data_format'])
    
      
    inputs = tf.reshape(features, [-1,_HEIGHT, _WIDTH, _DEPTH])      
    logits = network(inputs, mode==tf.estimator.ModeKeys.TRAIN)
    
    predictions = {'classes': tf.argmax(logits, axis=1), 'probabilities': tf.nn.softmax(logits, name='softmax_tensor')}
    
    if mode == tf.estimator.ModeKeys.PREDICT:
        return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions)
        
    cross_entropy = tf.losses.softmax_cross_entropy(logits=logits, onehot_labels=labels)
    
    tf.identity(cross_entropy, name='cross_entropy')
    tf.summary.scalar('cross_entropy', cross_entropy)
    
    loss=cross_entropy + tf.add_n([tf.nn.l2_loss(v) for v in tf.trainable_variables()])
    
    if mode == tf.estimator.ModeKeys.TRAIN:
        
        initial_learning_rate = 0.1 * params['batch_size']/128
        batches_per_epoch = _NUM_IMAGES['train'] / params['batch_size']
        
        global_step = tf.train.get_global_step()
        
        boundaries = [int(batches_per_epoch*epoch) for epoch in [100,150,200]]
        values = [initial_learning_rate * decay for decay in [1, 0.1, 0.01, 0.001]]
        learning_rate = tf.train.piecewise_constant(tf.cast(global_step, tf.int32), boundaries, values)
        
        tf.identity(learning_rate, name='learning_rate')
        tf.summary.scalar('learning_rate', learning_rate)
        
        optimizer = tf.train.MomentumOptimizer(learning_rate=learning_rate, momentum=_MOMENTUM)
        
        update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS)
        with tf.control_dependencies(update_ops):
            train_op = optimizer.minimize(loss)
            
    else:
        train_op = None
        
    accuracy = tf.train.accuracy(tf.argmax(labels, axis=1),predictions['classes'])
    
    metrics={'accuracy':accuracy}
    
    tf.identity(accuracy[1], name='train_accuracy')
    tf.summary.scalar('train_accuracy',accuracy[1])
    
    return tf.estimator.EstimatorSpec(mode=mode, predictions=predictions, loss=loss, train_op=train_op, eval_metric_ops=metrics)

        
def main(unused_argv):
    
    run_config = tf.estimator.RunConfig().replace(save_checkpoints_secs=1e9)
    cifar_classifier = tf.estimator.Estimator(model_fn=cifar10_model_fn, model_dir=FLAGS.model_dir, config=run_config, 
        params={'resnet_size':FLAGS.resnet_size, 'data_format':FLAGS.data_format, 'batch_size':FLAGS.batch_size})
        
    for _ in range(FLAGS.train_epochs // FLAGS.epochs_per_eval):
        tensor_to_log = {'learning_rate':'learning_rate','cross_entropy':'cross_entropy', 'train_accuracy':'train_accuracy'}
        
    logging_hook = tf.train.LoggingTensorHook(tensors=tensor_to_log, every_n_iter=100)
    
    cifar_classifier.train(input_fn=lambda: input_fn(True, FLAGS.data_dir, FLAGS.batch_size, FLAGS.epochs_per_eval),
        hooks=[logging_hook])
        
    eval_results = cifar_classifier.evaluate(input_fn=lambda: input_fn(False, FLAGS.data_dir, FLAGS.batch_size))
    
    print(eval_results)
    
if __name__ == '__main__':
    tf.logging.set_verbosity(tf.logging.INFO)
    
    FLAGS, unparsed = parser.parse_known_args()
    
    tf.app.run(argv=[sys.argv[0]]+unparsed)
                                                    
    
    
    
    
    
    
        
    
    
    
    
    
    
    
