from collections import defaultdict
from itertools import groupby

import tensorflow as tf
import glob

image_filenames = glob.glob('C:\\Users\\Lei\\regression\\TensorFlow\\Images\\n02*\\*.jpg')

train_data = defaultdict(list)
test_data = defaultdict(list)

image_filename_with_breed = map(lambda filename: (filename.split('\\')[2], filename), image_filenames)

for dog_breed, breed_images in groupby(image_filename_with_breed, lambda x:x[0]):
    for i, breed_image in enumerate(breed_images):
        if i%5==0:
            test_data[dog_breed].append(breed_image[1])
        else:
            train_data[dog_breed].append(breed_image[1])
            
            


def write_records_file(dataset, record_location):
    writer = None
    sess = tf.Session()
    current_index=0
    for breed, breed_images in dataset.items():
        for breed_image in breed_images:
            if current_index%100 == 0:
                if writer:
                    writer.close()
                
                record_filename = '{record_location}-{current_index}.tfrecords'.format(record_location=record_location, current_index=current_index)
                writer = tf.python_io.TFRecordWriter(path=record_filename)
            
            current_index +=1
            image_file = tf.read_file(breed_image)
            try:
                image = tf.image.decode_jpeg(image_file)
            except: 
                print(breed_image)
                continue
                
            grayscale_image = tf.image.rgb_to_grayscale(image)
            resized_image = tf.image.resize_images(grayscale_image, 250, 151)
            
            image_bytes = sess.run(tf.cast(resized_image),tf.uint8).tobytes()
            
            image_label = breed.encode('utf-8')
            
            example = tf.train.Example(features = tf.train.Features(feature = {
                'label': tf.train.Feature(bytes_list=tf.train.BytesList(value=[image_label])) ,
                'image': tf.train.Feature(bytes_list = tf.train.BytesList(value=[image_bytes]))
            }))
            
            writer.write(example.SerializeToString())
        
    writer.close()
    
    
write_records_file(train_data, '\\train_image')
                