#encoding:utf-8
from resnet import * 
import tensorflow as tf
import sys
import csv
import os
from resnet import inference
from utils import ProgressBar, ImageCoder, make_multi_image_batch
from image_processing import image_preprocessing
import math
import random
MAX_BATCH_SZ = 32
RESIZE_FINAL = 224
FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('data_dir', '../val/4212_1025/test_imgs/',
                           """test images Directory """)
tf.app.flags.DEFINE_string('model_dir', '',
                           "model saved directory")
tf.app.flags.DEFINE_string('ckpt_file', 'model.ckpt-401',
                           "checkpoint file saved path ")
tf.app.flags.DEFINE_string('target','result_401.csv',
                            'CSV file containing the filename processed along with best guess and score')
label_list = ['side','frontal']
def main(argv=None):
    with tf.Session() as sess:
        data_dir = FLAGS.data_dir
        files = [os.path.join(data_dir, item) for item in os.listdir(data_dir) ]
        # files = random.sample(files,  800)
        images = tf.placeholder(tf.float32, [None,RESIZE_FINAL,RESIZE_FINAL,3])
        logits = inference(images, False,
                  num_classes=2,
                  num_blocks=[3, 4, 6, 3],  # defaults to 50-layer network
                  use_bias=False, # defaults to using batch norm
                  bottleneck=True)
        init = tf.global_variables_initializer()
        resnet_variables = tf.global_variables()
        saver = tf.train.Saver(resnet_variables)
        saver.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.ckpt_file))

        softmax_output = tf.nn.softmax(logits)
        if FLAGS.target:
            print('Creating output file %s' %FLAGS.target)
            output = open(os.path.join(FLAGS.data_dir,FLAGS.target), 'w')
            writer = csv.writer(output)
            writer.writerow(('file', 'label', 'score'))

        num_batches = int(math.ceil(len(files)) / MAX_BATCH_SZ)
        pg = ProgressBar(num_batches)
        # try:
        for j in range(num_batches):
            start_offset = j * MAX_BATCH_SZ
            end_offset = min((j+1)*MAX_BATCH_SZ, len(files))

            batch_image_files = files[start_offset:end_offset]
            images_ = []
            for file in batch_image_files:
                print file
                image_buffer = tf.read_file(file)
                bbox = []
                image = image_preprocessing(image_buffer, [], False)
                images_.append(image)
            image_batch = tf.stack(images_)
            batch_results = sess.run(softmax_output, feed_dict={images:image_batch.eval()})
            batch_sz = batch_results.shape[0]

            for i in range(batch_sz):
                output_i = batch_results[i]
                best_i = np.argmax(output_i)

                best_choice = (label_list[best_i], output_i[best_i])
                if writer is not None:
                    f = batch_image_files[i]
                    writer.writerow((f, best_choice[0], '%.2f' %best_choice[1]))
            pg.update()
        pg.done()
        # except Exception as e:
        #     print(e)
        #     print('Failed to run all images')

if __name__ == "__main__":
    tf.app.run()




