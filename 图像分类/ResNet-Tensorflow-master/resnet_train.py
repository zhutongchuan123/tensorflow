#encoding:utf-8
from resnet import * 
import tensorflow as tf
import sys
import os

MOMENTUM = 0.9

FLAGS = tf.app.flags.FLAGS
tf.app.flags.DEFINE_string('train_dir', '',
                           """Directory where to write event logs """
                           """and checkpoint.""")
tf.app.flags.DEFINE_string('model_dir', '',
                           "model saved directory")
tf.app.flags.DEFINE_string('ckpt_file', 'ResNet-L50.ckpt',
                           "checkpoint file saved path ")
tf.app.flags.DEFINE_float('learning_rate', 0.01, "learning rate.")
tf.app.flags.DEFINE_integer('batch_size', 32, "batch size")
tf.app.flags.DEFINE_integer('max_steps', 20000, "max steps")
tf.app.flags.DEFINE_boolean('resume', True,
                            'resume from latest saved state')
tf.app.flags.DEFINE_boolean('minimal_summaries', True,
                            'produce fewer summaries to save HD space')


def top_k_error(predictions, labels, k):
    batch_size = float(FLAGS.batch_size) #tf.shape(predictions)[0]
    in_top1 = tf.to_float(tf.nn.in_top_k(predictions, labels, k=1))
    num_correct = tf.reduce_sum(in_top1)
    return (batch_size - num_correct) / batch_size

def eval(logits, images, labels):
    pass
def train(is_training, logits, images, labels):
    run_dir = '%s/run-%d' %(FLAGS.train_dir, os.getpid())
    global_step = tf.get_variable('global_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)
    val_step = tf.get_variable('val_step', [],
                                  initializer=tf.constant_initializer(0),
                                  trainable=False)

    loss_ = loss(logits, labels)
    predictions = tf.nn.softmax(logits)

    top1_error = top_k_error(predictions, labels, 1)


    # loss_avg
    ema = tf.train.ExponentialMovingAverage(MOVING_AVERAGE_DECAY, global_step)
    tf.add_to_collection(UPDATE_OPS_COLLECTION, ema.apply([loss_]))
    tf.summary.scalar('loss_avg', ema.average(loss_))

    # validation stats
    ema = tf.train.ExponentialMovingAverage(0.9, val_step)
    val_op = tf.group(val_step.assign_add(1), ema.apply([top1_error]))
    top1_error_avg = ema.average(top1_error)
    tf.summary.scalar('val_top1_error_avg', top1_error_avg)

    tf.summary.scalar('learning_rate', FLAGS.learning_rate)

    opt = tf.train.MomentumOptimizer(FLAGS.learning_rate, MOMENTUM)
    grads = opt.compute_gradients(loss_)
    for grad, var in grads:
        if grad is not None and not FLAGS.minimal_summaries:
            tf.histogram_summary(var.op.name + '/gradients', grad)
    apply_gradient_op = opt.apply_gradients(grads, global_step=global_step)

    if not FLAGS.minimal_summaries:
        # Display the training images in the visualizer.
        tf.image_summary('images', images)

        for var in tf.trainable_variables():
            tf.histogram_summary(var.op.name, var)

    batchnorm_updates = tf.get_collection(UPDATE_OPS_COLLECTION)
    batchnorm_updates_op = tf.group(*batchnorm_updates)
    train_op = tf.group(apply_gradient_op, batchnorm_updates_op)

    resnet_variables = tf.trainable_variables()
    saver1 = tf.train.Saver(resnet_variables)
    # 以下变量不需要从训练好的模型中恢复
    # resnet_variables = filter(lambda x:"val_step" not in x.name, resnet_variables)
    # resnet_variables = filter(lambda x:"Momentum" not in x.name, resnet_variables)
    # resnet_variables = filter(lambda x:"biased" not in x.name, resnet_variables)
    # resnet_variables = filter(lambda x:"local_step" not in x.name, resnet_variables)
    # resnet_variables = filter(lambda x:"global_step" not in x.name, resnet_variables)
    # resnet_variables = filter(lambda x:"ExponentialMovingAverage" not in x.name, resnet_variables)
    resnet_variables = filter(lambda x:"fc" not in x.name, resnet_variables)
    
    saver2 = tf.train.Saver(resnet_variables)

    summary_op = tf.summary.merge_all()

    init = tf.global_variables_initializer()

    sess = tf.Session(config=tf.ConfigProto(log_device_placement=False))
    sess.run(init)
    tf.train.start_queue_runners(sess=sess)

    summary_writer = tf.summary.FileWriter(FLAGS.train_dir, sess.graph)

    if FLAGS.resume:
        # resnet_variables = tf.get_collection(tf.GraphKeys.VARIABLES,scope="scale1")
        # resnet_variables.remove("val_step")
        # restorer = tf.train.Saver(resnet_variables)
        saver2.restore(sess, os.path.join(FLAGS.model_dir, FLAGS.ckpt_file))
        """
        latest = tf.train.latest_checkpoint(FLAGS.model_dir)
        if not latest:
            print "No checkpoint to continue from in", FLAGS.model_dir
            sys.exit(1)
        print "resume", latest
        saver.restore(sess, latest)
        """
    for x in xrange(FLAGS.max_steps + 1):
        start_time = time.time()

        step = sess.run(global_step)
        i = [train_op, loss_]

        write_summary = step % 100 and step > 1
        if write_summary:
            i.append(summary_op)

        o = sess.run(i,{is_training:True})

        loss_value = o[1]

        duration = time.time() - start_time

        assert not np.isnan(loss_value), 'Model diverged with loss = NaN'

        if step % 5 == 0:
            examples_per_sec = FLAGS.batch_size / float(duration)
            format_str = ('step %d, loss = %.2f (%.1f examples/sec; %.3f '
                          'sec/batch)')
            print(format_str % (step, loss_value, examples_per_sec, duration))

        if write_summary:
            summary_str = o[2]
            summary_writer.add_summary(summary_str, step)

        # Save the model checkpoint periodically.
        if step > 1 and step % 100 == 0:
            checkpoint_path = os.path.join(run_dir, 'model.ckpt')
            saver1.save(sess, checkpoint_path, global_step=global_step)

        # Run validation periodically
        if step > 1 and step % 100 == 0:
            _, top1_error_value = sess.run([val_op, top1_error], { is_training: False })
            print('Validation top1 error %.2f' % top1_error_value)



