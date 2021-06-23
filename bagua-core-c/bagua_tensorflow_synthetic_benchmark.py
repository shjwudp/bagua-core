import argparse
from tensorflow.keras import applications
import timeit
import numpy as np
import os
import tensorflow as tf
import bagua.torch_api as bagua
from bagua.bagua_define import DistributedAlgorithm

from . import bagua_tf

def bagua_init(opt: tf.train.Optimizer, distributed_algorithm=DistributedAlgorithm.GradientAllReduce):
    bagua_opt = bagua_tf.DistributeOptimizer(
        opt, distributed_algorithm=distributed_algorithm)
    return bagua_opt, None


# Benchmark settings
parser = argparse.ArgumentParser(description='TensorFlow Synthetic Benchmark',
                                 formatter_class=argparse.ArgumentDefaultsHelpFormatter)
parser.add_argument('--fp16-allreduce', action='store_true', default=False,
                    help='use fp16 compression during allreduce')

parser.add_argument('--model', type=str, default='ResNet50',
                    help='model to benchmark')
parser.add_argument('--batch-size', type=int, default=32,
                    help='input batch size')

parser.add_argument('--num-warmup-batches', type=int, default=10,
                    help='number of warm-up batches that don\'t count towards benchmark')
parser.add_argument('--num-batches-per-iter', type=int, default=10,
                    help='number of batches per benchmark iteration')
parser.add_argument('--num-iters', type=int, default=10,
                    help='number of benchmark iterations')

parser.add_argument('--eager', action='store_true', default=False,
                    help='enables eager execution')
parser.add_argument('--no-cuda', action='store_true', default=False,
                    help='disables CUDA training')
parser.add_argument('--use-adasum', action='store_true', default=False,
                    help='use adasum algorithm to do reduction')

args = parser.parse_args()
args.cuda = not args.no_cuda

bagua.init_process_group()

# Horovod: pin GPU to be used to process local rank (one GPU per process)
config = tf.ConfigProto()
config.gpu_options.allow_growth = True
config.gpu_options.visible_device_list = str(bagua.get_local_rank())

if args.eager:
    tf.enable_eager_execution(config)

# Set up standard model.
model = getattr(applications, args.model)(weights=None)

opt = tf.train.GradientDescentOptimizer(0.01)

opt, _ = bagua_init(opt)

init = tf.global_variables_initializer()

data = tf.random_uniform([args.batch_size, 224, 224, 3])
target = tf.random_uniform(
    [args.batch_size, 1], minval=0, maxval=999, dtype=tf.int64)


def loss_function():
    probs = model(data, training=True)
    return tf.losses.sparse_softmax_cross_entropy(target, probs)


def log(s, nl=True):
    if bagua.get_rank() != 0:
        return
    print(s, end='\n' if nl else '')


log('Model: %s' % args.model)
log('Batch size: %d' % args.batch_size)
device = 'GPU' if args.cuda else 'CPU'
log('Number of %ss: %d' % (device, bagua.get_world_size()))


def run(benchmark_step):
    # Warm-up
    log('Running warmup...')
    timeit.timeit(benchmark_step, number=args.num_warmup_batches)

    # Benchmark
    log('Running benchmark...')
    img_secs = []
    for x in range(args.num_iters):
        time = timeit.timeit(benchmark_step, number=args.num_batches_per_iter)
        img_sec = args.batch_size * args.num_batches_per_iter / time
        log('Iter #%d: %.1f img/sec per %s' % (x, img_sec, device))
        img_secs.append(img_sec)

    # Results
    img_sec_mean = np.mean(img_secs)
    img_sec_conf = 1.96 * np.std(img_secs)
    log('Img/sec per %s: %.1f +-%.1f' % (device, img_sec_mean, img_sec_conf))
    log('Total img/sec on %d %s(s): %.1f +-%.1f' %
        (bagua.get_world_size(), device, bagua.get_world_size() * img_sec_mean, bagua.get_world_size() * img_sec_conf))


if tf.executing_eagerly():
    with tf.device(device):
        run(lambda: opt.minimize(loss_function, var_list=model.trainable_variables))
else:
    with tf.Session(config=config) as session:
        init.run()

        loss = loss_function()
        train_opt = opt.minimize(loss)
        writer = tf.summary.FileWriter('./graphs', graph=session.graph)
        run(lambda: session.run(train_opt))
        writer.flush()
        writer.close()
