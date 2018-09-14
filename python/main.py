from __future__ import print_function

import keras
from keras import backend as K
from keras.preprocessing.image import ImageDataGenerator
from keras.regularizers import l2
from keras.utils import multi_gpu_model

from data import load_data
from lsuvinit import LSUVinit
from models import make_model, make_wrn
from optimizer import make_optimizer

from argparse import ArgumentParser

import math
import numpy as np
import tensorflow as tf

p = ArgumentParser()

p.add_argument('-a', '--activation', type=str, default='blu')
p.add_argument('-b', '--batch_size', type=int, default=128)
p.add_argument('-c', '--cpu_cores', type=int, default=1)
p.add_argument('-d', '--depth', type=int, default=16)
p.add_argument('-e', '--lr_epochs', type=int, nargs='*', default=[300])
p.add_argument('-g', '--multi_gpu', action='store_true')
p.add_argument('-k', '--k', type=int, default=8)
p.add_argument('-l', '--lr_values', type=float, nargs='*', default=[2e-1, 1e-6])
p.add_argument('-m', '--model', type=str, default='wrn')
p.add_argument('-o', '--optimizer', type=str, default='sgd')
p.add_argument('-r', '--l2_regularization', type=float, default=5e-4)
p.add_argument('-s', '--seed', type=int, default=0)

p.add_argument('--alpha', type=float, nargs='?')
p.add_argument('--beta', type=float, nargs='?')

p.add_argument('--no_batch_norm', dest='batch_norm', action='store_false')
p.add_argument('--no_cos_lr', dest='cos_lr', action='store_false')
p.add_argument('--no_dropout', dest='dropout', action='store_false')
p.add_argument('--no_lsuv_init', dest='lsuv_init', action='store_false')
p.add_argument('--no_save', dest='save', action='store_false')
p.add_argument('--no_share_params', dest='share_params', action='store_false')

p.add_argument('--aplu_segments', type=int, default=5)
p.add_argument('--blu_reg', type=float, default=0)
p.add_argument('--dataset', type=str, default='cifar10')
p.add_argument('--extreme_augmentation', action='store_true')
p.add_argument('--loss', type=str, default='categorical_crossentropy')
p.add_argument('--momentum', type=float, default=0.9)
p.add_argument('--save_dir', type=str, default='../results')
p.add_argument('--save_file', type=str, default=None)

args = p.parse_args()

args.lr_values += [args.lr_values[-1]] * (len(args.lr_epochs) - len(args.lr_values))
if args.cos_lr:
    args.lr_values = [args.lr_values[0], args.lr_values[-1]]
    args.lr_epochs = [sum(args.lr_epochs), 0]

print(args)

if args.seed >= 0:
    from numpy.random import seed
    seed(args.seed)
    from tensorflow import set_random_seed
    set_random_seed(args.seed)

if args.cpu_cores > 0:
    config = tf.ConfigProto(intra_op_parallelism_threads=args.cpu_cores, inter_op_parallelism_threads=args.cpu_cores)
    session = tf.Session(config=config)
    K.set_session(session)

if args.save_file is None:
    model = args.model + ('-{}-{}'.format(args.depth, args.k) if args.model == 'wrn' else '')
    args.save_file = args.dataset + '_' + model + '_' + args.activation + '.h5'

(x_train, y_train), (x_test, y_test), num_classes = load_data(args.dataset)
x_train = x_train.astype('float32')
x_test = x_test.astype('float32')
y_train = keras.utils.to_categorical(y_train, num_classes)
y_test = keras.utils.to_categorical(y_test, num_classes)

if args.extreme_augmentation:
    x_train = np.pad(x_train, ((0,), (48,), (48,), (0,)), 'constant')
    x_test = np.pad(x_test, ((0,), (48,), (48,), (0,)), 'constant')

    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.2,
        height_shift_range=0.2,
        horizontal_flip=True,
        rotation_range=45,
        shear_range=0.5,
        zoom_range=(0.5, 2))
else:
    datagen = ImageDataGenerator(
        featurewise_center=True,
        featurewise_std_normalization=True,
        width_shift_range=0.125,
        height_shift_range=0.125,
        horizontal_flip=True)

datagen.fit(x_train)
x_test = datagen.standardize(x_test)

def make_model_impl():
    return make_model(args.model,
                      input_shape=x_train.shape[1:],
                      output_size=num_classes,
                      args=args)

# ensure that the model lives on the cpu (for multi-gpu training)
if args.multi_gpu:
    with tf.device('/cpu:0'):
        model = make_model_impl()
else:
    model = make_model_impl()

print(model.summary())
if args.lsuv_init:
    model = LSUVinit(model, x_train[:args.batch_size,:,:,:])

optimizer = make_optimizer(args.optimizer, lr=args.lr_values[0], momentum=args.momentum)

orig_model = model
if args.multi_gpu:
    model = multi_gpu_model(model)

model.compile(loss=args.loss, optimizer=optimizer, metrics=['accuracy'])

def check_params():
    for layer in model.layers:
        if layer.__class__.__name__ in ['BLU', 'PReLU', 'SoftExp', 'ScaledReLU']:
            for weight in layer.weights:
                if 'alpha' in weight.name or 'beta' in weight.name:
                    value = K.get_value(weight)
                    print('\n{}: min={}, max={}, mean={}, stddev={}'.format(
                        weight.name,
                        value.min(),
                        value.max(),
                        value.mean(),
                        value.std()))

iteration_count = args.lr_epochs[0]
iteration_index = 0.
lr_max = args.lr_values[0]
lr_min = args.lr_values[-1]
def update_lr():
    global iteration_index
    K.set_value(model.optimizer.lr, lr_min + 0.5 * (lr_max - lr_min) * (1 + math.cos(math.pi * iteration_index / iteration_count)))
    iteration_index += 1

param_checker = keras.callbacks.LambdaCallback(on_epoch_end=lambda epoch, logs: check_params())
lr_updater = keras.callbacks.LambdaCallback(on_epoch_end=lambda batch, logs: update_lr())

for lr, epochs in zip(args.lr_values, args.lr_epochs):
    if epochs == 0:
        continue

    if args.cos_lr:
        print('training for {} epochs with cosine annealing lr from {} to {}'.format(epochs, lr_max, lr_min))
    else:
        print('training for {} epochs with lr={}'.format(epochs, lr))
        K.set_value(model.optimizer.lr, lr)

    callbacks = [keras.callbacks.TerminateOnNaN(), param_checker]
    if args.cos_lr:
        callbacks.append(lr_updater)

    model.fit_generator(datagen.flow(x_train, y_train, batch_size=args.batch_size),
        steps_per_epoch=len(x_train) / args.batch_size,
        epochs=epochs,
        validation_data=(x_test, y_test),
        callbacks=callbacks)

if args.save:
    orig_model.save(args.save_dir + '/' + args.save_file)
